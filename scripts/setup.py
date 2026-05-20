"""
One-time environment setup. Run this once after cloning, inside an activated
Python 3.10 venv.

What it does (in order):
  1. Pin pip<24.1 and setuptools<81 (required by omegaconf 2.0.6 and librosa 0.9.1)
  2. pip install -r mangio/requirements.txt  (PySimpleGUI line filtered — pulled
     from PyPI by the author; only used by mangio/gui_*.py, not the pipeline)
  3. pip install plotly (not in Mangio's requirements)
  4. Optional: reinstall torch 2.7 + CUDA 12.8 wheels for Blackwell GPUs
     (RTX 50xx, compute capability sm_120). Pass --torch_cu128.
  5. Download pretrained models into mangio/
  6. Run patch_mangio.py (patches Mangio train/infer + fairseq checkpoint_utils)

Usage:
    python scripts/setup.py
    python scripts/setup.py --torch_cu128    # RTX 50xx / Blackwell
    python scripts/setup.py --version v1     # download v1 pretrained instead of v2
    python scripts/setup.py --skip_models    # skip model download
    python scripts/setup.py --skip_deps      # skip pip install
"""
import argparse
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
MANGIO_DIR = REPO_ROOT / "mangio"

HF_BASE = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main"

# pretrained_v2 only has 40k variants in the official release
MODELS_V2 = {
    "pretrained_v2/f0D40k.pth": f"{HF_BASE}/pretrained_v2/f0D40k.pth",
    "pretrained_v2/f0G40k.pth": f"{HF_BASE}/pretrained_v2/f0G40k.pth",
    "pretrained_v2/D40k.pth":   f"{HF_BASE}/pretrained_v2/D40k.pth",
    "pretrained_v2/G40k.pth":   f"{HF_BASE}/pretrained_v2/G40k.pth",
}

MODELS_V1 = {
    "pretrained/f0D32k.pth": f"{HF_BASE}/pretrained/f0D32k.pth",
    "pretrained/f0D40k.pth": f"{HF_BASE}/pretrained/f0D40k.pth",
    "pretrained/f0D48k.pth": f"{HF_BASE}/pretrained/f0D48k.pth",
    "pretrained/f0G32k.pth": f"{HF_BASE}/pretrained/f0G32k.pth",
    "pretrained/f0G40k.pth": f"{HF_BASE}/pretrained/f0G40k.pth",
    "pretrained/f0G48k.pth": f"{HF_BASE}/pretrained/f0G48k.pth",
    "pretrained/D32k.pth":   f"{HF_BASE}/pretrained/D32k.pth",
    "pretrained/D40k.pth":   f"{HF_BASE}/pretrained/D40k.pth",
    "pretrained/D48k.pth":   f"{HF_BASE}/pretrained/D48k.pth",
    "pretrained/G32k.pth":   f"{HF_BASE}/pretrained/G32k.pth",
    "pretrained/G40k.pth":   f"{HF_BASE}/pretrained/G40k.pth",
    "pretrained/G48k.pth":   f"{HF_BASE}/pretrained/G48k.pth",
}

MODELS_BASE = {
    "hubert_base.pt": f"{HF_BASE}/hubert_base.pt",
    "rmvpe.pt":       f"{HF_BASE}/rmvpe.pt",
}

# Packages excluded from mangio/requirements.txt at install time.
# Reason: PySimpleGUI 4.x was pulled from PyPI (author went commercial).
# Only used by mangio/gui_*.py — not by the training/inference pipeline.
EXCLUDED_PACKAGES = {"pysimplegui"}

# Long timeout for slow networks / large wheels (gradio ~20MB, torch ~700MB).
PIP_ENV = {**os.environ, "PIP_DEFAULT_TIMEOUT": "300"}


def run(cmd, cwd=None, env=None):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=str(cwd or REPO_ROOT), env=env or PIP_ENV)
    if result.returncode != 0:
        print(f"FAILED (exit {result.returncode})")
        sys.exit(result.returncode)


def filtered_requirements(src: Path) -> Path:
    """Write a copy of `src` with EXCLUDED_PACKAGES removed. Returns temp path."""
    lines_out = []
    dropped = []
    for line in src.read_text(encoding="utf-8").splitlines():
        pkg = line.strip().split("==")[0].split(">=")[0].split("<=")[0].split("~=")[0].lower()
        if pkg in EXCLUDED_PACKAGES:
            dropped.append(line.strip())
            continue
        lines_out.append(line)
    if dropped:
        print(f"  filtered out: {', '.join(dropped)}")
    fd, tmp = tempfile.mkstemp(prefix="mangio-req-", suffix=".txt")
    os.close(fd)
    Path(tmp).write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    return Path(tmp)


def download(dest_rel: str, url: str):
    dest = MANGIO_DIR / dest_rel
    if dest.exists():
        print(f"  skip (exists): {dest_rel}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading: {dest_rel}")

    def progress(count, block_size, total):
        if total > 0:
            pct = min(count * block_size * 100 // total, 100)
            print(f"\r    {pct}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, dest, reporthook=progress)
        print()
    except Exception as e:
        print(f"\n  ERROR downloading {url}: {e}")
        if dest.exists():
            dest.unlink()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v2", choices=["v1", "v2"],
                        help="RVC model version to download pretrained for (default: v2)")
    parser.add_argument("--torch_cu128", action="store_true",
                        help="Reinstall torch 2.7 + CUDA 12.8 wheels (required for "
                             "RTX 50xx / Blackwell, compute capability sm_120)")
    parser.add_argument("--skip_deps",   action="store_true", help="Skip pip install")
    parser.add_argument("--skip_models", action="store_true", help="Skip pretrained model download")
    parser.add_argument("--skip_patch",  action="store_true", help="Skip patch_mangio.py")
    args = parser.parse_args()

    if not MANGIO_DIR.exists() or not (MANGIO_DIR / "infer-web.py").exists():
        print("ERROR: mangio/ submodule not initialised.")
        print("Run: git submodule update --init")
        sys.exit(1)

    pip = f'"{sys.executable}" -m pip'

    # ------------------------------------------------------------------ #
    # 1. Pin build tools BEFORE anything else                              #
    # ------------------------------------------------------------------ #
    if not args.skip_deps:
        print("\n=== [1/5] Pinning pip<24.1 and setuptools<81 ===")
        print("      pip<24.1   : omegaconf 2.0.6 has malformed metadata that pip 24.1+ rejects")
        print("      setuptools<81 : librosa 0.9.1 still imports pkg_resources (removed in 81)")
        run(f'{pip} install "pip<24.1" "setuptools<81"')

        # ------------------------------------------------------------------ #
        # 2. Install Mangio requirements (filtered)                           #
        # ------------------------------------------------------------------ #
        print("\n=== [2/5] Installing Mangio dependencies ===")
        print("      torch==2.0.0 will be installed (CPU/CUDA 11.7 wheel).")
        print("      For RTX 50xx (Blackwell) re-run with --torch_cu128.")
        tmp_req = filtered_requirements(MANGIO_DIR / "requirements.txt")
        try:
            run(f'{pip} install -r "{tmp_req}"')
        finally:
            tmp_req.unlink(missing_ok=True)

        print("\n=== [2/5] Installing extra dependencies (plotly) ===")
        run(f'{pip} install "plotly>=5.18.0"')

        # ------------------------------------------------------------------ #
        # 3. Optional: torch cu128 for Blackwell                              #
        # ------------------------------------------------------------------ #
        if args.torch_cu128:
            print("\n=== [3/5] Reinstalling torch 2.7 + CUDA 12.8 (Blackwell) ===")
            print("      Required for RTX 50xx (sm_120). torch 2.0 only ships kernels up to sm_86.")
            run(f'{pip} install --upgrade --force-reinstall '
                f'torch==2.7.0 torchaudio==2.7.0 '
                f'--index-url https://download.pytorch.org/whl/cu128')
        else:
            print("\n=== [3/5] Skipping torch cu128 (pass --torch_cu128 for RTX 50xx) ===")
    else:
        print("\n=== [1-3/5] Skipping pip install ===")

    # ------------------------------------------------------------------ #
    # 4. Download pretrained models                                        #
    # ------------------------------------------------------------------ #
    if not args.skip_models:
        print(f"\n=== [4/5] Downloading pretrained models (version={args.version}) ===")
        models = MODELS_V2 if args.version == "v2" else MODELS_V1
        for dest_rel, url in {**models, **MODELS_BASE}.items():
            download(dest_rel, url)
    else:
        print("\n=== [4/5] Skipping model download ===")

    # ------------------------------------------------------------------ #
    # 5. Patch Mangio + fairseq                                            #
    # ------------------------------------------------------------------ #
    if not args.skip_patch:
        print("\n=== [5/5] Patching Mangio + fairseq ===")
        run(f'"{sys.executable}" scripts/patch_mangio.py')
    else:
        print("\n=== [5/5] Skipping patch ===")

    print("\n=== Setup complete ===")
    print("\nNext step:")
    print("  1. Put raw audio in dataset/<name>/audios/")
    print("  2. Run the full pipeline:")
    print("       python scripts/run_pipeline.py --model_name my_voice --sr 40k --epochs 200 --gpus 0")


if __name__ == "__main__":
    main()
