"""
One-time environment setup. Run this once after cloning.

What it does:
  1. pip install -r mangio/requirements.txt  (torch, fairseq, librosa, ...)
  2. pip install plotly                       (not in Mangio's requirements)
  3. Download pretrained models into mangio/
  4. Run patch_mangio.py

Usage:
    python scripts/setup.py
    python scripts/setup.py --sr 48k          # download 48k pretrained instead of 40k
    python scripts/setup.py --skip_models     # skip model download (already done)
    python scripts/setup.py --skip_deps       # skip pip install (already done)
"""
import argparse
import subprocess
import sys
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


def run(cmd, cwd=None):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=str(cwd or REPO_ROOT))
    if result.returncode != 0:
        print(f"FAILED (exit {result.returncode})")
        sys.exit(result.returncode)


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
    parser.add_argument("--skip_deps",   action="store_true", help="Skip pip install")
    parser.add_argument("--skip_models", action="store_true", help="Skip pretrained model download")
    parser.add_argument("--skip_patch",  action="store_true", help="Skip patch_mangio.py")
    args = parser.parse_args()

    if not MANGIO_DIR.exists() or not (MANGIO_DIR / "infer-web.py").exists():
        print("ERROR: mangio/ submodule not initialised.")
        print("Run: git submodule update --init")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 1. pip install                                                       #
    # ------------------------------------------------------------------ #
    if not args.skip_deps:
        print("\n=== [1/4] Installing Mangio dependencies ===")
        print("NOTE: torch==2.0.0 will be installed. If you need a different")
        print("      CUDA version, install torch manually first and re-run with --skip_deps.")
        run(f'"{sys.executable}" -m pip install -r mangio/requirements.txt')

        print("\n=== [1/4] Installing extra dependencies (plotly) ===")
        run(f'"{sys.executable}" -m pip install "plotly>=5.18.0"')
    else:
        print("\n=== [1/4] Skipping pip install ===")

    # ------------------------------------------------------------------ #
    # 2. Download pretrained models                                        #
    # ------------------------------------------------------------------ #
    if not args.skip_models:
        print(f"\n=== [2/4] Downloading pretrained models (version={args.version}) ===")
        models = MODELS_V2 if args.version == "v2" else MODELS_V1
        for dest_rel, url in {**models, **MODELS_BASE}.items():
            download(dest_rel, url)
    else:
        print("\n=== [2/4] Skipping model download ===")

    # ------------------------------------------------------------------ #
    # 3. Patch Mangio training script                                      #
    # ------------------------------------------------------------------ #
    if not args.skip_patch:
        print("\n=== [3/4] Patching Mangio training script ===")
        run(f'"{sys.executable}" scripts/patch_mangio.py')
    else:
        print("\n=== [3/4] Skipping patch ===")

    # ------------------------------------------------------------------ #
    # Done                                                                 #
    # ------------------------------------------------------------------ #
    print("\n=== [4/4] Setup complete ===")
    print("\nNext step:")
    print("  1. Put raw audio in dataset/audios/")
    print("  2. Run the full pipeline:")
    print("       python scripts/run_pipeline.py --model_name my_voice --sr 40k --epochs 200 --gpus 0")


if __name__ == "__main__":
    main()
