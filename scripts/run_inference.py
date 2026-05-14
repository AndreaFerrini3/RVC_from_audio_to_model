"""
End-to-end inference: WAV input(s) -> converted WAV(s) using a trained RVC model.

Usage:
    python scripts/run_inference.py --model_name my_voice --input path/to/audio.wav
    python scripts/run_inference.py --model_name my_voice --input path/to/folder/ --transpose 2

Resolves automatically:
    - Model weights:  mangio/weights/<model_name>.pth   (final exported .pth)
    - Faiss index:    mangio/logs/<model_name>/added_IVF*_<model_name>_<version>.index
                      (built from features if missing and --build_index is set;
                       skipped if not found and index_rate=0)

Delegates the actual conversion to Mangio's infer_batch_rvc.py (cwd=mangio/).
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
MANGIO_DIR = REPO_ROOT / "mangio"


def run(cmd, cwd=None, env=None):
    label = cwd.name if cwd else "."
    print(f"\n[{label}] >>> {cmd}\n")
    result = subprocess.run(cmd, shell=True, cwd=str(cwd or REPO_ROOT), env=env)
    if result.returncode != 0:
        print(f"FAILED (exit {result.returncode}): {cmd}")
        sys.exit(result.returncode)


def check_submodule():
    if not (MANGIO_DIR / "infer_batch_rvc.py").exists():
        print("ERROR: Mangio submodule not initialised.")
        print("Run: git submodule update --init")
        sys.exit(1)


def resolve_model_path(model_name):
    pth = MANGIO_DIR / "weights" / f"{model_name}.pth"
    if not pth.exists():
        print(f"ERROR: model weights not found: {pth}")
        print(f"Train first with: python scripts/run_pipeline.py --model_name {model_name} ...")
        sys.exit(1)
    return pth


def detect_version(model_pth):
    import torch
    cpt = torch.load(str(model_pth), map_location="cpu")
    return cpt.get("version", "v1")


def find_index(model_name):
    log_dir = MANGIO_DIR / "logs" / model_name
    if not log_dir.is_dir():
        return None
    candidates = sorted(log_dir.glob("added_IVF*.index"))
    if not candidates:
        return None
    # Prefer files matching the model name; fall back to first match
    named = [c for c in candidates if model_name in c.name]
    return (named or candidates)[0]


def build_index(model_name, version):
    """Replicates Mangio's index-training routine (infer-web.py train_index)."""
    import numpy as np
    import faiss
    from sklearn.cluster import MiniBatchKMeans

    log_dir = MANGIO_DIR / "logs" / model_name
    feature_dir = log_dir / ("3_feature256" if version == "v1" else "3_feature768")
    if not feature_dir.is_dir() or not any(feature_dir.iterdir()):
        print(f"ERROR: feature dir empty/missing: {feature_dir}")
        print(f"Run training (or at least feature extraction) for {model_name} first.")
        sys.exit(1)

    print(f"[index] aggregating features from {feature_dir}")
    npys = [np.load(str(feature_dir / n)) for n in sorted(os.listdir(feature_dir))]
    big_npy = np.concatenate(npys, 0)

    idx = np.arange(big_npy.shape[0])
    np.random.shuffle(idx)
    big_npy = big_npy[idx]

    if big_npy.shape[0] > 2e5:
        print(f"[index] kmeans {big_npy.shape[0]} -> 10000 centers")
        big_npy = (
            MiniBatchKMeans(
                n_clusters=10000, verbose=False,
                batch_size=256 * max(os.cpu_count() or 4, 4),
                compute_labels=False, init="random",
            )
            .fit(big_npy)
            .cluster_centers_
        )

    np.save(str(log_dir / "total_fea.npy"), big_npy)
    dim = 256 if version == "v1" else 768
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    n_ivf = max(n_ivf, 1)

    index = faiss.index_factory(dim, f"IVF{n_ivf},Flat")
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1

    print(f"[index] training (n_ivf={n_ivf}, dim={dim})")
    index.train(big_npy)
    trained_path = log_dir / f"trained_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{model_name}_{version}.index"
    faiss.write_index(index, str(trained_path))

    print(f"[index] adding vectors")
    batch = 8192
    for i in range(0, big_npy.shape[0], batch):
        index.add(big_npy[i:i + batch])
    added_path = log_dir / f"added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{model_name}_{version}.index"
    faiss.write_index(index, str(added_path))
    print(f"[index] wrote {added_path}")
    return added_path


def stage_inputs(input_arg):
    """Return (staged_dir, cleanup_fn, ordered_basenames).
    Mangio's infer_batch_rvc.py walks a directory and processes every .wav inside.
    For single-file input we copy into a temp dir; for a directory we use it as-is."""
    p = Path(input_arg).resolve()
    if p.is_dir():
        wavs = sorted([f.name for f in p.iterdir() if f.suffix.lower() == ".wav"])
        if not wavs:
            print(f"ERROR: no .wav files in {p}")
            sys.exit(1)
        return p, (lambda: None), wavs
    if p.is_file():
        if p.suffix.lower() != ".wav":
            print(f"ERROR: input must be a .wav file (got {p.suffix})")
            sys.exit(1)
        tmp = Path(tempfile.mkdtemp(prefix="rvc_infer_"))
        shutil.copy2(p, tmp / p.name)
        return tmp, (lambda: shutil.rmtree(tmp, ignore_errors=True)), [p.name]
    print(f"ERROR: input not found: {p}")
    sys.exit(1)


def build_parser():
    p = argparse.ArgumentParser(description="RVC inference — runs Mangio's infer_batch_rvc.py end-to-end")

    p.add_argument("--model_name", required=True,
                   help="Trained model name (looks up mangio/weights/<model_name>.pth)")
    p.add_argument("--inference_dataset", default="inf_dataset_1",
                   help="Subfolder under inference_dataset/ holding the input WAVs "
                        "(default: inf_dataset_1). Output goes to "
                        "inference_results/<model_name>_<inference_dataset>/")
    p.add_argument("--input", default=None,
                   help="Override: explicit WAV file or folder of WAV files. "
                        "If omitted, uses inference_dataset/<inference_dataset>/")
    p.add_argument("--output_dir", default=None,
                   help="Override output folder (default: inference_results/<model_name>_<inference_dataset>/)")

    # Conversion params
    p.add_argument("--transpose", "-k", default=0, type=int,
                   help="Pitch shift in semitones (default: 0). +12 = up one octave")
    p.add_argument("--f0method", default="rmvpe",
                   choices=["pm", "harvest", "crepe", "rmvpe", "mangio-crepe"],
                   help="Pitch extraction method (default: rmvpe)")
    p.add_argument("--index_rate", default=0.66, type=float,
                   help="Faiss index influence 0..1 (default: 0.66)")
    p.add_argument("--filter_radius", default=3, type=int,
                   help="Median filter radius for harvest f0 (default: 3)")
    p.add_argument("--resample_sr", default=0, type=int,
                   help="Resample output to this sr; 0 = keep model sr (default: 0)")
    p.add_argument("--rms_mix_rate", default=1.0, type=float,
                   help="Volume envelope mix: 0=input volume, 1=output volume (default: 1.0)")
    p.add_argument("--protect", default=0.33, type=float,
                   help="Protect voiceless consonants 0..0.5 (default: 0.33)")
    p.add_argument("--crepe_hop_length", default=160, type=int,
                   help="Crepe hop length — only relevant when f0method=crepe/mangio-crepe (default: 160)")

    # Runtime
    p.add_argument("--device", default="cuda:0",
                   help="cuda:N or cpu (default: cuda:0)")
    p.add_argument("--is_half", default="true", choices=["true", "false"],
                   help="fp16 inference (default: true)")

    # Index handling
    p.add_argument("--index_path", default=None,
                   help="Explicit faiss .index path. Default: auto-discover in mangio/logs/<model_name>/")
    p.add_argument("--build_index", action="store_true",
                   help="Build the faiss index from features if missing")
    p.add_argument("--no_index", action="store_true",
                   help="Skip faiss index entirely (forces index_rate=0)")

    return p


def main():
    args = build_parser().parse_args()
    check_submodule()

    py = sys.executable
    model_pth = resolve_model_path(args.model_name)
    version = detect_version(model_pth)
    print(f"[infer] model: {model_pth}  (version={version})")

    # ---- Index resolution ----
    index_path = ""
    if args.no_index:
        args.index_rate = 0.0
        print("[infer] index disabled (--no_index)")
    elif args.index_path:
        if not Path(args.index_path).exists():
            print(f"ERROR: --index_path not found: {args.index_path}")
            sys.exit(1)
        index_path = str(Path(args.index_path).resolve())
    else:
        found = find_index(args.model_name)
        if found:
            index_path = str(found.resolve())
            print(f"[infer] index: {index_path}")
        elif args.build_index:
            index_path = str(build_index(args.model_name, version).resolve())
        else:
            print("[infer] no index found; running with index_rate=0 "
                  "(use --build_index to construct one, or --index_path to point to a file)")
            args.index_rate = 0.0

    # ---- Inputs / outputs ----
    inf_root = REPO_ROOT / "inference_dataset" / args.inference_dataset
    input_path = args.input or str(inf_root)
    if not args.input and not inf_root.is_dir():
        print(f"ERROR: inference dataset folder not found: {inf_root}")
        print(f"Create it and put .wav files inside, or pass --input <path>.")
        print(f"   mkdir -p inference_dataset/{args.inference_dataset}")
        sys.exit(1)

    in_dir, cleanup, _ = stage_inputs(input_path)
    out_dir = Path(args.output_dir).resolve() if args.output_dir \
        else (REPO_ROOT / "inference_results" / f"{args.model_name}_{args.inference_dataset}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[infer] input dir:  {in_dir}")
    print(f"[infer] output dir: {out_dir}")

    # Path passed to Mangio must be relative to mangio/ or absolute — use absolute.
    model_rel = str(model_pth.resolve())

    # ---- Run Mangio batch inference ----
    cmd = (
        f'"{py}" infer_batch_rvc.py '
        f'{args.transpose} '
        f'"{in_dir}" '
        f'"{index_path}" '
        f'{args.f0method} '
        f'"{out_dir}" '
        f'"{model_rel}" '
        f'{args.index_rate} '
        f'{args.device} '
        f'{args.is_half} '
        f'{args.filter_radius} '
        f'{args.resample_sr} '
        f'{args.rms_mix_rate} '
        f'{args.protect} '
        f'{args.crepe_hop_length}'
    )
    try:
        run(cmd, cwd=MANGIO_DIR)
    finally:
        cleanup()

    print(f"\nInference complete. Output: {out_dir}")


if __name__ == "__main__":
    main()
