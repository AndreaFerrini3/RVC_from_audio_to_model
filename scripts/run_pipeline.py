"""
End-to-end pipeline: audio → voice model.

Usage:
    python scripts/run_pipeline.py --model_name my_voice --sr 40k --epochs 200 --gpus 0

Steps:
    1. frag.py          — segment raw audio
    2. rem_noise.py     — filter by SNR
    3. Mangio preprocess — normalize + resample
    4. Mangio f0        — pitch extraction
    5. Mangio features  — feature extraction
    6. Mangio training  — train voice model
    7. plot_loss.py     — loss report
    8. gan_feature_report.py — feature map report
"""
import argparse
import math
import os
import subprocess
import sys
from pathlib import Path
from random import shuffle

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
    train_script = MANGIO_DIR / "train_nsf_sim_cache_sid_load_pretrain.py"
    if not train_script.exists():
        print("ERROR: Mangio submodule not initialised.")
        print("Run: git submodule update --init")
        sys.exit(1)


def generate_filelist(exp_dir, version, if_f0, sr, spk_id=0):
    exp_path = Path(exp_dir)
    gt_wavs_dir = exp_path / "0_gt_wavs"
    fea_dim = 768 if version == "v2" else 256
    feature_dir = exp_path / f"3_feature{fea_dim}"

    if if_f0:
        f0_dir = exp_path / "2a_f0"
        f0nsf_dir = exp_path / "2b-f0nsf"
        names = (
            set(n.split(".")[0] for n in os.listdir(gt_wavs_dir))
            & set(n.split(".")[0] for n in os.listdir(feature_dir))
            & set(n.split(".")[0] for n in os.listdir(f0_dir))
            & set(n.split(".")[0] for n in os.listdir(f0nsf_dir))
        )
    else:
        names = (
            set(n.split(".")[0] for n in os.listdir(gt_wavs_dir))
            & set(n.split(".")[0] for n in os.listdir(feature_dir))
        )

    opt = []
    for name in names:
        if if_f0:
            opt.append(
                f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|"
                f"{f0_dir}/{name}.wav.npy|{f0nsf_dir}/{name}.wav.npy|{spk_id}"
            )
        else:
            opt.append(f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{spk_id}")

    mute_base = MANGIO_DIR / "logs" / "mute"
    if if_f0:
        mute_line = (
            f"{mute_base}/0_gt_wavs/mute{sr}.wav|{mute_base}/3_feature{fea_dim}/mute.npy|"
            f"{mute_base}/2a_f0/mute.wav.npy|{mute_base}/2b-f0nsf/mute.wav.npy|{spk_id}"
        )
    else:
        mute_line = (
            f"{mute_base}/0_gt_wavs/mute{sr}.wav|{mute_base}/3_feature{fea_dim}/mute.npy|{spk_id}"
        )
    opt.extend([mute_line, mute_line])

    shuffle(opt)
    filelist_path = exp_path / "filelist.txt"
    filelist_path.write_text("\n".join(opt))
    print(f"[pipeline] filelist.txt: {len(opt) - 2} audio + 2 mute entries")


def build_parser():
    p = argparse.ArgumentParser(description="RVC_from_audio_to_model — full pipeline")

    p.add_argument("--model_name", required=True,
                   help="Experiment / model name (used as logs/{model_name}/ in Mangio)")
    p.add_argument("--dataset", default=None,
                   help="Dataset folder name under dataset/ (default: same as --model_name). "
                        "Expects dataset/<name>/audios/ — outputs go to "
                        "dataset/<name>/segmented/ and dataset/<name>/filtered/")

    # Audio / model settings
    p.add_argument("--sr", default="40k", choices=["32k", "40k", "48k"],
                   help="Target sample rate (default: 40k)")
    p.add_argument("--f0", default=1, type=int, choices=[0, 1],
                   help="Pitch guidance: 1=yes (singing/voice), 0=no (default: 1)")
    p.add_argument("--version", default="v2", choices=["v1", "v2"],
                   help="RVC model version (default: v2)")

    # Training
    p.add_argument("--epochs", default=50, type=int, help="Total training epochs (default: 50)")
    p.add_argument("--save_epoch", default=10, type=int, help="Save checkpoint every N epochs")
    p.add_argument("--batch_size", default=8, type=int, help="Batch size (default: 8)")
    p.add_argument("--gpus", default="0",
                   help="GPU indices dash-separated, e.g. '0' or '0-1' (default: 0)")
    p.add_argument("--cache_gpu", action="store_true", help="Cache dataset on GPU VRAM")
    p.add_argument("--save_latest", action="store_true",
                   help="Keep only latest checkpoint (saves disk space)")
    p.add_argument("--save_every_weights", action="store_true",
                   help="Export .pth weights after each save_epoch")

    # Pretrained models
    p.add_argument("--pretrained_G", default="",
                   help="Path to pretrained G model (relative to mangio/)")
    p.add_argument("--pretrained_D", default="",
                   help="Path to pretrained D model (relative to mangio/)")

    # Feature extraction
    p.add_argument("--f0method", default="rmvpe",
                   choices=["pm", "harvest", "crepe", "rmvpe", "mangio-crepe"],
                   help="Pitch extraction method (default: rmvpe)")
    p.add_argument("--n_processes", default=4, type=int,
                   help="CPU processes for preprocessing/f0 (default: 4)")
    p.add_argument("--echl", default=160, type=int,
                   help="Crepe hop length — only used when f0method=crepe/mangio-crepe")

    # Skip flags
    p.add_argument("--skip_segment", action="store_true",
                   help="Skip frag.py + rem_noise.py (dataset/filtered already ready)")
    p.add_argument("--skip_analysis", action="store_true",
                   help="Skip post-training analysis scripts")

    return p


def main():
    args = build_parser().parse_args()
    check_submodule()

    py = sys.executable
    sr_int = {"32k": 32000, "40k": 40000, "48k": 48000}[args.sr]
    dataset_name = args.dataset or args.model_name
    dataset_dir = REPO_ROOT / "dataset" / dataset_name
    dataset_filtered = str(dataset_dir / "filtered")
    exp_dir = str(MANGIO_DIR / "logs" / args.model_name)
    gpu_list = args.gpus.split("-")
    n_gpus = len(gpu_list)

    # ------------------------------------------------------------------ #
    # 1-2. Dataset preparation                                            #
    # ------------------------------------------------------------------ #
    if not args.skip_segment:
        audios_dir = dataset_dir / "audios"
        if not audios_dir.is_dir() or not any(audios_dir.iterdir()):
            print(f"ERROR: no audio files in {audios_dir}")
            print(f"Place raw audio there (or pass --skip_segment if dataset/{dataset_name}/filtered is ready).")
            sys.exit(1)
        os.makedirs(dataset_dir / "segmented", exist_ok=True)
        os.makedirs(dataset_dir / "filtered", exist_ok=True)
        sub_env = {**os.environ, "DATASET_DIR": str(dataset_dir)}
        run(f'"{py}" 1_before-training/frag.py', env=sub_env)
        run(f'"{py}" 1_before-training/rem_noise.py', env=sub_env)

    # ------------------------------------------------------------------ #
    # 3. Mangio: preprocess                                               #
    # ------------------------------------------------------------------ #
    os.makedirs(exp_dir, exist_ok=True)

    # Mangio reads these CSVs relative to its CWD (mangio/); create if absent
    csvdb_dir = MANGIO_DIR / "csvdb"
    os.makedirs(csvdb_dir, exist_ok=True)
    formanting_csv = csvdb_dir / "formanting.csv"
    if not formanting_csv.exists():
        formanting_csv.write_text("False,1.0,1.0\n")
    stop_csv = csvdb_dir / "stop.csv"
    if not stop_csv.exists():
        stop_csv.write_text("False\n")

    run(
        f'"{py}" trainset_preprocess_pipeline_print.py '
        f'"{dataset_filtered}" {sr_int} {args.n_processes} '
        f'"{exp_dir}" False',
        cwd=MANGIO_DIR,
    )

    # ------------------------------------------------------------------ #
    # 4. Mangio: pitch extraction (f0)                                   #
    # ------------------------------------------------------------------ #
    run(
        f'"{py}" extract_f0_print.py '
        f'"{exp_dir}" {args.n_processes} {args.f0method} {args.echl}',
        cwd=MANGIO_DIR,
    )

    # ------------------------------------------------------------------ #
    # 5. Mangio: feature extraction (one pass per GPU)                   #
    # ------------------------------------------------------------------ #
    for idx, gpu_id in enumerate(gpu_list):
        run(
            f'"{py}" extract_feature_print.py '
            f'cuda:{gpu_id} {n_gpus} {idx} {gpu_id} '
            f'"{exp_dir}" {args.version}',
            cwd=MANGIO_DIR,
        )

    # ------------------------------------------------------------------ #
    # 5b. Generate filelist.txt (normally done by Mangio GUI)             #
    # ------------------------------------------------------------------ #
    generate_filelist(exp_dir, args.version, args.f0, args.sr)

    # ------------------------------------------------------------------ #
    # 6. Mangio: training                                                 #
    # ------------------------------------------------------------------ #
    wavs_dir = Path(exp_dir) / "1_16k_wavs"
    if wavs_dir.exists():
        wav_count = len([f for f in os.listdir(wavs_dir) if f.endswith(".wav")])
        log_interval = math.ceil(wav_count / args.batch_size) if wav_count else 1
        if log_interval > 1:
            log_interval += 1
    else:
        log_interval = 1

    # Resolve pretrained paths: use downloaded models by default
    def _pretrained(suffix):
        if args.version == "v2":
            p = MANGIO_DIR / "pretrained_v2" / f"{suffix}40k.pth"
        else:
            p = MANGIO_DIR / "pretrained" / f"{suffix}{args.sr}.pth"
        return str(p.relative_to(MANGIO_DIR)) if p.exists() else ""

    pg = args.pretrained_G or (_pretrained("f0G") if args.f0 else _pretrained("G"))
    pd = args.pretrained_D or (_pretrained("f0D") if args.f0 else _pretrained("D"))

    pretrain_flags = ""
    if pg:
        pretrain_flags += f' -pg "{pg}"'
    if pd:
        pretrain_flags += f' -pd "{pd}"'

    run(
        f'"{py}" train_nsf_sim_cache_sid_load_pretrain.py '
        f'-e {args.model_name} '
        f'-sr {args.sr} '
        f'-f0 {args.f0} '
        f'-bs {args.batch_size} '
        f'-g {args.gpus} '
        f'-te {args.epochs} '
        f'-se {args.save_epoch}'
        f'{pretrain_flags} '
        f'-l {1 if args.save_latest else 0} '
        f'-c {1 if args.cache_gpu else 0} '
        f'-sw {1 if args.save_every_weights else 0} '
        f'-v {args.version} '
        f'-li {log_interval}',
        cwd=MANGIO_DIR,
    )

    # ------------------------------------------------------------------ #
    # 7-8. Post-training analysis                                         #
    # ------------------------------------------------------------------ #
    if not args.skip_analysis:
        out_dir = REPO_ROOT / "3_after-training" / args.model_name
        os.makedirs(out_dir / "plots", exist_ok=True)
        run(
            f'"{py}" 3_after-training/gan_feature_report.py '
            f'--input_folder "{exp_dir}" '
            f'--output "3_after-training/{args.model_name}/feature_report.html"',
        )
        run(
            f'"{py}" 3_after-training/plot_loss.py '
            f'--logs_folder "{exp_dir}" '
            f'--output_dir "3_after-training/{args.model_name}/plots" '
            f'--html_output "3_after-training/{args.model_name}/loss_report.html"',
        )

    print(f"\nPipeline complete. Model: {args.model_name}")
    print(f"Checkpoints: {exp_dir}")


if __name__ == "__main__":
    main()
