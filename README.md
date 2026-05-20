# RVC — From Audio to Voice Model

> Scripts and tools for the full voice model training pipeline using **Mangio RVC** (fork of RVC — Retrieval-based Voice Conversion).

---

## Prerequisites

- **Python 3.10** (NOT 3.12 — Python 3.12 removed the `imp` module that `audioread`, a librosa dep, still imports).
  On Debian/Ubuntu:
  ```bash
  sudo add-apt-repository ppa:deadsnakes/ppa
  sudo apt install python3.10 python3.10-venv
  ```
- **Always use a venv.** Debian 12+ and recent Ubuntu enforce PEP 668 (`externally-managed-environment`),
  which blocks `pip install` against the system Python. The venv also keeps the
  pinned `pip<24.1` / `setuptools<81` from polluting the system install.
  ```bash
  python3.10 -m venv .venv
  source .venv/bin/activate    # Windows: .venv\Scripts\activate
  ```
- **NVIDIA Blackwell (RTX 50xx)**: pass `--torch_cu128` to `setup.py` — see Quick Start step 2.

---

## Quick Start

```bash
# 1. Clone (includes Mangio-RVC-Fork as submodule) and enter a Python 3.10 venv
git clone --recurse-submodules <repo_url>
cd RVC_from_audio_to_model
python3.10 -m venv .venv && source .venv/bin/activate

# 2. One-time setup: pins build tools, installs deps, downloads models, patches Mangio + fairseq
python scripts/setup.py
#   For RTX 50xx (Blackwell, sm_120) reinstall torch on CUDA 12.8 wheels:
# python scripts/setup.py --torch_cu128

# 3. Create the dataset folder and drop raw audio inside dataset/<dataset_name>/audios/
#    (WAV, FLAC, MP3, OGG, M4A, AAC). Example:
#       mkdir -p dataset/my_voice/audios
#       cp /path/to/*.wav dataset/my_voice/audios/
#    Subfolders segmented/ and filtered/ are created by the pipeline.

# path a dataset LS: "Mangio-RVC-Fork/datasets/Luca/LS/"

# 4. Run the full pipeline (reads dataset/my_voice/audios/, writes mangio/logs/my_voice/)
python scripts/run_pipeline.py --model_name my_voice --dataset my_voice --sr 40k --epochs 200 --gpus 0

# — or use the shell wrapper (same defaults, less typing) —
./run_pipeline.sh my_voice          # Linux / Mac / WSL
run_pipeline.bat my_voice           # Windows CMD

# 5. Train another speaker without overwriting the first
mkdir -p dataset/another_voice/audios   # add audio
./run_pipeline.sh another_voice         # new model, new dataset folder
```

By default `--dataset` mirrors `--model_name`, so audio for `my_voice` lives in
`dataset/my_voice/audios/`. To reuse one dataset folder across models, pass
`--dataset <folder_name>` explicitly. Each dataset is self-contained, so the
pipeline can be run any number of times for different speakers without
overwriting prior `segmented/` or `filtered/` outputs.

The pipeline runs automatically:
1. Segment audio into ~10 s clips (`frag.py`)
2. Filter by SNR ≥ 25 dB (`rem_noise.py`)
3. Mangio preprocessing, pitch extraction, feature extraction
4. Model training
5. Loss report + feature map report (HTML)

```
RVC_from_audio_to_model/
├── dataset/
│   └── <dataset_name>/         ← one folder per speaker / experiment
│       ├── audios/             ← raw source audio (place files here)
│       ├── segmented/          ← frag.py output: ~10 s WAV clips
│       └── filtered/           ← rem_noise.py output: clips with SNR ≥ 25 dB
├── mangio/
│   ├── csvdb/
│   │   ├── formanting.csv      ← auto-created (formant shift disabled by default)
│   │   └── stop.csv            ← auto-created (GUI stop flag, set to False)
│   └── logs/
│       └── <model_name>/
│           ├── 0_gt_wavs/              preprocessed ground-truth WAVs
│           ├── 1_16k_wavs/             WAVs resampled to 16 kHz (HuBERT input)
│           ├── 2a_f0/                  pitch (f0) arrays — .wav.npy per clip
│           ├── 2b-f0nsf/               NSF-format pitch arrays — .wav.npy per clip
│           ├── 3_feature768/           HuBERT feature vectors — .npy per clip (v2)
│           ├── filelist.txt            training manifest (auto-generated)
│           ├── train.log               training log (loss values per step)
│           ├── feature_map_summary.csv GAN discriminator feature stats per epoch
│           ├── G_<step>.pth            generator checkpoints (every save_epoch)
│           ├── D_<step>.pth            discriminator checkpoints (every save_epoch)
│           └── added_IVF*_*.index      faiss index (built on demand by run_inference.py)
│   └── weights/
│       └── <model_name>.pth            final exported voice model ← use this for inference
└── 3_after-training/
    └── <model_name>/
        ├── feature_report.html         interactive GAN stability report (Plotly)
        ├── loss_report.html            training loss charts (per metric + combined)
        └── plots/
            ├── train_loss_disc.png
            ├── train_loss_gen.png
            ├── train_loss_fm.png
            ├── train_loss_mel.png
            ├── train_loss_kl.png
            └── combined_*.png          overlaid curves (useful when comparing models)
```

---

## Re-training the same speaker

When you re-run the pipeline for a speaker whose `dataset/<name>/filtered/` is
already populated (e.g. to tune `--epochs`, `--batch_size`, or swap the
pretrained model), skip the dataset-prep stage:

```bash
# Skip frag.py + rem_noise.py (steps 1-2). Mangio preprocess, f0 and feature
# extraction still re-run and overwrite mangio/logs/<model_name>/{0_gt_wavs,
# 2a_f0, 2b-f0nsf, 3_feature768}.
./run_pipeline.sh my_voice --skip_segment --epochs 300

# Also skip post-training HTML reports
./run_pipeline.sh my_voice --skip_segment --skip_analysis
```

| Stage | Skip flag | When safe to skip |
|---|---|---|
| `frag.py` + `rem_noise.py` | `--skip_segment` | `dataset/<name>/filtered/` already populated |
| Loss + feature-map HTML reports | `--skip_analysis` | You don't need fresh reports for this run |

> Mangio preprocess / f0 / feature extraction (steps 3-5) have no skip flag —
> they always re-run and overwrite the contents of
> `mangio/logs/<model_name>/{0_gt_wavs, 1_16k_wavs, 2a_f0, 2b-f0nsf, 3_feature768}/`.
> If you want to keep the existing features, back up that folder first or train
> under a different `--model_name` (passing `--dataset <shared_folder>` to share
> the filtered audio).

---

## Repository Structure

```
RVC_from_audio_to_model/
├── run_pipeline.sh                # Training wrapper — Linux / Mac / WSL
├── run_pipeline.bat               # Training wrapper — Windows CMD
├── run_inference.sh               # Inference wrapper — Linux / Mac / WSL
├── run_inference.bat              # Inference wrapper — Windows CMD
├── mangio/                        # Mangio-RVC-Fork (git submodule, MIT)
├── scripts/
│   ├── setup.py                   # One-time setup (deps + models + patch)
│   ├── patch_mangio.py            # Injects GAN monitoring into Mangio train script
│   ├── run_pipeline.py            # End-to-end training pipeline orchestrator
│   └── run_inference.py           # End-to-end inference pipeline orchestrator
├── dataset/
│   └── <dataset_name>/            # One folder per speaker / experiment
│       ├── audios/                # Raw source audio — place files here
│       ├── segmented/             # Output of frag.py (~10 s clips)
│       └── filtered/              # Output of rem_noise.py (SNR ≥ 25 dB)
├── inference_dataset/
│   └── <inference_dataset>/       # Drop WAVs to convert here (e.g. inf_dataset_1/)
├── inference_results/
│   └── <model_name>_<inference_dataset>/   # Converted WAVs land here
├── 1_before-training/
│   ├── frag.py                    # Audio segmentation
│   └── rem_noise.py               # SNR filtering
├── 2_during-training/
│   └── feature_functions.txt      # Reference: functions injected by patch_mangio.py
├── 3_after-training/
│   ├── gan_feature_report.py      # Interactive HTML report of GAN feature maps
│   └── plot_loss.py               # Training loss charts and HTML report
└── requirements.txt
```

---

## Shell Wrappers

`run_pipeline.sh` / `run_pipeline.bat` call `scripts/run_pipeline.py` with sensible defaults.
All `--flags` are forwarded verbatim; env vars let you override defaults without touching the script.

**Linux / Mac / WSL**
```bash
# Basic — model name only
./run_pipeline.sh my_voice

# Override specific flags
./run_pipeline.sh my_voice --epochs 300 --sr 48k

# Override via env vars (valid for that run only)
GPUS=0-1 EPOCHS=500 ./run_pipeline.sh my_voice
```

**Windows CMD**
```bat
rem Basic
run_pipeline.bat my_voice

rem Override specific flags
run_pipeline.bat my_voice --epochs 300 --sr 48k

rem Override via env vars (persist for the CMD session)
set GPUS=0-1 && run_pipeline.bat my_voice
```

**Override priority** (highest wins):

| Level | Example |
|---|---|
| `--flag` on command line | `--epochs 300` |
| Env var | `EPOCHS=500 ./run_pipeline.sh` |
| Default in script | `EPOCHS=200` |

**Env vars recognised by the wrappers:**

| Var | Default | Equivalent flag |
|---|---|---|
| `MODEL_NAME` | `my_voice` | first positional arg |
| `SR` | `40k` | `--sr` |
| `EPOCHS` | `200` | `--epochs` |
| `SAVE_EPOCH` | `10` | `--save_epoch` |
| `BATCH_SIZE` | `8` | `--batch_size` |
| `GPUS` | `0` | `--gpus` |
| `F0METHOD` | `rmvpe` | `--f0method` |
| `N_PROCESSES` | `4` | `--n_processes` |

> Use `--dataset <folder>` on the command line to point at a dataset folder
> whose name differs from `--model_name`. Example:
> `./run_pipeline.sh my_voice_v2 --dataset my_voice`

> **Linux/Mac**: run `chmod +x run_pipeline.sh` once before first use.

---

## run_pipeline.py — Options

```
--model_name        Model / experiment name (required)
--dataset           Dataset folder under dataset/ (default: same as --model_name)
--sr                Sample rate: 32k | 40k | 48k  (default: 40k)
--f0                Pitch guidance: 1=yes 0=no     (default: 1)
--version           RVC version: v1 | v2           (default: v2)
--epochs            Total training epochs           (default: 200)
--save_epoch        Save checkpoint every N epochs  (default: 10)
--batch_size        Batch size                      (default: 8)
--gpus              GPU indices, dash-separated     (default: 0)
--f0method          Pitch method: rmvpe | harvest | crepe | pm  (default: rmvpe)
--pretrained_G      Path to pretrained G .pth (relative to mangio/)
--pretrained_D      Path to pretrained D .pth (relative to mangio/)
--cache_gpu         Cache dataset in GPU VRAM
--save_latest       Keep only the latest checkpoint
--save_every_weights  Export .pth weights every save_epoch
--skip_segment      Skip frag.py + rem_noise.py (dataset/<name>/filtered/ already ready)
--skip_analysis     Skip post-training HTML reports
```

---

## setup.py — Options

```
--version           Pretrained model version to download: v1 | v2  (default: v2)
--torch_cu128       Reinstall torch 2.7 + CUDA 12.8 wheels (required for RTX 50xx / sm_120)
--skip_deps         Skip pip install (already done)
--skip_models       Skip pretrained model download (already done)
--skip_patch        Skip patch_mangio.py
```

### What setup.py patches outside the repo

Three things sit outside `git` control and are re-applied on every fresh venv:

| What | Why | Where it's handled |
|---|---|---|
| `pip<24.1` pin | pip 24.1+ rejects `omegaconf==2.0.6` (malformed metadata) | `setup.py` step 1 |
| `setuptools<81` pin | setuptools 81 removed `pkg_resources`; `librosa 0.9.1` still imports it | `setup.py` step 1 |
| `PySimpleGUI` line removed from `mangio/requirements.txt` | Pulled from PyPI when the author went commercial. Only used by `mangio/gui_*.py`, not by the training/inference pipeline | `setup.py` step 2 (filters at install time — submodule file untouched) |
| `fairseq/checkpoint_utils.py` → `weights_only=False` | torch 2.6+ flipped `torch.load` default to `True`; refuses the `Dictionary` object in `hubert_base.pt` | `patch_mangio.py` → `patch_fairseq()` (called by `setup.py` step 5) |

> **PyTorch**: by default `setup.py` installs `torch==2.0.0` (CUDA 11.7 wheel) from Mangio's requirements.
> Pass `--torch_cu128` to reinstall on CUDA 12.8 wheels — required for RTX 50xx (Blackwell, sm_120), since
> `torch==2.0.0` only ships kernels up to sm_86 and crashes with `no kernel image is available for execution
> on the device`. For a custom CUDA version: install PyTorch manually, then run `python scripts/setup.py --skip_deps`.

---

## Inference (audio → converted audio)

After training a model you can convert any WAV through it.

### Where to put the audio

Drop the WAV(s) to convert into a subfolder of `inference_dataset/`:

```
inference_dataset/
└── inf_dataset_1/         ← default subfolder name
    ├── clip1.wav
    └── clip2.wav
```

Create as many sibling folders as you like (`inf_dataset_2/`, `my_test/`, …) and
pass the folder name to the wrapper.

```bash
# Linux / Mac / WSL — reads inference_dataset/inf_dataset_1/ by default
./run_inference.sh my_voice

# Pick a different subfolder
./run_inference.sh my_voice inf_dataset_2 --transpose 2

# Windows CMD
run_inference.bat my_voice inf_dataset_1

# Python script directly
python scripts/run_inference.py --model_name my_voice --inference_dataset inf_dataset_1

# Override the convention with an explicit path (single file or folder)
python scripts/run_inference.py --model_name my_voice --input C:\path\to\clip.wav
```

Output goes to `inference_results/<model_name>_<inference_dataset>/` by default
(keeps inputs and results in separate top-level folders, one result folder per
model/dataset pair so nothing gets clobbered). Override with `--output_dir`.

### What it resolves automatically

| Asset | Location |
|---|---|
| Model weights | `mangio/weights/<model_name>.pth` |
| Faiss index | `mangio/logs/<model_name>/added_IVF*_<model_name>_<version>.index` |
| RVC version | Read from the `.pth` checkpoint |

If the faiss index is missing pass `--build_index` to construct it from the
extracted features in `mangio/logs/<model_name>/3_feature{256|768}/`. Without an
index (and without `--build_index`) inference still runs with `index_rate=0`.

### run_inference.py — Options

```
--model_name        Trained model name (required; looks up mangio/weights/<name>.pth)
--inference_dataset Subfolder under inference_dataset/ (default: inf_dataset_1)
--input             Override: explicit WAV file or folder (default: inference_dataset/<inference_dataset>/)
--output_dir        Override output folder (default: inference_results/<model_name>_<inference_dataset>/)
--transpose, -k   Pitch shift in semitones (default: 0)
--f0method        Pitch method: rmvpe | harvest | crepe | pm | mangio-crepe (default: rmvpe)
--index_rate      Index influence 0..1 (default: 0.66)
--filter_radius   Median filter radius for harvest f0 (default: 3)
--resample_sr     Resample output sr; 0 = keep model sr (default: 0)
--rms_mix_rate    Volume envelope mix 0..1 (default: 1.0)
--protect         Protect voiceless consonants 0..0.5 (default: 0.33)
--device          cuda:0 | cpu (default: cuda:0)
--is_half         fp16 inference: true | false (default: true)
--index_path      Explicit .index file (skips auto-discovery)
--build_index     Build index from features if missing
--no_index        Skip index entirely (forces index_rate=0)
```

### Shell wrapper env vars

| Var | Default | Equivalent flag |
|---|---|---|
| `MODEL_NAME` | `my_voice` | first positional arg |
| `INFERENCE_DATASET` | `inf_dataset_1` | second positional arg |
| `TRANSPOSE` | `0` | `--transpose` |
| `F0METHOD` | `rmvpe` | `--f0method` |
| `INDEX_RATE` | `0.66` | `--index_rate` |
| `DEVICE` | `cuda:0` | `--device` |
| `IS_HALF` | `true` | `--is_half` |

---

## Post-training analysis (standalone)

If you want to run analysis separately after training:

```bash
python 3_after-training/gan_feature_report.py \
    --input_folder mangio/logs/my_voice \
    --output 3_after-training/my_voice/feature_report.html

python 3_after-training/plot_loss.py \
    --logs_folder mangio/logs/my_voice \
    --output_dir 3_after-training/my_voice/plots \
    --html_output 3_after-training/my_voice/loss_report.html
```

---

## Recommended Audio Requirements

| Property | Recommendation |
|---|---|
| Environment | Soundproofed room, no reverb or echo |
| Microphone | Condenser (e.g. Neumann, AKG, Rode NT) with quality preamp |
| Sample rate | 44.1 kHz or 48 kHz |
| Duration | 10–15 min clean speech minimum; 30–60 min for high quality |
| Formats | WAV, FLAC, MP3, OGG, M4A, AAC |

---

## Phase Details

### 1 — Dataset preparation (`1_before-training/`)

**`frag.py`** — Segments audio into ~10 s clips at natural silences.

| Parameter | Value |
|---|---|
| `TARGET_MS` | 10 000 ms |
| `MIN_SEGMENT_MS` | 350 ms |
| `MIN_SILENCE_MS` | 1 000 ms |
| `SILENCE_THRESH_DB` | –40 dBFS |
| `SEARCH_WINDOW_MS` | ±5 000 ms |

**`rem_noise.py`** — Keeps segments with SNR ≥ 25 dB.
Formula: `SNR = 20 · log₁₀((RMS_voice + ε) / (RMS_noise + ε))` with ε = 10⁻¹²

### 2 — During training (`2_during-training/`)

**`feature_functions.txt`** — Reference copy of the two functions injected by `patch_mangio.py` into `mangio/train_nsf_sim_cache_sid_load_pretrain.py`.

- **`_summarize_feature_maps`** — Per discriminator/layer: `mean_r/g`, `std_r/g`, `l1_mean`, `shape_r/g`.
- **`_append_feature_map_summary`** — Appends stats to `mangio/logs/<model>/feature_map_summary.csv`.

Key metric: **`l1_mean`** should decrease over time. Flat or rising values signal instability or mode collapse.

### 3 — Post-training (`3_after-training/`)

**`gan_feature_report.py`** — Reads `feature_map_summary.csv`, generates interactive Plotly HTML:
- Global L1 trend (mean ± 1σ)
- Per-discriminator curves per layer
- Aggregated table

**`plot_loss.py`** — Parses `.log` files, extracts `loss_disc`, `loss_gen`, `loss_fm`, `loss_mel`, `loss_kl`:
- Per-model charts
- Multi-model comparative overlay
- Navigable HTML report

---

## References

- [Mangio-RVC-Fork](https://github.com/Mangio621/Mangio-RVC-Fork) — MIT License © liujing04, 源文雨
