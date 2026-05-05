# RVC — From Audio to Voice Model

> Scripts and tools for the full voice model training pipeline using **Mangio RVC** (fork of RVC — Retrieval-based Voice Conversion).

---

## Quick Start

```bash
# 1. Clone (includes Mangio-RVC-Fork as submodule)
git clone --recurse-submodules <repo_url>
cd RVC_from_audio_to_model

# 2. One-time setup: dependencies + pretrained models + patch Mangio
python scripts/setup.py

# 3. Create the dataset folder and drop raw audio inside dataset/<dataset_name>/audios/
#    (WAV, FLAC, MP3, OGG, M4A, AAC). Example:
#       mkdir -p dataset/my_voice/audios
#       cp /path/to/*.wav dataset/my_voice/audios/
#    Subfolders segmented/ and filtered/ are created by the pipeline.

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
│           └── <model_name>.pth        final exported voice model ← use this for inference
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

## Repository Structure

```
RVC_from_audio_to_model/
├── run_pipeline.sh                # Shell wrapper — Linux / Mac / WSL
├── run_pipeline.bat               # Shell wrapper — Windows CMD
├── mangio/                        # Mangio-RVC-Fork (git submodule, MIT)
├── scripts/
│   ├── setup.py                   # One-time setup (deps + models + patch)
│   ├── patch_mangio.py            # Injects GAN monitoring into Mangio train script
│   └── run_pipeline.py            # End-to-end pipeline orchestrator
├── dataset/
│   └── <dataset_name>/            # One folder per speaker / experiment
│       ├── audios/                # Raw source audio — place files here
│       ├── segmented/             # Output of frag.py (~10 s clips)
│       └── filtered/              # Output of rem_noise.py (SNR ≥ 25 dB)
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
--skip_deps         Skip pip install (already done)
--skip_models       Skip pretrained model download (already done)
--skip_patch        Skip patch_mangio.py
```

> **Note on PyTorch**: `setup.py` installs `torch==2.0.0` from Mangio's `requirements.txt`.
> If you need a specific CUDA version, install PyTorch manually first, then run:
> `python scripts/setup.py --skip_deps`

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
