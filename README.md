# RVC — Dal Segmento Audio al Modello Vocale / From Audio to Voice Model

> Documentazione e script per la pipeline completa di addestramento di un modello vocale con **Mangio RVC** (fork di RVC — Retrieval-based Voice Conversion).
>
> Documentation and scripts for the full voice model training pipeline using **Mangio RVC** (fork of RVC — Retrieval-based Voice Conversion).

---

## Italiano

### Panoramica

Questa repository raccoglie tutto il materiale relativo alla procedura di addestramento di un modello vocale con Mangio RVC. Il processo si divide in tre macro-fasi, ciascuna con la propria cartella dedicata.

### Pipeline

```
Registrazione → Segmentazione → Filtraggio SNR → Training → Analisi
```

### Struttura della repository

```
RVC_from_audio_to_model/
├── dataset/
│   ├── audios/               # Audio sorgente grezzo (input di frag.py)
│   ├── segmented/            # Segmenti ~10s prodotti da frag.py
│   └── filtered/             # Segmenti filtrati (SNR ≥ 25 dB) pronti per il training
├── 1_before-training/
│   ├── frag.py               # Segmentazione audio in clip ~10s
│   └── rem_noise.py          # Filtraggio segmenti per SNR ≥ 25 dB
├── 2_during-training/
│   └── feature_functions.txt # Funzioni di monitoraggio feature map GAN
└── 3_after-training/
    ├── gan_feature_report.py  # Report HTML interattivo delle feature map
    └── plot_loss.py           # Grafici e report HTML delle loss di training
```

### Fasi

#### Dataset (`dataset/`)

La cartella `dataset/` segue il flusso dei dati attraverso la pipeline:

| Cartella | Contenuto |
|---|---|
| `audios/` | File audio grezzi sorgente — metti qui i tuoi WAV prima di eseguire `frag.py` |
| `segmented/` | Output di `frag.py` — segmenti ~10s con tagli ai silenzi naturali |
| `filtered/` | Output di `rem_noise.py` — segmenti filtrati (SNR ≥ 25 dB) pronti per il training in RVC |

#### 1 — Preparazione del dataset (`1_before-training/`)

**`frag.py`** — Segmentazione intelligente dell'audio sorgente.
- Taglia il file WAV in segmenti di circa **10 secondi** cercando i silenzi naturali (≥ 1000 ms, soglia –40 dBFS) entro una finestra di **±5 secondi** attorno all'istante target.
- Scarta automaticamente i segmenti più corti di **350 ms** (troppo brevi per il pitch estimator RMVPE di RVC).

| Parametro | Default |
|---|---|
| `TARGET_MS` | 10 000 ms |
| `MIN_SEGMENT_MS` | 350 ms |
| `MIN_SILENCE_MS` | 1 000 ms |
| `SILENCE_THRESH_DB` | –40 dBFS |
| `SEARCH_WINDOW_MS` | ±5 000 ms |

**`rem_noise.py`** — Filtraggio per qualità SNR.
- Calcola il rapporto segnale-rumore di ogni segmento separando frame vocali e silenziosi tramite `librosa.effects.split`.
- Formula: `SNR = 20 · log₁₀((RMS_voce + ε) / (RMS_rumore + ε))` con ε = 10⁻¹²
- Conserva solo i segmenti con **SNR ≥ 25 dB**.

#### 2 — Durante il training (`2_during-training/`)

**`feature_functions.txt`** — Contiene le due funzioni personalizzate da inserire nel codice di training di Mangio RVC per monitorare l'andamento della GAN a livello interno.

- **`_summarize_feature_maps(fmap_r, fmap_g)`** — Per ogni coppia discriminatore/layer calcola: `mean_r`, `mean_g`, `std_r`, `std_g`, `l1_mean`, `shape_r`, `shape_g`.
- **`_append_feature_map_summary(...)`** — Scrive le statistiche in un CSV (append), con colonne `epoch`, `batch_idx`, `global_step`, `disc`, `layer` e le metriche sopra.

La metrica chiave è **`l1_mean`**: deve decrescere nel tempo per indicare convergenza. Valori stabili o crescenti segnalano instabilità del generatore o mode collapse.

#### 3 — Analisi post-training (`3_after-training/`)

**`gan_feature_report.py`** — Legge i CSV prodotti durante il training e genera un report HTML interattivo (Plotly) con:
- Grafico globale L1 (media ± 1σ su tutti i discriminatori e layer)
- Grafici per discriminatore (una curva per layer)
- Tabella aggregata per epoch/discriminatore/layer

```bash
python gan_feature_report.py --input_folder ./csv_dir --output report.html
```

**`plot_loss.py`** — Analizza i file `.log` di RVC ed estrae le loss per epoca (`loss_disc`, `loss_gen`, `loss_fm`, `loss_mel`, `loss_kl`). Genera:
- Grafici per singolo modello
- Grafici comparativi multi-modello sovrapposti
- Report HTML navigabile

```bash
python plot_loss.py
```

### Requisiti audio consigliati

- **Ambiente**: stanza insonorizzata, assenza di riverbero ed eco.
- **Hardware**: microfono a condensatore (es. Neumann, AKG, Rode NT), interfaccia audio con buon preamp.
- **Frequenza di campionamento**: 44.1 kHz o 48 kHz.
- **Quantità**: minimo 10–15 minuti di parlato pulito; 30–60 minuti per qualità elevata.

### Riferimenti

- [Mangio-RVC-Fork](https://github.com/Mangio621/Mangio-RVC-Fork)

---

## English

### Overview

This repository contains all material related to the voice model training procedure using Mangio RVC. The process is divided into three macro-phases, each with its own dedicated folder.

### Pipeline

```
Recording → Segmentation → SNR Filtering → Training → Analysis
```

### Repository Structure

```
RVC_from_audio_to_model/
├── dataset/
│   ├── audios/               # Raw source audio (input to frag.py)
│   ├── segmented/            # ~10s segments produced by frag.py
│   └── filtered/             # SNR-filtered segments (≥ 25 dB) ready for training
├── 1_before-training/
│   ├── frag.py               # Splits audio into ~10s clips
│   └── rem_noise.py          # Filters segments by SNR ≥ 25 dB
├── 2_during-training/
│   └── feature_functions.txt # GAN feature map monitoring functions
└── 3_after-training/
    ├── gan_feature_report.py  # Interactive HTML report of feature maps
    └── plot_loss.py           # Training loss charts and HTML report
```

### Phases

#### Dataset (`dataset/`)

The `dataset/` folder mirrors the data flow through the pipeline:

| Folder | Contents |
|---|---|
| `audios/` | Raw source audio files — place your WAVs here before running `frag.py` |
| `segmented/` | Output of `frag.py` — ~10s clips cut at natural silences |
| `filtered/` | Output of `rem_noise.py` — SNR-filtered segments (≥ 25 dB) ready for RVC training |

#### 1 — Dataset preparation (`1_before-training/`)

**`frag.py`** — Smart audio segmentation.
- Splits a WAV file into segments of approximately **10 seconds** by locating natural silences (≥ 1000 ms, threshold –40 dBFS) within a **±5-second** window around the target timestamp.
- Automatically discards segments shorter than **350 ms** (too short for RVC's RMVPE pitch estimator).

| Parameter | Default |
|---|---|
| `TARGET_MS` | 10 000 ms |
| `MIN_SEGMENT_MS` | 350 ms |
| `MIN_SILENCE_MS` | 1 000 ms |
| `SILENCE_THRESH_DB` | –40 dBFS |
| `SEARCH_WINDOW_MS` | ±5 000 ms |

**`rem_noise.py`** — SNR-based quality filtering.
- Computes the signal-to-noise ratio of each segment by separating voiced and silent frames via `librosa.effects.split`.
- Formula: `SNR = 20 · log₁₀((RMS_voice + ε) / (RMS_noise + ε))` with ε = 10⁻¹²
- Keeps only segments with **SNR ≥ 25 dB**.

#### 2 — During training (`2_during-training/`)

**`feature_functions.txt`** — Contains two custom functions to inject into the Mangio RVC training code for internal GAN monitoring.

- **`_summarize_feature_maps(fmap_r, fmap_g)`** — For each discriminator/layer pair, computes: `mean_r`, `mean_g`, `std_r`, `std_g`, `l1_mean`, `shape_r`, `shape_g`.
- **`_append_feature_map_summary(...)`** — Appends statistics to a CSV file with columns `epoch`, `batch_idx`, `global_step`, `disc`, `layer`, and the metrics above.

The key metric is **`l1_mean`**: it should decrease over time to indicate convergence. Flat or increasing values signal generator instability or partial mode collapse.

#### 3 — Post-training analysis (`3_after-training/`)

**`gan_feature_report.py`** — Reads the CSVs produced during training and generates an interactive HTML report (Plotly) with:
- Global L1 chart (mean ± 1σ across all discriminators and layers)
- Per-discriminator charts (one curve per layer)
- Aggregated table by epoch/discriminator/layer

```bash
python gan_feature_report.py --input_folder ./csv_dir --output report.html
```

**`plot_loss.py`** — Parses RVC `.log` files and extracts per-epoch losses (`loss_disc`, `loss_gen`, `loss_fm`, `loss_mel`, `loss_kl`). Produces:
- Per-model charts
- Multi-model comparative overlay charts
- Navigable HTML report

```bash
python plot_loss.py
```

### Recommended Audio Requirements

- **Environment**: soundproofed room, no reverb or echo.
- **Hardware**: condenser microphone (e.g. Neumann, AKG, Rode NT), audio interface with a quality preamp.
- **Sample rate**: 44.1 kHz or 48 kHz.
- **Amount**: minimum 10–15 minutes of clean speech; 30–60 minutes for high-quality models.

### References

- [Mangio-RVC-Fork](https://github.com/Mangio621/Mangio-RVC-Fork)

---
