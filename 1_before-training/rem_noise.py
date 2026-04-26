import os
import shutil
import librosa
import numpy as np

input_dir = "dataset/segmented" # directory with the segmented audio files from frag.py
output_dir = "dataset/filtered" # directory to save the filtered audio files with good SNR

os.makedirs(output_dir, exist_ok=True)

def compute_snr_metrics(audio, sr):
    # Identifies non-silent segments using librosa.effects.split
    non_silent = librosa.effects.split(audio, top_db=30)

    # If no non-silent segments are found, consider the entire audio as noise
    if len(non_silent) == 0:
        global_rms = float(np.sqrt(np.mean(audio ** 2)))
        return {
            "voice_rms": 0.0,
            "noise_rms": global_rms,
            "snr_db": -np.inf,
        }

    # Build list of silent intervals based on non-silent segments
    silent_intervals = []
    previous_end = 0

    for start, end in non_silent:
        if start > previous_end:
            silent_intervals.append((previous_end, start))
        previous_end = end

    if previous_end < len(audio):
        silent_intervals.append((previous_end, len(audio)))

    # voice RMS: median RMS of the non-silent segments
    voice_rms_values = []
    for start, end in non_silent:
        segment = audio[start:end]
        if len(segment) > 0:
            voice_rms_values.append(float(np.sqrt(np.mean(segment ** 2))))

    voice_rms = float(np.median(voice_rms_values)) if len(voice_rms_values) > 0 else 0.0

    # noise RMS: median RMS of the silent segments
    noise_rms = []
    for start, end in silent_intervals:
        segment = audio[start:end]
        if len(segment) > 0:
            noise_rms.append(float(np.sqrt(np.mean(segment ** 2))))

    # If no silent segments are found, use the global RMS as noise estimate
    if len(noise_rms) == 0:
        noise_rms_value = float(np.sqrt(np.mean(audio ** 2)))
    else:
        noise_rms_value = float(np.median(noise_rms))

    eps = 1e-12
    snr_db = 20.0 * np.log10((voice_rms + eps) / (noise_rms_value + eps))

    return {
        "voice_rms": voice_rms,
        "noise_rms": noise_rms_value,
        "snr_db": float(snr_db),
    }

# Minimum SNR threshold in dB to keep a file
# Adjust this value based on your needs. A common threshold for good quality speech is around 20-30 dB, but you may want to experiment with it.
SNR_THRESHOLD_DB = 25.0

for filename in os.listdir(input_dir):
    if not filename.lower().endswith((".wav", ".flac", ".mp3", ".ogg")):
        continue

    filepath = os.path.join(input_dir, filename)

    try:
        audio, sr = librosa.load(filepath, sr=None)
        metrics = compute_snr_metrics(audio, sr)
        voice_rms = metrics["voice_rms"]
        noise_rms = metrics["noise_rms"]
        snr_db = metrics["snr_db"]

        if snr_db >= SNR_THRESHOLD_DB:
            shutil.copy(filepath, os.path.join(output_dir, filename))
            print(
                f"✔ Kept: {filename} "
                f"(snr={snr_db:.2f} dB, "
                f"voice_rms={voice_rms:.4f}, noise_rms={noise_rms:.4f})"
            )
        else:
            print(
                f"✘ Removed: {filename} "
                f"(snr={snr_db:.2f} dB, "
                f"voice_rms={voice_rms:.4f}, noise_rms={noise_rms:.4f})"
            )

    except Exception as e:
        print(f"Error with {filename}: {e}")
