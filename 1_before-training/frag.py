import os
from pydub import AudioSegment
from pydub.silence import detect_silence

# --- CONFIG ---
INPUT_DIR = "dataset/audios"
OUTPUT_DIR = "dataset/segmented"

TARGET_MS = 10000           # Target segment duration (~10s)
MIN_SEGMENT_MS = 350        # Min segment duration for RMVPE (~0.35s)
MIN_SILENCE_MS = 1000       # Minimum silence duration considered
SILENCE_THRESH_DB = -40     # Silence threshold
SEARCH_WINDOW_MS = 5000     # ±5s to search for silence near the target

SUPPORTED_FORMATS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- FUNCTIONS ---
def load_audio(src_path):
    ext = os.path.splitext(src_path)[1].lower()
    fmt_map = {".wav": "wav", ".flac": "flac", ".mp3": "mp3",
               ".ogg": "ogg", ".m4a": "mp4", ".aac": "aac"}
    fmt = fmt_map.get(ext, ext.lstrip("."))
    audio = AudioSegment.from_file(src_path, format=fmt)
    # normalise to 16-bit PCM WAV in memory so the rest of the pipeline is uniform
    return audio.set_sample_width(2)

def find_silence_points(audio):
    silences = detect_silence(
        audio,
        min_silence_len=MIN_SILENCE_MS,
        silence_thresh=SILENCE_THRESH_DB
    )
    return [(start + end) // 2 for start, end in silences]

def segment_audio(src_path):
    audio = load_audio(src_path)
    silence_points = find_silence_points(audio)
    duration = len(audio)
    base = os.path.splitext(os.path.basename(src_path))[0]

    cursor = 0
    count = 0

    while cursor < duration:
        target = cursor + TARGET_MS

        # search for silence near the target
        candidates = [
            point for point in silence_points
            if cursor < point < duration and abs(point - target) <= SEARCH_WINDOW_MS
        ]

        if candidates:
            cut_point = min(candidates, key=lambda p: abs(p - target))
        else:
            cut_point = min(target, duration)

        segment = audio[cursor:cut_point]

        # avoid segments that are too short
        if len(segment) >= MIN_SEGMENT_MS:
            out_file = os.path.join(OUTPUT_DIR, f"{base}_seg_{count:03d}.wav")
            segment.export(out_file, format="wav")
            count += 1

        cursor = cut_point

    print(f"{count} segments generated from {src_path}")

def main():
    if not os.path.isdir(INPUT_DIR):
        print(f"ERROR: Input directory '{INPUT_DIR}' not found. Create it and add audio files.")
        return
    for file in os.listdir(INPUT_DIR):
        if file.lower().endswith(SUPPORTED_FORMATS):
            segment_audio(os.path.join(INPUT_DIR, file))

if __name__ == "__main__":
    main()
