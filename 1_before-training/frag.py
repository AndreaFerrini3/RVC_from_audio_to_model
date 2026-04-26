import os
from pydub import AudioSegment
from pydub.silence import detect_silence

# --- CONFIG ---
INPUT_DIR = "dataset/audios" # directory containing the original audio files to be segmented
OUTPUT_DIR = os.path.join(INPUT_DIR, "dataset/segmented") # directory to save the segmented audio files (e.g., "segmented")

TARGET_MS = 22000          # Target segment duration (~10s)
MIN_SEGMENT_MS = 350        # Min segment duration for RMVPE (~0.35s)
MIN_SILENCE_MS = 500       # Minimum silence duration considered
SILENCE_THRESH_DB = -40     # Silence threshold
SEARCH_WINDOW_MS = 8000    # ±5s to search for silence near the target

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- FUNCTIONS ---
def find_silence_points(audio):
    """Returns list of centers of detected silences (ms)"""
    silences = detect_silence(
        audio,
        min_silence_len=MIN_SILENCE_MS,
        silence_thresh=SILENCE_THRESH_DB
    )
    return [(start + end) // 2 for start, end in silences]

def segment_audio(src_path):
    audio = AudioSegment.from_wav(src_path)
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

        # if the segment is too short, skip it and move the cursor forward
        cursor = cut_point

    print(f"{count} segments generated from {src_path}")

def main():
    for file in os.listdir(INPUT_DIR):
        if file.lower().endswith(".wav"):
            segment_audio(os.path.join(INPUT_DIR, file))

if __name__ == "__main__":
    main()
