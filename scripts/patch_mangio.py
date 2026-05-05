"""
Patches mangio/train_nsf_sim_cache_sid_load_pretrain.py to inject
feature map monitoring. Run once after: git submodule update --init
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
TRAIN_SCRIPT = REPO_ROOT / "mangio" / "train_nsf_sim_cache_sid_load_pretrain.py"

ALREADY_PATCHED_MARKER = "_summarize_feature_maps"

FUNCTIONS_MARKER = "global_step = 0\n"

FUNCTIONS_BLOCK = """

# --- RVC_from_audio_to_model: feature map monitoring ---

def _summarize_feature_maps(fmap_r, fmap_g, max_discriminators=2, max_layers=2):
    summary = []
    for d_idx, (dr, dg) in enumerate(zip(fmap_r, fmap_g)):
        if d_idx >= max_discriminators:
            break
        for l_idx, (rl, gl) in enumerate(zip(dr, dg)):
            if l_idx >= max_layers:
                break
            rl_f = rl.detach().float()
            gl_f = gl.detach().float()
            summary.append({
                "disc": d_idx,
                "layer": l_idx,
                "shape_r": list(rl_f.shape),
                "shape_g": list(gl_f.shape),
                "mean_r": float(rl_f.mean().item()),
                "std_r": float(rl_f.std(unbiased=False).item()),
                "mean_g": float(gl_f.mean().item()),
                "std_g": float(gl_f.std(unbiased=False).item()),
                "l1_mean": float(torch.mean(torch.abs(rl_f - gl_f)).item()),
            })
    return summary


def _append_feature_map_summary(file_path, summary, epoch, batch_idx, global_step):
    if not summary:
        return
    file_exists = os.path.exists(file_path)
    with open(file_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "epoch", "batch_idx", "global_step", "disc", "layer",
                "shape_r", "shape_g", "mean_r", "std_r", "mean_g", "std_g", "l1_mean",
            ])
        for item in summary:
            writer.writerow([
                int(epoch), int(batch_idx), int(global_step),
                item["disc"], item["layer"],
                "x".join(map(str, item["shape_r"])),
                "x".join(map(str, item["shape_g"])),
                item["mean_r"], item["std_r"],
                item["mean_g"], item["std_g"],
                item["l1_mean"],
            ])

# --- end feature map monitoring ---
"""

# Unique string that ends the Train Epoch logger.info block in the original file.
CALL_MARKER = (
    '                    "Train Epoch: {} [{:.0f}%]".format(\n'
    "                        epoch, 100.0 * batch_idx / len(train_loader)\n"
    "                    )\n"
    "                )\n"
)

CALL_BLOCK = (
    "\n"
    "                fmap_summary = _summarize_feature_maps(fmap_r, fmap_g)\n"
    "                if len(fmap_summary) > 0:\n"
    '                    logger.info("feature_map_summary=%s" % json.dumps(fmap_summary))\n'
    "                    fmap_log_path = os.path.join(hps.model_dir, \"feature_map_summary.csv\")\n"
    "                    _append_feature_map_summary(\n"
    "                        fmap_log_path, fmap_summary, epoch, batch_idx, global_step,\n"
    "                    )\n"
)


def main():
    if not TRAIN_SCRIPT.exists():
        print(f"ERROR: {TRAIN_SCRIPT} not found.")
        print("Run: git submodule update --init")
        sys.exit(1)

    text = TRAIN_SCRIPT.read_text(encoding="utf-8")

    if ALREADY_PATCHED_MARKER in text:
        print("Already patched — nothing to do.")
        return

    if FUNCTIONS_MARKER not in text:
        print("ERROR: marker 'global_step = 0' not found. Mangio version may have changed.")
        sys.exit(1)

    if CALL_MARKER not in text:
        print("ERROR: Train Epoch logger.info marker not found. Mangio version may have changed.")
        sys.exit(1)

    text = text.replace(FUNCTIONS_MARKER, FUNCTIONS_MARKER + FUNCTIONS_BLOCK, 1)
    text = text.replace(CALL_MARKER, CALL_MARKER + CALL_BLOCK, 1)

    TRAIN_SCRIPT.write_text(text, encoding="utf-8")
    print(f"Patched: {TRAIN_SCRIPT}")


if __name__ == "__main__":
    main()
