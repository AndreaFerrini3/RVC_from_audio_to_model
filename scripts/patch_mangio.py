"""
Patches Mangio scripts:
  - train_nsf_sim_cache_sid_load_pretrain.py : inject feature map monitoring
  - infer_batch_rvc.py                       : fix load_audio() signature mismatch
  - fairseq/checkpoint_utils.py              : force weights_only=False
                                                (torch 2.6+ defaults to True and
                                                 refuses fairseq Dictionary in
                                                 hubert_base.pt)

Run once after: git submodule update --init
"""
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
TRAIN_SCRIPT = REPO_ROOT / "mangio" / "train_nsf_sim_cache_sid_load_pretrain.py"
INFER_SCRIPT = REPO_ROOT / "mangio" / "infer_batch_rvc.py"

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


def patch_train():
    if not TRAIN_SCRIPT.exists():
        print(f"ERROR: {TRAIN_SCRIPT} not found.")
        print("Run: git submodule update --init")
        sys.exit(1)

    text = TRAIN_SCRIPT.read_text(encoding="utf-8")

    if ALREADY_PATCHED_MARKER in text:
        print(f"[train] already patched — skipping")
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
    print(f"[train] patched: {TRAIN_SCRIPT}")


INFER_PATCHES = [
    # (description, old, new)
    (
        "load_audio signature",
        "audio = load_audio(input_audio, 16000)",
        'audio = load_audio(input_audio, 16000, "false", 1.0, 1.0)',
    ),
    (
        "crepe_hop_length CLI arg",
        "protect = float(sys.argv[13])\nprint(sys.argv)",
        "protect = float(sys.argv[13])\n"
        "crepe_hop_length = int(sys.argv[14]) if len(sys.argv) > 14 else 160\n"
        "print(sys.argv)",
    ),
    (
        "crepe_hop_length pipeline arg",
        "        version,\n        protect,\n        f0_file=f0_file,\n    )",
        "        version,\n        protect,\n        crepe_hop_length,\n        f0_file=f0_file,\n    )",
    ),
]


def patch_infer():
    if not INFER_SCRIPT.exists():
        print(f"[infer] {INFER_SCRIPT} not found — skipping")
        return

    text = INFER_SCRIPT.read_text(encoding="utf-8")
    changed = False
    for label, old, new in INFER_PATCHES:
        if new in text:
            print(f"[infer] {label} already patched — skipping")
            continue
        if old not in text:
            print(f"[infer] WARNING: marker for '{label}' not found — Mangio version may have changed")
            continue
        text = text.replace(old, new, 1)
        changed = True
        print(f"[infer] applied: {label}")

    if changed:
        INFER_SCRIPT.write_text(text, encoding="utf-8")
        print(f"[infer] wrote: {INFER_SCRIPT}")


FAIRSEQ_OLD = 'state = torch.load(f, map_location=torch.device("cpu"))'
FAIRSEQ_NEW = 'state = torch.load(f, map_location=torch.device("cpu"), weights_only=False)'


def patch_fairseq():
    """Force weights_only=False in fairseq.checkpoint_utils.load_checkpoint_to_cpu.

    torch 2.6+ flipped torch.load's default to weights_only=True for safety.
    hubert_base.pt contains a fairseq.data.dictionary.Dictionary object that is
    not on the safe-globals whitelist, so the load is rejected. The .pt comes
    from the official HuggingFace repo (lj1995/VoiceConversionWebUI), so we
    trust it and override the default.
    """
    spec = importlib.util.find_spec("fairseq")
    if spec is None or spec.origin is None:
        print("[fairseq] not installed — skipping (run setup.py first)")
        return
    cu_path = Path(spec.origin).parent / "checkpoint_utils.py"
    if not cu_path.exists():
        print(f"[fairseq] {cu_path} not found — skipping")
        return

    text = cu_path.read_text(encoding="utf-8")
    if FAIRSEQ_NEW in text:
        print(f"[fairseq] already patched — skipping")
        return
    if FAIRSEQ_OLD not in text:
        print(f"[fairseq] WARNING: marker not found in {cu_path}")
        print(f"[fairseq] expected: {FAIRSEQ_OLD}")
        print(f"[fairseq] fairseq version may differ — patch manually if torch.load fails")
        return

    text = text.replace(FAIRSEQ_OLD, FAIRSEQ_NEW, 1)
    cu_path.write_text(text, encoding="utf-8")
    print(f"[fairseq] patched: {cu_path}")


def main():
    patch_train()
    patch_infer()
    patch_fairseq()


if __name__ == "__main__":
    main()
