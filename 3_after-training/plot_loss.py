import argparse
import os
import re
import glob
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

OUTPUT_DIR = ""
HTML_OUTPUT = ""

# Regex to extract metrics from log lines, e.g.:
# "loss_disc=0.123 loss_gen=0.456 loss_fm=0
METRIC_REGEX = re.compile(
    r"(loss_disc|loss_gen_all|loss_gen|loss_fm|loss_mel|loss_kl)=([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
)

METRICS = ["loss_disc", "loss_gen", "loss_gen_all", "loss_fm", "loss_mel", "loss_kl"]

# Functions to parse logs, plot graphs, and build HTML report

def parse_log_file(path):
    """Estrae le metriche da un file train.log."""
    model_name = os.path.basename(path).replace(".log", "")
    epochs = []
    metrics = {name: [] for name in METRICS}

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    current_epoch = None
    for line in lines:
        if "Train Epoch:" in line:
            epoch_match = re.search(r"Train Epoch:\s*(\d+)", line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))

        matches = METRIC_REGEX.findall(line)
        if matches and current_epoch is not None:
            values_map = {name: float(value) for name, value in matches}
            for name in METRICS:
                metrics[name].append(values_map.get(name))
            epochs.append(current_epoch)

    return model_name, epochs, metrics


def plot_single_model(model_name, epochs, metrics):
    """Grafici di ogni singola metrica per un modello."""
    file_paths = []
    for metric_name, values in metrics.items():
        if all(v is None for v in values):
            continue

        filtered_epochs = [e for e, v in zip(epochs, values) if v is not None]
        filtered_values = [v for v in values if v is not None]
        if not filtered_values:
            continue

        plt.figure(figsize=(6, 4))
        plt.plot(filtered_epochs, filtered_values)
        plt.title(f"{model_name} - {metric_name}")
        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        out = f"{OUTPUT_DIR}/{model_name}_{metric_name}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        file_paths.append(out)
    return file_paths


def plot_combined(models_data):
    """Grafici con metriche sovrapposte tra modelli."""
    combined_paths = []

    metrics_list = METRICS

    for metric in metrics_list:
        plotted_any = False
        plt.figure(figsize=(6,4))
        for model_name, epochs, metrics in models_data:
            values = metrics.get(metric, [])
            if all(v is None for v in values):
                continue

            filtered_epochs = [e for e, v in zip(epochs, values) if v is not None]
            filtered_values = [v for v in values if v is not None]
            if not filtered_values:
                continue

            plt.plot(filtered_epochs, filtered_values, label=model_name)
            plotted_any = True

        if not plotted_any:
            plt.close()
            continue

        plt.title(f"Confronto - {metric}")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend()

        out = f"{OUTPUT_DIR}/combined_{metric}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        combined_paths.append(out)

    return combined_paths


def build_html(image_paths):
    """
    Build an HTML report with the generated images.
    Layout:
    - each row contains the 3 graphs for the same metric (one for each model)
    - at the end the combined graphs
    """
    soup = BeautifulSoup(
        "<html><head><meta charset='utf-8'><title>Training Report</title></head><body></body></html>",
        "html.parser"
    )
    body = soup.body

    metrics_map = {}
    combined_imgs = []

    for img in image_paths:
        filename = os.path.basename(img)

        if filename.startswith("combined_"):
            combined_imgs.append(img)
            continue

        metric = None
        for known_metric in METRICS:
            if filename.endswith(f"_{known_metric}.png"):
                metric = known_metric
                break

        if metric:
            metrics_map.setdefault(metric, []).append(img)

    # Order metrics as defined in METRICS list
    ordered_metrics = METRICS

    # CSS
    style = soup.new_tag("style")
    style.string = """
        body { font-family: Arial, sans-serif; padding: 16px; }
        .row { display:flex; gap:10px; margin-bottom:24px; }
        .row img { width:33%; border:1px solid #aaa; }
    """
    soup.head.append(style)

    # Row for each metric
    for metric in ordered_metrics:
        if metric not in metrics_map:
            continue

        header = soup.new_tag("h2")
        header.string = metric
        body.append(header)

        row = soup.new_tag("div", **{"class": "row"})

        imgs_sorted = sorted(metrics_map[metric])

        for img in imgs_sorted:
            rel = os.path.relpath(img, os.path.dirname(os.path.abspath(HTML_OUTPUT)))
            row.append(soup.new_tag("img", src=rel))

        body.append(row)

    # Combined graphs at the end
    comb_header = soup.new_tag("h1")
    comb_header.string = "Confronto tra modelli"
    body.append(comb_header)

    for img in combined_imgs:
        row = soup.new_tag("div", **{"class": "row"})
        rel = os.path.relpath(img, os.path.dirname(os.path.abspath(HTML_OUTPUT)))
        row.append(soup.new_tag("img", src=rel, style="width:50%;"))
        body.append(row)

    with open(HTML_OUTPUT, "w", encoding="utf-8") as f:
        f.write(str(soup))

    print("[OK] HTML generato:", HTML_OUTPUT)


# MAIN

def main(logs_folder):
    log_files = glob.glob(os.path.join(logs_folder, "*.log"))
    models_data = []

    for log in log_files:
        model_name, epochs, metrics = parse_log_file(log)
        models_data.append((model_name, epochs, metrics))

    all_images = []

    # Single model graphs
    for model_name, epochs, metrics in models_data:
        imgs = plot_single_model(model_name, epochs, metrics)
        all_images.extend(imgs)

    # Combined graphs
    combined = plot_combined(models_data)
    all_images.extend(combined)

    # HTML report
    build_html(all_images)

    print(f"Report generato in: {HTML_OUTPUT}")
    print(f"Immagini salvate in: {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_folder", required=True, help="Folder containing .log files")
    parser.add_argument("--output_dir", required=True, help="Folder to save generated images")
    parser.add_argument("--html_output", required=True, help="Path for the HTML report")
    args = parser.parse_args()

    OUTPUT_DIR = args.output_dir
    HTML_OUTPUT = args.html_output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(HTML_OUTPUT)), exist_ok=True)

    main(args.logs_folder)
