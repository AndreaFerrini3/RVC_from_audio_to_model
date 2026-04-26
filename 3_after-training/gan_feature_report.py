import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

def load_csvs(folder_path):
    folder = Path(folder_path)
    csv_files = list(folder.glob("*.csv"))
    if not csv_files:
        raise ValueError("No CSV files found in the specified folder.")

    dfs = [pd.read_csv(f) for f in csv_files]
    return pd.concat(dfs, ignore_index=True)


def aggregate_metrics(df):
    df["mean_diff"] = (df["mean_r"] - df["mean_g"]).abs()
    df["std_diff"] = (df["std_r"] - df["std_g"]).abs()

    summary = (
        df.groupby(["epoch", "disc", "layer"])
        .agg(
            l1_mean_avg=("l1_mean", "mean"),
            l1_mean_std=("l1_mean", "std"),
            mean_diff_avg=("mean_diff", "mean"),
            std_diff_avg=("std_diff", "mean"),
        )
        .reset_index()
    )

    # Global trend: average across all discriminators and layers for each epoch
    global_epoch = (
        summary.groupby("epoch")
        .agg(
            global_l1_mean=("l1_mean_avg", "mean"),
            global_l1_std=("l1_mean_avg", "std"),
        )
        .reset_index()
    )

    return summary, global_epoch


def create_global_plot(global_df):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=global_df["epoch"],
            y=global_df["global_l1_mean"],
            mode="lines+markers",
            name="Global L1 Mean"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=global_df["epoch"],
            y=global_df["global_l1_mean"] + global_df["global_l1_std"],
            mode="lines",
            name="+1 std",
            line=dict(dash="dash")
        )
    )

    fig.add_trace(
        go.Scatter(
            x=global_df["epoch"],
            y=global_df["global_l1_mean"] - global_df["global_l1_std"],
            mode="lines",
            name="-1 std",
            line=dict(dash="dash")
        )
    )

    fig.update_layout(
        title="Global L1 Trend (All Discriminators & Layers)",
        xaxis_title="Epoch",
        yaxis_title="L1 Mean"
    )

    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def create_layer_plots(summary_df):
    plots = []

    for disc in summary_df["disc"].unique():
        sub = summary_df[summary_df["disc"] == disc]

        fig = px.line(
            sub,
            x="epoch",
            y="l1_mean_avg",
            color="layer",
            title=f"L1 Trend - Discriminator {disc}",
            markers=True,
        )

        plots.append(fig.to_html(full_html=False, include_plotlyjs=False))

    return plots


def generate_html(global_plot, summary_df, layer_plots, output_file):
    html = """
    <html>
    <head>
        <title>GAN Feature Matching Report</title>
        <style>
            body { font-family: Arial; margin: 40px; }
            h1 { margin-bottom: 10px; }
            table { border-collapse: collapse; width: 100%; margin-top: 40px; }
            th, td { border: 1px solid #ccc; padding: 6px; text-align: center; }
            th { background-color: #f4f4f4; }
        </style>
    </head>
    <body>
    <h1>GAN Global Stability Overview</h1>
    """

    # Global plot at the top
    html += global_plot

    html += "<h2>Aggregated Metrics Table</h2>"
    html += summary_df.to_html(index=False, float_format="%.6f")

    html += "<h2>Layer Trends</h2>"
    for plot in layer_plots:
        html += plot

    html += "</body></html>"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", required=True)
    parser.add_argument("--output", default="report.html")
    args = parser.parse_args()

    df = load_csvs(args.input_folder)
    summary, global_epoch = aggregate_metrics(df)

    global_plot = create_global_plot(global_epoch)
    layer_plots = create_layer_plots(summary)

    generate_html(global_plot, summary, layer_plots, args.output)

    print(f"Report generato: {args.output}")


if __name__ == "__main__":
    main()