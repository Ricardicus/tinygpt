#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a training CSV and plot loss curves."
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--input",
        help="Path to input CSV file (e.g., metrics.csv)",
    )
    inputs.add_argument(
        "--norms", metavar="CSV",
        help="Plot block or individual-parameter norms from a monitoring CSV.",
    )
    parser.add_argument(
        "--start-step",
        help="Start the plot after this step",
        default=0,
        type=int
    )
    parser.add_argument("--series", action="append", default=[], metavar="NAME",
                        help="Only plot this layer index or exact parameter name; repeat for multiple series.")
    parser.add_argument("--log-gradients", action="store_true",
                        help="Use a logarithmic gradient axis for norm plots (zero values are omitted).")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the plot image (e.g., plot.png). If omitted, shows interactively.",
    )
    args = parser.parse_args(argv)
    if not args.norms and (args.series or args.log_gradients):
        parser.error("--series and --log-gradients require --norms")
    return args


def plot_validation(args):
    input_path = Path(args.norms or args.input)

    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        return 1

    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        return 1

    if args.norms:
        try:
            plot_norms(df, input_path, args.series, args.log_gradients)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        return finish_plot(args.output)
    expected_cols = {"epoch", "global_step", "validation_loss", "perplexity", "tokens"}
    missing = expected_cols - set(df.columns)
    if missing:
        print(
            f"Error: missing required columns: {sorted(missing)}. "
            f"Found columns: {list(df.columns)}",
            file=sys.stderr,
        )
        return 1

    # Ensure numeric columns are numeric
    for col in ["epoch", "global_step", "validation_loss", "perplexity"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where both plotted values are missing
    df = df.dropna(subset=["validation_loss", "perplexity"], how="all")

    if df.empty:
        print("Error: no valid rows to plot after cleaning.", file=sys.stderr)
        return 1

    # IMPORTANT: keep original CSV order.
    # 'step' repeats per epoch, so row order is the real chronological step.
    if len(df) > args.start_step:
        df = df.iloc[args.start_step:].reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    x = df.index + 1  # global chronological step derived from row order

    fig, axs = plt.subplots(2)

    axs[0].plot(x, df["validation_loss"], label="loss")
    axs[1].plot(x, df["perplexity"], label="perplexity")

    fig.suptitle(f"Training Metrics: {input_path.name}")

    axs[0].set(ylabel="validation_loss", xlabel="step")
    axs[1].set(ylabel="perplexity", xlabel="step")

    return finish_plot(args.output)



def plot_norms(df, input_path, series=None, log_gradients=False):
    """Plot sampled pre-update parameter and pre-clipping gradient norms."""
    labels = [name for name in ("layer", "parameter") if name in df.columns]
    required = {"global_step", "parameter_norm", "gradient_norm"}
    missing = required - set(df.columns)
    if missing or len(labels) != 1:
        raise ValueError(f"Expected global_step, parameter_norm, gradient_norm and exactly one of layer or parameter; missing {sorted(missing)}")
    label = labels[0]
    df = df.copy()
    df[label] = df[label].astype("string")
    for col in ("global_step", "parameter_norm", "gradient_norm"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[label, "global_step"])
    if series:
        available = set(df[label].dropna())
        absent = set(series) - available
        if absent:
            raise ValueError(f"Unknown {label} values: {', '.join(sorted(absent))}")
        df = df[df[label].isin(series)]
    if df.empty or df[["parameter_norm", "gradient_norm"]].notna().sum().sum() == 0:
        raise ValueError("No valid norm rows to plot")
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for name, group in df.groupby(label, sort=False):
        # Preserve file order: step numbers can repeat after checkpoint resumes.
        axes[0].plot(group["global_step"], group["parameter_norm"], label=str(name))
        gradients = group["gradient_norm"]
        if log_gradients:
            gradients = gradients.where(gradients > 0)
        axes[1].plot(group["global_step"], gradients, label=str(name))
    if log_gradients:
        axes[1].set_yscale("log")
    axes[0].set_ylabel("Parameter L2 norm")
    axes[1].set_ylabel("Gradient L2 norm (before clipping)")
    axes[1].set_xlabel("Global step")
    axes[0].set_title(f"Norms before update: {input_path.name}")
    for ax in axes:
        ax.grid(True)
    axes[0].legend(title=label, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize="small")
    fig.tight_layout()
    return fig

def plot_loss(args):
    input_path = Path(args.norms or args.input)

    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        return 1

    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        return 1

    if args.norms:
        try:
            plot_norms(df, input_path, args.series, args.log_gradients)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        return finish_plot(args.output)
    expected_cols = {"epoch", "step", "loss", "mean_loss"}
    missing = expected_cols - set(df.columns)
    if missing:
        print(
            f"Error: missing required columns: {sorted(missing)}. "
            f"Found columns: {list(df.columns)}",
            file=sys.stderr,
        )
        return 1

    # Ensure numeric columns are numeric
    for col in ["epoch", "step", "loss", "mean_loss"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where both plotted values are missing
    df = df.dropna(subset=["loss", "mean_loss"], how="all")

    if df.empty:
        print("Error: no valid rows to plot after cleaning.", file=sys.stderr)
        return 1

    # IMPORTANT: keep original CSV order.
    # 'step' repeats per epoch, so row order is the real chronological step.
    if len(df) > args.start_step:
        df = df.iloc[args.start_step:].reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    x = df.index + 1  # global chronological step derived from row order

    plt.figure(figsize=(10, 6))
    plt.plot(x, df["loss"], label="loss")
    plt.plot(x, df["mean_loss"], label="mean_loss")

    plt.xlabel("step")
    plt.ylabel("value")
    plt.title(f"Training Metrics: {input_path.name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    return finish_plot(args.output)



def main() -> int:
    args = parse_args()

    if args.input and "validation" in args.input:
        plot_validation(args)
    else:
        plot_loss(args)




def finish_plot(output):
    if output:
        try:
            plt.savefig(output, dpi=150)
            print(f"Saved plot to {output}")
        except Exception as e:
            print(f"Error saving plot: {e}", file=sys.stderr)
            return 1
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
