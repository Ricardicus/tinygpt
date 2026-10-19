#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a training CSV and plot loss curves."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV file (e.g., metrics.csv)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the plot image (e.g., plot.png). If omitted, shows interactively.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        return 1

    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        return 1

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

    if args.output:
        try:
            plt.savefig(args.output, dpi=150)
            print(f"Saved plot to {args.output}")
        except Exception as e:
            print(f"Error saving plot: {e}", file=sys.stderr)
            return 1
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())