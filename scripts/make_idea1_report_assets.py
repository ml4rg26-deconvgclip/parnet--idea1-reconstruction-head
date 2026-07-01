#!/usr/bin/env python
"""Create lightweight report assets for Idea 1 globalCLIP reconstruction runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def _require_file(path: Path) -> Path:
    """Return an existing path or raise a clear error."""
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    return path


def _require_columns(df: pd.DataFrame, path: Path, columns: list[str]) -> None:
    """Validate that a dataframe has required columns."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def _method_label(method: str) -> str:
    """Make method names compact enough for figure axes."""
    return method.replace("_", "\n")


def _weight_label(row: pd.Series) -> str:
    """Return a readable label for one reconstruction-weight row."""
    rbp_name = str(row.get("rbp_name", "") or "").strip()
    cell_line = str(row.get("cell_line", "") or "").strip()
    track_name = str(row.get("experiment_id_or_track_name", "") or "").strip()
    track_index = row.get("track_index", "")

    if rbp_name and cell_line:
        return f"{rbp_name} ({cell_line})"
    if rbp_name:
        return rbp_name
    if track_name:
        return track_name
    return f"track {track_index}"


def _to_jsonable(value: Any) -> Any:
    """Convert pandas/nan values into JSON-safe primitives."""
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _records_jsonable(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Return dataframe records with JSON-safe scalar values."""
    return [
        {key: _to_jsonable(value) for key, value in row.items()}
        for row in df.to_dict(orient="records")
    ]


def _best_epoch_metric(
    training_df: pd.DataFrame,
    *,
    metric: str,
    maximize: bool,
) -> dict[str, Any] | None:
    """Return the best epoch for one training metric."""
    valid_df = training_df.dropna(subset=[metric])
    if valid_df.empty:
        return None
    row = valid_df.loc[valid_df[metric].idxmax() if maximize else valid_df[metric].idxmin()]
    return {
        "epoch": _to_jsonable(row["epoch"]),
        metric: _to_jsonable(row[metric]),
    }


def _plot_training_curve(training_df: pd.DataFrame, output_path: Path) -> None:
    """Plot validation Pearson and validation loss over epochs."""
    fig, ax_loss = plt.subplots(figsize=(8, 4.8))
    ax_pearson = ax_loss.twinx()

    ax_loss.plot(
        training_df["epoch"],
        training_df["valid_loss"],
        marker="o",
        color="#d55e00",
        label="valid loss",
    )
    ax_pearson.plot(
        training_df["epoch"],
        training_df["valid_pearson"],
        marker="o",
        color="#0072b2",
        label="valid Pearson",
    )

    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("valid loss", color="#d55e00")
    ax_pearson.set_ylabel("valid Pearson", color="#0072b2")
    ax_loss.tick_params(axis="y", labelcolor="#d55e00")
    ax_pearson.tick_params(axis="y", labelcolor="#0072b2")
    ax_loss.grid(axis="y", alpha=0.25)

    handles_loss, labels_loss = ax_loss.get_legend_handles_labels()
    handles_pearson, labels_pearson = ax_pearson.get_legend_handles_labels()
    ax_loss.legend(
        handles_loss + handles_pearson,
        labels_loss + labels_pearson,
        loc="best",
        frameon=False,
    )
    fig.suptitle("Training curve")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_metric_bar(
    eval_df: pd.DataFrame,
    *,
    metric: str,
    output_path: Path,
    title: str,
    ylabel: str,
    lower_is_better: bool,
) -> None:
    """Plot one evaluation metric by method."""
    plot_df = eval_df[["method", metric]].dropna().copy()
    plot_df = plot_df.sort_values(metric, ascending=lower_is_better)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(
        [_method_label(method) for method in plot_df["method"]],
        plot_df[metric],
        color="#4c78a8",
    )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_top_weights(weights_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Plot the top 10 track-level reconstruction weights."""
    top_df = weights_df.sort_values("weight", ascending=False).head(10).copy()
    plot_df = top_df.iloc[::-1]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.barh(
        [_weight_label(row) for _, row in plot_df.iterrows()],
        plot_df["weight"],
        color="#59a14f",
    )
    ax.set_xlabel("reconstruction weight")
    ax.set_title("Top 10 reconstruction weights")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return top_df


def _summary_numbers(
    training_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    top_weights_df: pd.DataFrame,
) -> dict[str, Any]:
    """Build a compact summary JSON payload."""
    final_row = training_df.sort_values("epoch").iloc[-1]

    eval_metrics = eval_df.set_index("method").to_dict(orient="index")
    trained = eval_metrics.get("trained_reconstruction_head", {})
    baseline_rows = eval_df[eval_df["method"] != "trained_reconstruction_head"].copy()

    best_baseline_pearson = None
    best_baseline_ce = None
    if not baseline_rows.empty:
        pearson_rows = baseline_rows.dropna(subset=["mean_pearson"])
        if not pearson_rows.empty:
            row = pearson_rows.loc[pearson_rows["mean_pearson"].idxmax()]
            best_baseline_pearson = {
                "method": _to_jsonable(row["method"]),
                "mean_pearson": _to_jsonable(row["mean_pearson"]),
            }
        ce_rows = baseline_rows.dropna(subset=["profile_cross_entropy"])
        if not ce_rows.empty:
            row = ce_rows.loc[ce_rows["profile_cross_entropy"].idxmin()]
            best_baseline_ce = {
                "method": _to_jsonable(row["method"]),
                "profile_cross_entropy": _to_jsonable(row["profile_cross_entropy"]),
            }

    return {
        "final_epoch": _to_jsonable(final_row["epoch"]),
        "final_valid_loss": _to_jsonable(final_row["valid_loss"]),
        "final_valid_pearson": _to_jsonable(final_row["valid_pearson"]),
        "best_valid_pearson": _best_epoch_metric(
            training_df,
            metric="valid_pearson",
            maximize=True,
        ),
        "best_valid_loss": _best_epoch_metric(
            training_df,
            metric="valid_loss",
            maximize=False,
        ),
        "trained_reconstruction_head": {
            key: _to_jsonable(value) for key, value in trained.items()
        },
        "best_baseline_by_pearson": best_baseline_pearson,
        "best_baseline_by_cross_entropy": best_baseline_ce,
        "evaluation_by_method": {
            method: {key: _to_jsonable(value) for key, value in metrics.items()}
            for method, metrics in eval_metrics.items()
        },
        "top_10_track_weights": _records_jsonable(top_weights_df),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory containing training/evaluation CSV outputs.",
    )
    return parser.parse_args()


def main() -> None:
    """Create report assets under <run-dir>/report_assets."""
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    assets_dir = run_dir / "report_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    training_path = _require_file(run_dir / "training_metrics.csv")
    weights_path = _require_file(run_dir / "reconstruction_weights_track_level.csv")
    evaluation_path = _require_file(
        run_dir / "evaluation_valid" / "interphase" / "evaluation_metrics.csv"
    )

    training_df = pd.read_csv(training_path)
    weights_df = pd.read_csv(weights_path)
    eval_df = pd.read_csv(evaluation_path)

    _require_columns(training_df, training_path, ["epoch", "valid_loss", "valid_pearson"])
    _require_columns(weights_df, weights_path, ["track_index", "weight"])
    _require_columns(eval_df, evaluation_path, ["method", "mean_pearson", "profile_cross_entropy"])

    _plot_training_curve(training_df, assets_dir / "training_curve.png")
    _plot_metric_bar(
        eval_df,
        metric="mean_pearson",
        output_path=assets_dir / "baseline_pearson.png",
        title="Evaluation baseline comparison: Pearson",
        ylabel="mean Pearson",
        lower_is_better=False,
    )
    _plot_metric_bar(
        eval_df,
        metric="profile_cross_entropy",
        output_path=assets_dir / "baseline_ce.png",
        title="Evaluation baseline comparison: profile cross entropy",
        ylabel="profile cross entropy",
        lower_is_better=True,
    )
    top_weights_df = _plot_top_weights(weights_df, assets_dir / "top_weights.png")

    summary = _summary_numbers(training_df, eval_df, top_weights_df)
    with (assets_dir / "summary_numbers.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Saved report assets to: {assets_dir}")


if __name__ == "__main__":
    main()
