#!/usr/bin/env python3
"""
Generate all report figures from benchmark CSVs.

This script does two things:
1. Reuses `scripts/plot_serial_csv.py` for the standard per-gamma and summary plots.
2. Generates the paper-only figures that were previously assembled manually.

The default inputs match the benchmark artifacts currently referenced by `report/paper.tex`.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_serial_csv import main as plot_serial_main


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench-results"
REPORT = ROOT / "report"
REPORT_FIGURES = REPORT / "figures"

FIXED_CSV = BENCH / "serial-fixed.csv"
AUTO_CSV = BENCH / "serial-auto.csv"
MERGED_CSV = BENCH / "serial-merged.csv"
KEYMODE_CSV = BENCH / "serial-keymode.csv"
FASTRANGE_CSV = BENCH / "serial-fastrange.csv"


def run_plot_serial(csvs: list[Path], out_dir: Path) -> None:
    import sys

    old_argv = sys.argv
    try:
        sys.argv = ["plot_serial_csv.py", *[str(p) for p in csvs], "-o", str(out_dir)]
        plot_serial_main()
    finally:
        sys.argv = old_argv


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def merge_csvs(paths: list[Path], dst_path: Path) -> None:
    header: list[str] | None = None
    rows: list[list[str]] = []
    for path in paths:
        with path.open(newline="") as f:
            reader = csv.reader(f)
            this_header = next(reader)
            if header is None:
                header = this_header
            elif header != this_header:
                raise SystemExit(f"header mismatch while merging {path}")
            rows.extend(list(reader))
    if header is None:
        raise SystemExit("no CSV inputs to merge")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def copy_standard_plots(src_dir: Path, filenames: list[str], dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        shutil.copy2(src_dir / name, dst_dir / name)


def build_gamma_grid(src_dir: Path, dst_path: Path, include_auto: bool = False) -> None:
    gamma_files = sorted(src_dir.glob("gamma_*.png"))
    if not include_auto:
        gamma_files = [p for p in gamma_files if p.name != "gamma_auto.png"]
    if not gamma_files:
        raise SystemExit(f"no gamma plots found in {src_dir}")

    ncols = 3
    nrows = int(np.ceil(len(gamma_files) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.2 * nrows), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).ravel()

    for ax, img_path in zip(axes_arr, gamma_files):
        ax.imshow(plt.imread(img_path))
        ax.set_title(img_path.stem.replace("_", " "))
        ax.axis("off")

    for ax in axes_arr[len(gamma_files):]:
        ax.axis("off")

    fig.savefig(dst_path, dpi=150)
    plt.close(fig)


def seed_variance_plot(src_csv: Path, dst_path: Path, n_target: int = 1_048_576) -> None:
    rows = load_rows(src_csv)
    buckets: dict[tuple[int, float, int], list[float]] = defaultdict(list)
    for row in rows:
        if row["impl"] != "better-mphf" or row["phase"] != "build":
            continue
        if int(row["n"]) != n_target:
            continue
        key = (int(row["n"]), float(row["gamma"]), int(row["seed_index"]))
        buckets[key].append(float(row["ms"]))

    per_seed: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for (_, gamma, seed_index), vals in buckets.items():
        per_seed[seed_index].append((gamma, statistics.median(vals)))

    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    for seed_index, pts in sorted(per_seed.items()):
        pts.sort(key=lambda t: t[0])
        ax.plot(
            [g for g, _ in pts],
            [ms for _, ms in pts],
            marker="o",
            linewidth=1.3,
            alpha=0.55,
            label=f"seed {seed_index}",
        )

    ax.set_title(f"Seed Sensitivity for better-mphf Build at n={n_target:,}")
    ax.set_xlabel("gamma")
    ax.set_ylabel("median build time per seed (ms)")
    ax.grid(True, alpha=0.35)
    ax.legend(ncol=4, fontsize=7, loc="upper left")
    fig.savefig(dst_path, dpi=150)
    plt.close(fig)


def median_by_mode(rows: list[dict[str, str]], phase: str, mode_field: str, baseline: str) -> tuple[list[str], list[float]]:
    grouped: dict[tuple[str, int, float, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["impl"] != "better-mphf" or row["phase"] != phase:
            continue
        mode = row[mode_field]
        grouped[(phase, int(row["n"]), float(row["gamma"]), mode)].append(float(row["ms"]))

    cell_medians = {key: statistics.median(vals) for key, vals in grouped.items()}
    modes = sorted({mode for (_, _, _, mode) in cell_medians})
    modes = [m for m in modes if m != baseline]
    deltas: list[float] = []
    labels: list[str] = []
    for mode in modes:
        per_cell = []
        for _, n, gamma, m in cell_medians:
            if m != mode:
                continue
            base_key = (phase, n, gamma, baseline)
            if base_key not in cell_medians:
                continue
            val = cell_medians[(phase, n, gamma, mode)]
            base = cell_medians[base_key]
            per_cell.append((val - base) / base * 100.0)
        labels.append(mode)
        deltas.append(statistics.median(per_cell))
    return labels, deltas


def fastrange_tradeoff_plot(src_csv: Path, dst_path: Path) -> None:
    rows = load_rows(src_csv)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), constrained_layout=True)
    for ax, phase in zip(axes, ["build", "lookup"]):
        labels, deltas = median_by_mode(rows, phase, "fastrange_mode", "low32")
        colors = ["#d62728" if d > 0 else "#2ca02c" for d in deltas]
        ax.bar(labels, deltas, color=colors)
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_title(f"{phase.capitalize()} vs low32")
        ax.set_ylabel("median delta (%)")
        ax.grid(True, axis="y", alpha=0.35)
    fig.suptitle("Fast-Range Comparison")
    fig.savefig(dst_path, dpi=150)
    plt.close(fig)


def keymode_tradeoff_plot(src_csv: Path, dst_path: Path) -> None:
    rows = load_rows(src_csv)
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2), constrained_layout=True)
    for ax, phase in zip(axes, ["build", "lookup"]):
        labels, deltas = median_by_mode(rows, phase, "key_mode", "multiplicative")
        colors = ["#d62728" if d > 0 else "#2ca02c" for d in deltas]
        ax.bar(labels, deltas, color=colors)
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_title(f"{phase.capitalize()} vs multiplicative")
        ax.set_ylabel("median delta (%)")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.35)
    fig.suptitle("Key-Distribution Comparison")
    fig.savefig(dst_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixed-csv", type=Path, default=FIXED_CSV)
    ap.add_argument("--auto-csv", type=Path, default=AUTO_CSV)
    ap.add_argument("--merged-csv", type=Path, default=MERGED_CSV)
    ap.add_argument("--keymode-csv", type=Path, default=KEYMODE_CSV)
    ap.add_argument("--fastrange-csv", type=Path, default=FASTRANGE_CSV)
    ap.add_argument("--merged-plots-dir", type=Path, default=BENCH / "plots-serial-merged")
    ap.add_argument("--report-figures-dir", type=Path, default=REPORT_FIGURES)
    return ap.parse_args()


def ensure_files(paths: list[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit("missing required input(s):\n" + "\n".join(missing))


def main() -> None:
    args = parse_args()
    ensure_files(
        [
            args.fixed_csv,
            args.auto_csv,
            args.keymode_csv,
            args.fastrange_csv,
        ]
    )

    merge_csvs([args.fixed_csv, args.auto_csv], args.merged_csv)
    run_plot_serial([args.merged_csv], args.merged_plots_dir)

    copy_standard_plots(
        args.merged_plots_dir,
        [
            "best_mode_build.png",
            "best_mode_lookup.png",
            "better_auto_vs_fixed_build.png",
            "better_auto_vs_fixed_lookup.png",
        ],
        args.report_figures_dir,
    )

    build_gamma_grid(args.merged_plots_dir, args.report_figures_dir / "gamma_grid_serial.png")
    seed_variance_plot(args.fixed_csv, args.report_figures_dir / "seed_variance_1m_build.png")
    fastrange_tradeoff_plot(args.fastrange_csv, args.report_figures_dir / "fastrange_tradeoff.png")
    keymode_tradeoff_plot(args.keymode_csv, args.report_figures_dir / "keymode_tradeoff.png")

    print(args.merged_csv)
    print(args.report_figures_dir / "best_mode_build.png")
    print(args.report_figures_dir / "best_mode_lookup.png")
    print(args.report_figures_dir / "better_auto_vs_fixed_build.png")
    print(args.report_figures_dir / "better_auto_vs_fixed_lookup.png")
    print(args.report_figures_dir / "gamma_grid_serial.png")
    print(args.report_figures_dir / "seed_variance_1m_build.png")
    print(args.report_figures_dir / "fastrange_tradeoff.png")
    print(args.report_figures_dir / "keymode_tradeoff.png")


if __name__ == "__main__":
    main()
