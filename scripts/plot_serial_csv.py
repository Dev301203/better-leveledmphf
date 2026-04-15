#!/usr/bin/env python3
"""
Plot serial benchmark CSV(s): one PNG per gamma with two panels (build | lookup),
plus summary plots for best-mode comparisons and better-mphf auto-vs-fixed.

CSV columns: impl,phase,n,gamma,threads,trial,ms

Usage:
  pip install matplotlib numpy
  python3 scripts/plot_serial_csv.py bench-results/serial-bench-154021.csv
  python3 scripts/plot_serial_csv.py run1.csv run2.csv -o bench-results/plots

Multiple CSVs overlay the same axes (labels include the file stem).
Gamma value 0.0 is reserved for better-mphf auto-gamma runs.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


IMPL_ORDER = ["better-mphf", "better-mphf-auto", "boomphf-rust", "bbhash-cpp"]
IMPL_STYLES = {
    "better-mphf": {"color": "#1f77b4", "marker": "o"},
    "better-mphf-auto": {"color": "#17becf", "marker": "D"},
    "boomphf-rust": {"color": "#ff7f0e", "marker": "s"},
    "bbhash-cpp": {"color": "#2ca02c", "marker": "^"},
}
FILE_LINESTYLES = ["-", "--", "-.", ":"]


def _gamma_slug(g: float) -> str:
    if g == 0.0:
        return "auto"
    s = f"{g:.6f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def _gamma_title(g: float) -> str:
    return "auto" if g == 0.0 else f"{g:g}"


def load_rows(paths: list[Path], threads: int) -> list[tuple[str, str, str, int, float, float]]:
    out: list[tuple[str, str, str, int, float, float]] = []
    for path in paths:
        stem = path.stem
        with path.open(newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                if int(row["threads"]) != threads:
                    continue
                out.append(
                    (
                        stem,
                        row["impl"],
                        row["phase"],
                        int(row["n"]),
                        float(row["gamma"]),
                        float(row["ms"]),
                    )
                )
    return out


def aggregate(
    rows: list[tuple[str, str, str, int, float, float]],
    stat: str,
) -> dict[tuple[float, str, str, str, int], float]:
    buckets: dict[tuple[float, str, str, str, int], list[float]] = defaultdict(list)
    for stem, impl, phase, n, gamma, ms in rows:
        buckets[(gamma, stem, impl, phase, n)].append(ms)
    agg: dict[tuple[float, str, str, str, int], float] = {}
    for k, vals in buckets.items():
        if stat == "median":
            agg[k] = float(statistics.median(vals))
        else:
            agg[k] = float(statistics.mean(vals))
    return agg


def plot_gamma(
    gamma: float,
    agg: dict[tuple[float, str, str, str, int], float],
    csv_stems: list[str],
    out_path: Path,
    stat_label: str,
) -> None:
    phases = ["build", "lookup"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    fig.suptitle(f"gamma = {_gamma_title(gamma)}  ({stat_label} ms across trials)", fontsize=12)

    for ax, phase in zip(axes, phases):
        for f_i, stem in enumerate(csv_stems):
            ls = FILE_LINESTYLES[f_i % len(FILE_LINESTYLES)]
            for impl in IMPL_ORDER:
                pts = [
                    (n, agg[(gamma, stem, impl, phase, n)])
                    for (g, s, im, ph, n) in agg
                    if g == gamma and s == stem and im == impl and ph == phase
                ]
                if not pts:
                    continue
                pts.sort(key=lambda t: t[0])
                ns = np.array([p[0] for p in pts])
                ys = np.array([p[1] for p in pts])
                sty = IMPL_STYLES.get(impl, {"color": None, "marker": "o"})
                label = impl if len(csv_stems) == 1 else f"{impl} [{stem}]"
                ax.plot(
                    ns,
                    ys,
                    linestyle=ls,
                    marker=sty["marker"],
                    color=sty["color"],
                    markersize=5 if len(ns) < 20 else 4,
                    linewidth=1.8,
                    label=label,
                )

        ax.set_xscale("log", base=2)
        ax.set_xlabel("n (keys)")
        ax.set_ylabel("time (ms)")
        ax.set_title(phase.capitalize())
        ax.grid(True, which="both", alpha=0.35)
        ax.legend(fontsize=8, loc="best")

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def best_points(
    agg: dict[tuple[float, str, str, str, int], float],
    stem: str,
    impl: str,
    phase: str,
) -> list[tuple[int, float, float]]:
    pts: list[tuple[int, float, float]] = []
    ns = sorted({n for (g, s, im, ph, n) in agg if s == stem and im == impl and ph == phase})
    for n in ns:
        candidates = [
            (g, agg[(g, stem, impl, phase, n)])
            for (g, s, im, ph, nn) in agg
            if s == stem and im == impl and ph == phase and nn == n
        ]
        if not candidates:
            continue
        gamma, value = min(candidates, key=lambda item: item[1])
        pts.append((n, value, gamma))
    return pts


def plot_single_phase_summary(
    series: list[tuple[str, dict[str, object]]],
    phase: str,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.2), constrained_layout=True)
    ax.set_title(title)
    for label, meta in series:
        pts = meta["points"]
        if not pts:
            continue
        ns = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        ax.plot(
            ns,
            ys,
            linestyle=meta.get("linestyle", "-"),
            marker=meta.get("marker", "o"),
            color=meta.get("color"),
            linewidth=1.8,
            markersize=5 if len(ns) < 20 else 4,
            label=label,
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("n (keys)")
    ax.set_ylabel("time (ms)")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(fontsize=8, loc="best")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def emit_summary_plots(
    agg: dict[tuple[float, str, str, str, int], float],
    stems: list[str],
    out_dir: Path,
) -> list[Path]:
    outputs: list[Path] = []
    for stem in stems:
        stem_suffix = "" if len(stems) == 1 else f"_{stem}"
        for phase in ["build", "lookup"]:
            auto_points = best_points(agg, stem, "better-mphf-auto", phase)
            fixed_points = best_points(agg, stem, "better-mphf", phase)
            practical_series: list[tuple[str, dict[str, object]]] = []
            for impl in ["better-mphf-auto", "better-mphf", "boomphf-rust", "bbhash-cpp"]:
                pts = best_points(agg, stem, impl, phase)
                if not pts:
                    continue
                style = IMPL_STYLES[impl].copy()
                practical_series.append((impl, {"points": pts, **style}))
            out_best = out_dir / f"best_mode_{phase}{stem_suffix}.png"
            plot_single_phase_summary(
                practical_series,
                phase,
                f"Best-mode {phase} ({stem})" if len(stems) > 1 else f"Best-mode {phase}",
                out_best,
            )
            outputs.append(out_best)

            better_series: list[tuple[str, dict[str, object]]] = []
            if auto_points:
                better_series.append((
                    "better-mphf-auto",
                    {"points": auto_points, **IMPL_STYLES["better-mphf-auto"], "linestyle": "-"},
                ))
            if fixed_points:
                better_series.append((
                    "better-mphf (best fixed)",
                    {"points": fixed_points, **IMPL_STYLES["better-mphf"], "linestyle": "--"},
                ))
            out_better = out_dir / f"better_auto_vs_fixed_{phase}{stem_suffix}.png"
            plot_single_phase_summary(
                better_series,
                phase,
                f"better-mphf auto vs fixed {phase} ({stem})" if len(stems) > 1 else f"better-mphf auto vs fixed {phase}",
                out_better,
            )
            outputs.append(out_better)
    return outputs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "csv",
        nargs="+",
        type=Path,
        help="Serial benchmark CSV file(s) from bench runners",
    )
    ap.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for PNGs (default: <first_csv_dir>/plots-serial)",
    )
    ap.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Filter rows to this thread count (default: 1)",
    )
    ap.add_argument(
        "--stat",
        choices=["median", "mean"],
        default="median",
        help="Aggregate trials with median or mean (default: median)",
    )
    args = ap.parse_args()
    paths = [p.resolve() for p in args.csv]
    for p in paths:
        if not p.is_file():
            raise SystemExit(f"not a file: {p}")

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = paths[0].parent / "plots-serial"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(paths, args.threads)
    if not rows:
        raise SystemExit("no rows after filter; check --threads and CSV contents")

    agg = aggregate(rows, args.stat)
    gammas = sorted({g for (g, _, _, _, _) in agg.keys()})
    stems = list(dict.fromkeys(p.stem for p in paths))

    stat_label = args.stat.capitalize()
    for gamma in gammas:
        slug = _gamma_slug(gamma)
        out_png = out_dir / f"gamma_{slug}.png"
        plot_gamma(gamma, agg, stems, out_png, stat_label)
        print(out_png)

    for extra in emit_summary_plots(agg, stems, out_dir):
        print(extra)

    _write_index_html(out_dir, gammas, stems)


def _write_index_html(out_dir: Path, gammas: list[float], stems: list[str]) -> None:
    html = out_dir / "index.html"
    lines = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Objective serial plots</title>",
        "<style>body{font-family:sans-serif;margin:1.5rem;} img{max-width:100%;border:1px solid #ccc;margin:1rem 0;}</style>",
        "</head><body><h1>Objective serial plots</h1>",
        "<p>Per-gamma fairness plots plus best-mode and auto-vs-fixed summaries.</p>",
        "<h2>Summary</h2>",
    ]
    for stem in stems:
        stem_suffix = "" if len(stems) == 1 else f"_{stem}"
        if len(stems) > 1:
            lines.append(f"<h3>{stem}</h3>")
        for name in [
            f"best_mode_build{stem_suffix}.png",
            f"best_mode_lookup{stem_suffix}.png",
            f"better_auto_vs_fixed_build{stem_suffix}.png",
            f"better_auto_vs_fixed_lookup{stem_suffix}.png",
        ]:
            lines.append(f"<img src='{name}' alt='{name}'/>")
    lines.append("<h2>By gamma</h2>")
    for g in gammas:
        slug = _gamma_slug(g)
        fn = f"gamma_{slug}.png"
        lines.append(f"<h3>gamma = {_gamma_title(g)}</h3>")
        lines.append(f"<img src='{fn}' alt='gamma {_gamma_title(g)}'/>")
    lines.append("</body></html>")
    html.write_text("\n".join(lines), encoding="utf-8")
    print(html)


if __name__ == "__main__":
    main()
