#!/usr/bin/env python3
r"""Generate benchmark-derived numbers used by report/paper.tex.

Outputs:
- report/generated_paper_numbers.tex
- report/generated_paper_numbers.json

The TeX file is intended to be \input{} by the paper so benchmark/result
numbers do not need to be maintained by hand.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
BENCH = ROOT / "bench-results"
REPORT = ROOT / "report"

FIXED_CSV = BENCH / "serial-fixed.csv"
AUTO_CSV = BENCH / "serial-auto.csv"
KEYMODE_CSV = BENCH / "serial-keymode.csv"
FASTRANGE_CSV = BENCH / "serial-fastrange.csv"
TEX_OUT = REPORT / "generated_paper_numbers.tex"
JSON_OUT = REPORT / "generated_paper_numbers.json"

LARGE_N_THRESHOLD = 65_536


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def collapse_trials(rows: list[dict[str, str]], group_keys: list[str]) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in group_keys)].append(float(row["ms"]))

    out: list[dict[str, object]] = []
    for key, vals in groups.items():
        item: dict[str, object] = {k: v for k, v in zip(group_keys, key)}
        item["ms"] = statistics.median(vals)
        out.append(item)
    return out


def wins_by_group(rows: list[dict[str, object]], group_keys: list[str], winner_field: str) -> tuple[Counter, int]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in group_keys)].append(row)

    wins: Counter = Counter()
    ties = 0
    for grp in groups.values():
        best = min(float(row["ms"]) for row in grp)
        winners: list[str] = []
        for row in grp:
            if abs(float(row["ms"]) - best) < 1e-12 and row[winner_field] not in winners:
                winners.append(str(row[winner_field]))
        if len(winners) == 1:
            wins[winners[0]] += 1
        else:
            ties += 1
    return wins, ties


def fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def fmt_float(x: float, digits: int) -> str:
    return f"{x:.{digits}f}"


def fmt_float_signed(x: float, digits: int) -> str:
    return f"{x:+.{digits}f}"


def ensure_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit("missing required input(s):\n" + "\n".join(missing))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixed-csv", type=Path, default=FIXED_CSV)
    ap.add_argument("--auto-csv", type=Path, default=AUTO_CSV)
    ap.add_argument("--keymode-csv", type=Path, default=KEYMODE_CSV)
    ap.add_argument("--fastrange-csv", type=Path, default=FASTRANGE_CSV)
    ap.add_argument("--tex-out", type=Path, default=TEX_OUT)
    ap.add_argument("--json-out", type=Path, default=JSON_OUT)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ensure_files([
        args.fixed_csv,
        args.auto_csv,
        args.keymode_csv,
        args.fastrange_csv,
    ])

    fixed_rows = load_rows(args.fixed_csv)
    auto_rows = load_rows(args.auto_csv)
    key_rows = load_rows(args.keymode_csv)
    fastrange_rows = load_rows(args.fastrange_csv)

    raw: dict[str, int | float] = {}
    display: dict[str, str] = {}

    def add(name: str, raw_value: int | float, display_value: str | None = None) -> None:
        raw[name] = raw_value
        if display_value is None:
            display_value = str(raw_value)
        display[name] = display_value

    fixed_cell = collapse_trials(fixed_rows, ["impl", "phase", "n", "gamma", "seed_index"])
    fixed_build = [row for row in fixed_cell if row["phase"] == "build"]
    fixed_lookup = [row for row in fixed_cell if row["phase"] == "lookup"]

    n_values = sorted({int(row["n"]) for row in fixed_cell})
    gamma_values = sorted({float(row["gamma"]) for row in fixed_cell})
    seed_values = sorted({int(row["seed_index"]) for row in fixed_cell})

    add("PaperSeedCount", len(seed_values))
    add("PaperSizeCount", len(n_values))
    add("PaperGammaCount", len(gamma_values))
    add("PaperNMin", min(n_values), fmt_int(min(n_values)))
    add("PaperNMax", max(n_values), fmt_int(max(n_values)))
    add("PaperGammaMin", min(gamma_values), fmt_float(min(gamma_values), 2))
    add("PaperGammaMax", max(gamma_values), fmt_float(max(gamma_values), 2))
    add("PaperFixedCellCount", len(seed_values) * len(n_values) * len(gamma_values))
    add("PaperCollapsedCellCount", len(n_values) * len(gamma_values))
    add("PaperLargeNThreshold", LARGE_N_THRESHOLD, fmt_int(LARGE_N_THRESHOLD))
    add("PaperLargeCellCount", sum(1 for n in n_values if n >= LARGE_N_THRESHOLD) * len(gamma_values))

    fixed_build_wins, fixed_build_ties = wins_by_group(fixed_build, ["n", "gamma", "seed_index"], "impl")
    fixed_lookup_wins, fixed_lookup_ties = wins_by_group(fixed_lookup, ["n", "gamma", "seed_index"], "impl")
    add("PaperFixedBuildBetterWins", fixed_build_wins["better-mphf"])
    add("PaperFixedBuildBoomWins", fixed_build_wins["boomphf-rust"])
    add("PaperFixedBuildTieCount", fixed_build_ties)
    add("PaperFixedLookupBetterWins", fixed_lookup_wins["better-mphf"])
    add("PaperFixedLookupBoomWins", fixed_lookup_wins["boomphf-rust"])
    add("PaperFixedLookupCppWins", fixed_lookup_wins["bbhash-cpp"])
    add("PaperFixedLookupTieCount", fixed_lookup_ties)

    fixed_seed_median = collapse_trials(fixed_cell, ["impl", "phase", "n", "gamma"])
    seed_median_build = [row for row in fixed_seed_median if row["phase"] == "build"]
    seed_median_lookup = [row for row in fixed_seed_median if row["phase"] == "lookup"]
    seed_median_build_wins, _ = wins_by_group(seed_median_build, ["n", "gamma"], "impl")
    seed_median_lookup_wins, _ = wins_by_group(seed_median_lookup, ["n", "gamma"], "impl")
    add("PaperMedianBuildBetterWins", seed_median_build_wins["better-mphf"])
    add("PaperMedianBuildBoomWins", seed_median_build_wins["boomphf-rust"])
    add("PaperMedianLookupBetterWins", seed_median_lookup_wins["better-mphf"])
    add("PaperMedianLookupBoomWins", seed_median_lookup_wins["boomphf-rust"])
    add("PaperMedianLookupCppWins", seed_median_lookup_wins["bbhash-cpp"])

    large_build = [row for row in seed_median_build if int(str(row["n"])) >= LARGE_N_THRESHOLD]
    large_lookup = [row for row in seed_median_lookup if int(str(row["n"])) >= LARGE_N_THRESHOLD]
    large_build_wins, _ = wins_by_group(large_build, ["n", "gamma"], "impl")
    large_lookup_wins, _ = wins_by_group(large_lookup, ["n", "gamma"], "impl")
    add("PaperLargeBuildBetterWins", large_build_wins["better-mphf"])
    add("PaperLargeLookupBetterWins", large_lookup_wins["better-mphf"])

    fixed_better = [row for row in fixed_rows if row["impl"] == "better-mphf"]
    fixed_better_cell = collapse_trials(fixed_better, ["phase", "n", "gamma", "seed_index"])
    auto_better = [row for row in auto_rows if row["impl"] == "better-mphf-auto"]
    auto_cell = collapse_trials(auto_better, ["phase", "n", "gamma", "seed_index"])
    fixed_best: dict[tuple[str, str, str], float] = {}
    for row in fixed_better_cell:
        key = (str(row["phase"]), str(row["n"]), str(row["seed_index"]))
        fixed_best[key] = min(fixed_best.get(key, float("inf")), float(row["ms"]))

    auto_build_deltas: list[float] = []
    auto_lookup_deltas: list[float] = []
    auto_build_better = auto_build_worse = 0
    auto_lookup_better = auto_lookup_worse = 0
    for row in auto_cell:
        key = (str(row["phase"]), str(row["n"]), str(row["seed_index"]))
        delta = (float(row["ms"]) - fixed_best[key]) / fixed_best[key] * 100.0
        if row["phase"] == "build":
            auto_build_deltas.append(delta)
            if delta < 0:
                auto_build_better += 1
            elif delta > 0:
                auto_build_worse += 1
        else:
            auto_lookup_deltas.append(delta)
            if delta < 0:
                auto_lookup_better += 1
            elif delta > 0:
                auto_lookup_worse += 1

    add("PaperAutoComparisonCellCount", len(auto_build_deltas))
    add("PaperAutoBuildBetterWins", auto_build_better)
    add("PaperAutoBuildWorseWins", auto_build_worse)
    add("PaperAutoBuildMedianSlowdownPct", statistics.median(auto_build_deltas), fmt_float(statistics.median(auto_build_deltas), 2))
    add("PaperAutoLookupBetterWins", auto_lookup_better)
    add("PaperAutoLookupWorseWins", auto_lookup_worse)
    add("PaperAutoLookupMedianAdvantagePct", abs(statistics.median(auto_lookup_deltas)), fmt_float(abs(statistics.median(auto_lookup_deltas)), 2))

    seed_cell = [row for row in fixed_better_cell if row["phase"] == "build"]
    seed_groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in seed_cell:
        key = (str(row["n"]), str(row["gamma"]))
        seed_groups[key].append(float(row["ms"]))

    ratios: list[float] = []
    for key, vals in seed_groups.items():
        ratios.append(max(vals) / min(vals))

    add("PaperSeedRatioMedianX", statistics.median(ratios), fmt_float(statistics.median(ratios), 2))
    add("PaperSeedRatioMaxX", max(ratios), fmt_float(max(ratios), 2))
    add("PaperSeedExampleN", 1_048_576, fmt_int(1_048_576))
    add("PaperSeedExampleGammaLow", 1.35, fmt_float(1.35, 2))
    add("PaperSeedExampleGammaHigh", 2.00, fmt_float(2.00, 2))
    low_vals = seed_groups[("1048576", "1.350000")]
    high_vals = seed_groups[("1048576", "2.000000")]
    add("PaperSeedLowBestMs", min(low_vals), fmt_float(min(low_vals), 1))
    add("PaperSeedLowMedianMs", statistics.median(low_vals), fmt_float(statistics.median(low_vals), 1))
    add("PaperSeedLowWorstMs", max(low_vals), fmt_float(max(low_vals), 1))
    add("PaperSeedHighBestMs", min(high_vals), fmt_float(min(high_vals), 1))
    add("PaperSeedHighMedianMs", statistics.median(high_vals), fmt_float(statistics.median(high_vals), 1))
    add("PaperSeedHighWorstMs", max(high_vals), fmt_float(max(high_vals), 1))

    fastrange_better = [row for row in fastrange_rows if row["impl"] == "better-mphf"]
    fastrange_cell = collapse_trials(fastrange_better, ["fastrange_mode", "phase", "n", "gamma", "seed_index"])
    fastrange_median = collapse_trials(fastrange_cell, ["fastrange_mode", "phase", "n", "gamma"])
    fr_build = [row for row in fastrange_median if row["phase"] == "build"]
    fr_lookup = [row for row in fastrange_median if row["phase"] == "lookup"]
    fr_build_wins, _ = wins_by_group(fr_build, ["n", "gamma"], "fastrange_mode")
    fr_lookup_wins, _ = wins_by_group(fr_lookup, ["n", "gamma"], "fastrange_mode")
    add("PaperFastRangeBuildMulSixtyFourWins", fr_build_wins["mul64"])
    add("PaperFastRangeBuildLowThirtyTwoWins", fr_build_wins["low32"])
    add("PaperFastRangeBuildHighThirtyTwoWins", fr_build_wins["high32"])
    add("PaperFastRangeLookupMulSixtyFourWins", fr_lookup_wins["mul64"])
    add("PaperFastRangeLookupLowThirtyTwoWins", fr_lookup_wins["low32"])
    add("PaperFastRangeLookupHighThirtyTwoWins", fr_lookup_wins["high32"])

    def mode_delta(rows: list[dict[str, object]], mode_field: str, baseline: str, mode: str, phase: str) -> float:
        base: dict[tuple[str, str], float] = {}
        alt: dict[tuple[str, str], float] = {}
        for row in rows:
            if row["phase"] != phase:
                continue
            key = (str(row["n"]), str(row["gamma"]))
            if row[mode_field] == baseline:
                base[key] = float(row["ms"])
            if row[mode_field] == mode:
                alt[key] = float(row["ms"])
        deltas = [(alt[key] - base[key]) / base[key] * 100.0 for key in base]
        return statistics.median(deltas)

    fr_mul64_build = mode_delta(fastrange_median, "fastrange_mode", "low32", "mul64", "build")
    fr_mul64_lookup = mode_delta(fastrange_median, "fastrange_mode", "low32", "mul64", "lookup")
    fr_high32_build = mode_delta(fastrange_median, "fastrange_mode", "low32", "high32", "build")
    fr_high32_lookup = mode_delta(fastrange_median, "fastrange_mode", "low32", "high32", "lookup")
    add("PaperFastRangeMulSixtyFourBuildImprovePct", abs(fr_mul64_build), fmt_float(abs(fr_mul64_build), 1))
    add("PaperFastRangeMulSixtyFourLookupImprovePct", abs(fr_mul64_lookup), fmt_float(abs(fr_mul64_lookup), 1))
    add("PaperFastRangeHighThirtyTwoBuildImprovePct", abs(fr_high32_build), fmt_float(abs(fr_high32_build), 1))
    add("PaperFastRangeHighThirtyTwoLookupPenaltyPct", abs(fr_high32_lookup), fmt_float(abs(fr_high32_lookup), 1))

    key_better = [row for row in key_rows if row["impl"] == "better-mphf"]
    key_cell = collapse_trials(key_better, ["key_mode", "phase", "n", "gamma", "seed_index"])
    key_median = collapse_trials(key_cell, ["key_mode", "phase", "n", "gamma"])

    key_deltas: dict[tuple[str, str], float] = {}
    key_modes = ["sequential", "clustered", "high-bit-heavy", "splitmix-random"]
    for mode in key_modes:
        for phase in ["build", "lookup"]:
            key_deltas[(mode, phase)] = mode_delta(key_median, "key_mode", "multiplicative", mode, phase)

    lookup_bound = max(abs(key_deltas[(mode, "lookup")]) for mode in key_modes)
    build_min = min(key_deltas[(mode, "build")] for mode in key_modes)
    build_max = max(key_deltas[(mode, "build")] for mode in key_modes)
    add("PaperKeyLookupBoundPct", lookup_bound, fmt_float(lookup_bound, 1))
    add("PaperKeyBuildShiftMinPct", build_min, fmt_float_signed(build_min, 1))
    add("PaperKeyBuildShiftMaxPct", build_max, fmt_float_signed(build_max, 1))

    generated_at = datetime.now(timezone.utc).isoformat()
    args.tex_out.parent.mkdir(parents=True, exist_ok=True)
    tex_lines = [
        "% Generated by scripts/generate_paper_numbers.py. Do not edit by hand.",
        f"% Generated at {generated_at}",
        f"% Sources: {args.fixed_csv.name}, {args.auto_csv.name}, {args.keymode_csv.name}, {args.fastrange_csv.name}",
    ]
    for name in sorted(display):
        tex_lines.append(f"\\newcommand{{\\{name}}}{{{display[name]}}}")
    args.tex_out.write_text("\n".join(tex_lines) + "\n")

    payload = {
        "generated_at_utc": generated_at,
        "sources": {
            "fixed_csv": str(args.fixed_csv),
            "auto_csv": str(args.auto_csv),
            "seed_sensitivity_csv": str(args.fixed_csv),
            "keymode_sweep_csv": str(args.keymode_csv),
            "fastrange_sweep_csv": str(args.fastrange_csv),
        },
        "display": display,
        "raw": raw,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"wrote {args.tex_out}")
    print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
