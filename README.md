# Better Leveled MPHF

Leveled minimal perfect hash function in Rust. Hashes `n` keys to `[0, n)` with no collisions.

Keys are hashed into a bitset level by level. Keys that land in a unique slot are placed there, while the rest cascade to the next level. Lookup checks each level until it finds the key's slot and returns the cumulative rank.

## Implementation Details

- Fast range reduction instead of modulo
- Rank blocks are cache-aligned; bitsets are oversized to the nearest cache line
- Fused `rank_if_set`, single cache-line load does the get and popcount together, bails early on miss
- Level 0 peeled out since cumulative rank is always 0
- Poisson acceptance, construction rejects a hash function if unique-landing count falls below `n·e^(−load)`
- SplitMix64 hash, giga fast and random enough
- `new*` is the fast path and assumes the input is a non-empty set of unique `u64` keys
- `try_new*` validates empty or duplicate input and returns a `BuildError`
- Lookup is only defined for keys in the original set; a missing key causes a panic

## Usage

```rust
let keys: Vec<u64> = (0..1_000_000).collect();
let mphf = LeveledMphf::new(&keys, seed, offset, 1.5);
let idx = mphf.lookup(42); // in [0, n)
```

## Paper

The paper is in `report/paper.pdf`

The figures used by the paper are in `report/figures`

The benchmark/result numbers used by the paper are generated into:

- `report/generated_paper_numbers.tex`
- `report/generated_paper_numbers.json`

Generate them with:

```bash
python3 scripts/generate_paper_numbers.py
```

## Generate The Plots

Requirements:

- `python3`
- Python packages: `matplotlib`, `numpy`

Install plotting dependencies if needed:

```bash
python3 -m pip install matplotlib numpy
```

Generate the plot set used by the paper:

```bash
python3 scripts/plot_paper_figures.py
```

## Plot Inputs Used By The Current Paper

The current paper figures are generated from these checked-in CSVs:

- `bench-results/serial-fixed.csv`
- `bench-results/serial-auto.csv`
- `bench-results/serial-keymode.csv`
- `bench-results/serial-fastrange.csv`

`serial-fixed.csv` is used both for the fixed-gamma comparison and for the paper's seed-sensitivity analysis.

`plot_paper_figures.py` merges the fixed and auto 16-seed benchmark CSVs into:

- `bench-results/serial-merged.csv`

## Reproduce The Benchmark CSVs From Scratch

The benchmark driver is:

- `scripts/run_serial_bench.sh`

### 16-Seed Fixed-Gamma Benchmark

```bash
srun --partition cpunodes -c 12 --mem=24G -t 0-4:00 --pty env \
  MPHF_BENCH_RUN_NAME=serial-fixed \
  MPHF_BENCH_RUN_FIXED=1 \
  MPHF_BENCH_RUN_BASELINES=1 \
  MPHF_BENCH_RUN_AUTO=0 \
  MPHF_BENCH_SEED_INDICES="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15" \
  MPHF_BENCH_KEY_MODES="multiplicative" \
  MPHF_BENCH_FASTRANGE_MODES="mul64" \
  ./scripts/run_serial_bench.sh
```

### 16-Seed Auto-Gamma Benchmark

```bash
srun --partition cpunodes -c 12 --mem=24G -t 0-4:00 --pty env \
  MPHF_BENCH_RUN_NAME=serial-auto \
  MPHF_BENCH_RUN_FIXED=0 \
  MPHF_BENCH_RUN_BASELINES=0 \
  MPHF_BENCH_RUN_AUTO=1 \
  MPHF_BENCH_SEED_INDICES="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15" \
  MPHF_BENCH_KEY_MODES="multiplicative" \
  MPHF_BENCH_FASTRANGE_MODES="mul64" \
  MPHF_BENCH_AUTO_GAMMA_BASES="1.5" \
  ./scripts/run_serial_bench.sh
```

### Key-Distribution Comparison

```bash
srun --partition cpunodes -c 12 --mem=24G -t 0-4:00 --pty env \
  MPHF_BENCH_RUN_NAME=serial-keymode \
  MPHF_BENCH_RUN_FIXED=1 \
  MPHF_BENCH_RUN_BASELINES=1 \
  MPHF_BENCH_RUN_AUTO=0 \
  MPHF_BENCH_SEED_INDICES="0" \
  MPHF_BENCH_KEY_MODES="multiplicative sequential splitmix-random clustered high-bit-heavy" \
  MPHF_BENCH_FASTRANGE_MODES="mul64" \
  ./scripts/run_serial_bench.sh
```

### Fast-Range Comparison

```bash
srun --partition cpunodes -c 12 --mem=24G -t 0-4:00 --pty env \
  MPHF_BENCH_RUN_NAME=serial-fastrange \
  MPHF_BENCH_RUN_FIXED=1 \
  MPHF_BENCH_RUN_BASELINES=1 \
  MPHF_BENCH_RUN_AUTO=0 \
  MPHF_BENCH_SEED_INDICES="0" \
  MPHF_BENCH_KEY_MODES="multiplicative" \
  MPHF_BENCH_FASTRANGE_MODES="low32 high32 mul64" \
  ./scripts/run_serial_bench.sh
```

