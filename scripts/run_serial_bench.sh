#!/usr/bin/env bash
#
# Serial benchmark
#
# Run under Slurm (e.g. from comps[0-3].cs):
#
#   cd /path/to/better-leveledmphf
#   srun --partition cpunodes -c 8 --mem=16G -t 0-4:00 --pty ./scripts/run_serial_bench.sh
#
# Text output: bench-results/serial-<jobid>.txt (override dir: export MPHF_BENCH_RESULTS_DIR=/path/to/dir).
#
# Optional env (export before srun):
#   MPHF_BENCH_WARMUP, MPHF_BENCH_TIMED — override warmup/timed counts (benchmarks/src/lib.rs)
#   CXX — C++ compiler (default: g++)

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

if [[ -d "${HOME}/.cargo/bin" ]]; then
  export PATH="${HOME}/.cargo/bin:${PATH}"
fi
if [[ -f "${HOME}/.cargo/env" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/.cargo/env"
fi

echo "==> $(date -Is)  repo=${REPO_ROOT}"
echo "==> host=$(hostname)  SLURM_JOB_ID=${SLURM_JOB_ID:-}  SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-}"
command -v cargo >/dev/null
command -v rustc >/dev/null
rustc --version
cargo --version
command -v "${CXX:-g++}" >/dev/null
"${CXX:-g++}" --version | head -1

echo "==> Building Rust: mphf-benchmarks (release)"
cargo build -p mphf-benchmarks --release

echo "==> Building C++: benchmarks/cpp/bbhash_bench"
mkdir -p benchmarks/cpp
"${CXX:-g++}" -O3 -pthread -std=c++14 \
  -o benchmarks/cpp/bbhash_bench \
  benchmarks/cpp/bbhash_bench.cpp

RESULTS_DIR="${MPHF_BENCH_RESULTS_DIR:-${REPO_ROOT}/bench-results}"
mkdir -p "$RESULTS_DIR"
OUT="${RESULTS_DIR}/serial-${SLURM_JOB_ID:-local}.txt"
echo "==> Running bench_compare (tee -> ${OUT})"
cargo run -p mphf-benchmarks --bin bench_compare --release 2>&1 | tee "$OUT"

echo "==> $(date -Is)  finished; main table copy: ${OUT}"
