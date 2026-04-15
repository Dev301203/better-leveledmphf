#!/usr/bin/env bash
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

WARMUP=${MPHF_BENCH_WARMUP:-2}
TIMED=${MPHF_BENCH_TIMED:-7}
LOOKUP_TIMED=${MPHF_BENCH_LOOKUP_TIMED:-5}

RUN_FIXED=${MPHF_BENCH_RUN_FIXED:-1}
RUN_AUTO=${MPHF_BENCH_RUN_AUTO:-1}
RUN_BASELINES=${MPHF_BENCH_RUN_BASELINES:-1}

RESULTS_DIR="${MPHF_BENCH_RESULTS_DIR:-${REPO_ROOT}/bench-results}"
mkdir -p "$RESULTS_DIR"
RUN_NAME="${MPHF_BENCH_RUN_NAME:-serial-bench}"
OUT="${RESULTS_DIR}/${RUN_NAME}.csv"
ERR="${RESULTS_DIR}/${RUN_NAME}.log"

DEFAULT_NS="512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152"
DEFAULT_GAMMAS="1.15 1.25 1.35 1.5 1.65 1.8 2.0 2.25 2.5"

read -r -a NS <<< "${MPHF_BENCH_NS:-$DEFAULT_NS}"
read -r -a GAMMAS <<< "${MPHF_BENCH_GAMMAS:-$DEFAULT_GAMMAS}"
read -r -a SEED_INDICES <<< "${MPHF_BENCH_SEED_INDICES:-0}"
read -r -a KEY_MODES <<< "${MPHF_BENCH_KEY_MODES:-multiplicative}"
read -r -a AUTO_GAMMA_BASES <<< "${MPHF_BENCH_AUTO_GAMMA_BASES:-${MPHF_BENCH_AUTO_GAMMA_BASE:-1.5}}"
read -r -a FASTRANGE_MODES <<< "${MPHF_BENCH_FASTRANGE_MODES:-mul64}"

: > "$OUT"
: > "$ERR"

fixed_better_cells=$(( ${#SEED_INDICES[@]} * ${#KEY_MODES[@]} * ${#GAMMAS[@]} * ${#NS[@]} * ${#FASTRANGE_MODES[@]} ))
auto_better_cells=$(( ${#SEED_INDICES[@]} * ${#KEY_MODES[@]} * ${#NS[@]} * ${#AUTO_GAMMA_BASES[@]} * ${#FASTRANGE_MODES[@]} ))
baseline_cells=$(( ${#SEED_INDICES[@]} * ${#KEY_MODES[@]} * ${#GAMMAS[@]} * ${#NS[@]} ))

echo "==> $(date -Is) repo=${REPO_ROOT}" | tee -a "$ERR"
echo "==> host=$(hostname) warmup=${WARMUP} timed=${TIMED} lookup_timed=${LOOKUP_TIMED}" | tee -a "$ERR"
echo "==> run_fixed=${RUN_FIXED} run_auto=${RUN_AUTO} run_baselines=${RUN_BASELINES}" | tee -a "$ERR"
echo "==> seed_indices=${SEED_INDICES[*]} key_modes=${KEY_MODES[*]}" | tee -a "$ERR"
echo "==> fixed_gammas=${GAMMAS[*]} auto_gamma_bases=${AUTO_GAMMA_BASES[*]} auto_shape=fixed-piecewise" | tee -a "$ERR"
echo "==> fastrange_modes=${FASTRANGE_MODES[*]}" | tee -a "$ERR"
echo "==> estimated_cells better_fixed=${fixed_better_cells} better_auto=${auto_better_cells} baselines=${baseline_cells}" | tee -a "$ERR"
echo "==> fairness note: boomphf-rust and bbhash-cpp expose fixed gamma only; better-mphf auto-gamma is emitted as a separate extra series" | tee -a "$ERR"

echo "==> Building Rust runners" | tee -a "$ERR"
cargo build -p mphf-benchmarks --release --bin bench_better_runner --bin bench_boom_runner 2>&1 | tee -a "$ERR"

BETTER_BIN="${REPO_ROOT}/target/release/bench_better_runner"
BOOM_BIN="${REPO_ROOT}/target/release/bench_boom_runner"

echo "==> Building C++ runner" | tee -a "$ERR"
"${CXX:-g++}" -O3 -pthread -std=c++14 \
  -o benchmarks/cpp/bbhash_runner \
  benchmarks/cpp/bbhash_runner.cpp 2>&1 | tee -a "$ERR"

emit_runner_csv() {
  local runner=$1
  shift
  local tmp
  tmp=$(mktemp)
  "$runner" "$@" > "$tmp" 2>> "$ERR"
  if [[ ! -s "$OUT" ]]; then
    cat "$tmp" >> "$OUT"
  else
    tail -n +2 "$tmp" >> "$OUT"
  fi
  rm -f "$tmp"
}

for seed_index in "${SEED_INDICES[@]}"; do
  for key_mode in "${KEY_MODES[@]}"; do
    if [[ "$RUN_FIXED" == "1" ]]; then
      for gamma in "${GAMMAS[@]}"; do
        for n in "${NS[@]}"; do
          echo "==> fixed gamma=${gamma} n=${n} seed_index=${seed_index} key_mode=${key_mode}" | tee -a "$ERR"

          if [[ "$RUN_BASELINES" == "1" ]]; then
            emit_runner_csv "$BOOM_BIN" \
              --n "$n" --gamma "$gamma" --seed-index "$seed_index" --key-mode "$key_mode" \
              --warmup "$WARMUP" --timed "$TIMED" --lookup-timed "$LOOKUP_TIMED"

            emit_runner_csv ./benchmarks/cpp/bbhash_runner \
              --n "$n" --gamma "$gamma" --seed-index "$seed_index" --key-mode "$key_mode" \
              --warmup "$WARMUP" --timed "$TIMED" --lookup-timed "$LOOKUP_TIMED" --threads 1
          fi

          for fastrange_mode in "${FASTRANGE_MODES[@]}"; do
            emit_runner_csv "$BETTER_BIN" \
              --n "$n" --gamma "$gamma" --seed-index "$seed_index" --key-mode "$key_mode" \
              --fastrange-mode "$fastrange_mode" \
              --warmup "$WARMUP" --timed "$TIMED" --lookup-timed "$LOOKUP_TIMED"
          done
        done
      done
    fi

    if [[ "$RUN_AUTO" == "1" ]]; then
      for n in "${NS[@]}"; do
        for gamma_base in "${AUTO_GAMMA_BASES[@]}"; do
          echo "==> auto n=${n} gamma_base=${gamma_base} auto_shape=piecewise seed_index=${seed_index} key_mode=${key_mode}" | tee -a "$ERR"
          for fastrange_mode in "${FASTRANGE_MODES[@]}"; do
            emit_runner_csv "$BETTER_BIN" \
              --n "$n" --auto-gamma --gamma-base "$gamma_base" \
              --seed-index "$seed_index" --key-mode "$key_mode" \
              --fastrange-mode "$fastrange_mode" \
              --warmup "$WARMUP" --timed "$TIMED" --lookup-timed "$LOOKUP_TIMED"
          done
        done
      done
    fi
  done
done

echo "==> finished csv=${OUT} log=${ERR}" | tee -a "$ERR"
