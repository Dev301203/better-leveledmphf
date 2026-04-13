//! Serial construction + lookup comparison: **better-mphf** vs **boomphf** vs optional **BBHash** (C++).
//!
//! Construction: **median ms** over repeated timed runs; bracket shows **min–max** spread.
//! Use on a quiet / dedicated node (e.g. Slurm exclusive) for conclusive numbers.
//!
//! Parallel construction lives in **`bench_compare_parallel`** (`--features parallel`).
//!
//! Build C++ (from repo root):  
//! `g++ -O3 -pthread -std=c++14 -o benchmarks/cpp/bbhash_bench benchmarks/cpp/bbhash_bench.cpp`
//!
//! Run: `cargo run -p mphf-benchmarks --bin bench_compare --release`

use mphf_benchmarks::{
    format_mmm_wide, format_timing_na, hostname_hint, lookup_better_serial_ms, lookup_boom_ms,
    make_keys, print_compare_timing_header, resolve_bbhash_exe, run_bbhash_subprocess_mmm,
    time_better_serial_mmm, time_boom_serial_mmm, GAMMAS, GAMMA_LARGE_N, LOOKUP_TIMED, NS_GRID,
    NS_LARGE, TIMED_GRID, TIMED_LARGE, WARMUP_GRID, WARMUP_LARGE,
};
use std::path::PathBuf;

fn main() {
    let avail = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .max(1);
    // Serial bench: C++ BooPHF uses one thread, same as bmf-ser / boom-ser construction.
    const BBHASH_THREADS_SERIAL: usize = 1;

    let cpp_bbhash: Option<PathBuf> = resolve_bbhash_exe();

    println!(
        "Serial benchmark - host `{}`, logical CPUs (hint): {}\n\
Construction: median ms over timed runs; `[min–max]` = spread across those runs.\n\
Warmup / timed: {} / {} for n < 1M; {} / {} for n >= 1M (unless MPHF_BENCH_WARMUP/TIMED override).\n\
bbhash(C++): {} pthread worker(s); serial BooPHF build (matches Rust construction columns).\n\
Lookup: single-threaded full key scan (ms), mean over {} runs; one serial build per lookup column.\n",
        hostname_hint(),
        avail,
        WARMUP_GRID,
        TIMED_GRID,
        WARMUP_LARGE,
        TIMED_LARGE,
        BBHASH_THREADS_SERIAL,
        LOOKUP_TIMED,
    );

    for &gamma in GAMMAS {
        println!("=== γ = {gamma:.3} | serial construction ===");
        println!(
            "    (warmup/timed: {} / {} for n < 1M, else {} / {})",
            WARMUP_GRID, TIMED_GRID, WARMUP_LARGE, TIMED_LARGE
        );
        print_compare_timing_header("bmf-ser", "boom-ser");
        for &n in NS_GRID {
            let keys = make_keys(n);
            let (wu, td) = mphf_benchmarks::warmup_timed_for_n(n);

            let (b_med, b_lo, b_hi) = time_better_serial_mmm(&keys, gamma, wu, td);
            let (o_med, o_lo, o_hi) = time_boom_serial_mmm(&keys, gamma, wu, td);

            let cpp_cell = cpp_bbhash.as_ref().and_then(|p| {
                run_bbhash_subprocess_mmm(p, n, BBHASH_THREADS_SERIAL, gamma, wu, td)
                    .map(|(m, lo, hi)| format_mmm_wide(m, lo, hi))
            });

            let cpp_cell = cpp_cell.unwrap_or_else(format_timing_na);

            let lk_b = lookup_better_serial_ms(&keys, gamma);
            let lk_o = lookup_boom_ms(&keys, gamma);

            println!(
                "{n:>10} | {} | {} | {} | {lk_b:>8.2} | {lk_o:>8.2}",
                format_mmm_wide(b_med, b_lo, b_hi),
                format_mmm_wide(o_med, o_lo, o_hi),
                cpp_cell,
            );
        }
        println!();
    }

    println!(
        "=== Large n | γ = {GAMMA_LARGE_N} | serial construction ===\n\
(warmup/timed: {WARMUP_LARGE} / {TIMED_LARGE}; bbhash threads = {BBHASH_THREADS_SERIAL})"
    );
    print_compare_timing_header("bmf-ser", "boom-ser");
    for &n in NS_LARGE {
        let keys = make_keys(n);
        let (wu, td) = mphf_benchmarks::warmup_timed_for_n(n);

        let (b_med, b_lo, b_hi) = time_better_serial_mmm(&keys, GAMMA_LARGE_N, wu, td);
        let (o_med, o_lo, o_hi) = time_boom_serial_mmm(&keys, GAMMA_LARGE_N, wu, td);

        let cpp_cell = cpp_bbhash.as_ref().and_then(|p| {
            run_bbhash_subprocess_mmm(p, n, BBHASH_THREADS_SERIAL, GAMMA_LARGE_N, wu, td)
                .map(|(m, lo, hi)| format_mmm_wide(m, lo, hi))
        });
        let cpp_cell = cpp_cell.unwrap_or_else(format_timing_na);

        let lk_b = lookup_better_serial_ms(&keys, GAMMA_LARGE_N);
        let lk_o = lookup_boom_ms(&keys, GAMMA_LARGE_N);

        println!(
            "{n:>10} | {} | {} | {} | {lk_b:>8.2} | {lk_o:>8.2}",
            format_mmm_wide(b_med, b_lo, b_hi),
            format_mmm_wide(o_med, o_lo, o_hi),
            cpp_cell,
        );
    }

    if cpp_bbhash.is_none() {
        eprintln!(
            "\nNote: bbhash(C++) is n/a without `benchmarks/cpp/bbhash_bench`. Build from repo root:\n  \
             g++ -O3 -pthread -std=c++14 -o benchmarks/cpp/bbhash_bench benchmarks/cpp/bbhash_bench.cpp"
        );
    }
}
