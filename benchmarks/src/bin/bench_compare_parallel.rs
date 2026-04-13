//! Parallel construction + lookup: **better-mphf** (Rayon) vs **boomphf** (`new_parallel`) vs **BBHash** (C++).
//!
//! Sweeps **γ**, **n**, and **thread count**; the same `threads` value is used for Rayon, boomphf, and bbhash.
//! Construction: **median ms** over repeated timed runs; `[min–max]` = spread.
//!
//! Serial-only comparison: **`bench_compare`**.
//!
//! Build C++ (from repo root):  
//! `g++ -O3 -pthread -std=c++14 -o benchmarks/cpp/bbhash_bench benchmarks/cpp/bbhash_bench.cpp`
//!
//! Run: `cargo run -p mphf-benchmarks --features parallel --bin bench_compare_parallel --release`

use mphf_benchmarks::{
    format_mmm_wide, format_timing_na, hostname_hint, lookup_better_parallel_ms,
    lookup_boom_parallel_ms, make_keys, print_compare_timing_header, resolve_bbhash_exe,
    run_bbhash_subprocess_mmm, time_better_parallel_mmm, time_boom_parallel_mmm, GAMMAS,
    GAMMA_LARGE_N, LOOKUP_TIMED, NS_GRID, NS_LARGE, THREAD_COUNTS, TIMED_GRID, TIMED_LARGE,
    WARMUP_GRID, WARMUP_LARGE,
};
use rayon::ThreadPoolBuilder;
use std::path::PathBuf;

fn run_rows(
    cpp_exe: &Option<PathBuf>,
    pool: &rayon::ThreadPool,
    threads: usize,
    gamma: f64,
    ns: &[usize],
) {
    for &n in ns {
        let keys = make_keys(n);
        let (wu, td) = mphf_benchmarks::warmup_timed_for_n(n);

        let (b_med, b_lo, b_hi) = time_better_parallel_mmm(pool, &keys, gamma, wu, td);
        let (o_med, o_lo, o_hi) = time_boom_parallel_mmm(pool, &keys, gamma, wu, td);

        let cpp_cell = cpp_exe.as_ref().and_then(|p| {
            run_bbhash_subprocess_mmm(p, n, threads, gamma, wu, td)
                .map(|(m, lo, hi)| format_mmm_wide(m, lo, hi))
        });
        let cpp_cell = cpp_cell.unwrap_or_else(format_timing_na);

        let lk_b = lookup_better_parallel_ms(pool, &keys, gamma);
        let lk_o = lookup_boom_parallel_ms(pool, &keys, gamma);

        println!(
            "{n:>10} | {} | {} | {} | {lk_b:>8.2} | {lk_o:>8.2}",
            format_mmm_wide(b_med, b_lo, b_hi),
            format_mmm_wide(o_med, o_lo, o_hi),
            cpp_cell,
        );
    }
}

fn main() {
    let avail = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .max(1);

    let cpp_bbhash = resolve_bbhash_exe();

    println!(
        "Parallel benchmark - host `{}`, logical CPUs (hint): {}\n\
Construction: median ms; `[min–max]` across timed runs.\n\
Warmup / timed: {} / {} for n < 1M; {} / {} for n >= 1M (unless MPHF_BENCH_WARMUP/TIMED override).\n\
Lookup: full key scan (ms), mean over {} runs; one parallel build per lookup column (same pool).\n",
        hostname_hint(),
        avail,
        WARMUP_GRID,
        TIMED_GRID,
        WARMUP_LARGE,
        TIMED_LARGE,
        LOOKUP_TIMED,
    );

    for &threads in THREAD_COUNTS {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("rayon thread pool");

        for &gamma in GAMMAS {
            println!(
                "=== γ = {gamma:.3} | parallel threads = {threads} (Rayon + boomphf + bbhash) ==="
            );
            println!(
                "    (warmup/timed: {} / {} for n < 1M, else {} / {})",
                WARMUP_GRID, TIMED_GRID, WARMUP_LARGE, TIMED_LARGE
            );
            print_compare_timing_header("bmf-par", "boom-par");
            run_rows(&cpp_bbhash, &pool, threads, gamma, NS_GRID);
            println!();
        }
    }

    let threads_large = avail.min(32).max(1);
    let pool_large = ThreadPoolBuilder::new()
        .num_threads(threads_large)
        .build()
        .expect("rayon thread pool");

    println!(
        "=== Large n | γ = {GAMMA_LARGE_N} | parallel threads = {threads_large} ===\n\
            (warmup/timed: {WARMUP_LARGE} / {TIMED_LARGE})"
    );
    print_compare_timing_header("bmf-par", "boom-par");
    run_rows(
        &cpp_bbhash,
        &pool_large,
        threads_large,
        GAMMA_LARGE_N,
        NS_LARGE,
    );

    if cpp_bbhash.is_none() {
        eprintln!(
            "\nNote: bbhash(C++) is n/a without `benchmarks/cpp/bbhash_bench`. Build from repo root:\n  \
             g++ -O3 -pthread -std=c++14 -o benchmarks/cpp/bbhash_bench benchmarks/cpp/bbhash_bench.cpp"
        );
    }
}
