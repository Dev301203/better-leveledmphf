//! Benchmark different PAR_THRESHOLD values to find where parallelization pays off.
//!
//! Run with: `cargo run -p mphf-benchmarks --bin bench_par_threshold --release`
//!
//! Without the `parallel` feature, every threshold behaves the same (serial only).

use better_mphf::LeveledMphf;
use std::time::Instant;

const SEED: u64 = 0xa0761d6478bd642f;
const OFFSET: u64 = 0xe7037ed1a0b428db;
const EXPANSION: f64 = 1.5;

// Key counts to test: small (threshold crossover) and large (parallel clearly wins).
const KEY_COUNTS: &[usize] = &[
    512, 1024, 2_048, 4_096, 6_000, 8_192, 12_000, 16_384, 24_000, 32_768, 65_536, 131_072,
    500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000, 25_000_000, 62_500_000,
];

// Thresholds to try: usize::MAX = serial; then ascending so columns line up left-to-right.
const THRESHOLDS: &[usize] = &[
    usize::MAX,   // serial
    512,
    1_024,
    2_048,
    4_096,
    8_192,
    16_384,
    32_768,
    65_536,
    131_072,
    262_144,
    524_288,
    1_000_000,
    2_000_000,
    3_000_000,
    4_000_000,
    5_000_000,
];

const WARMUP_RUNS: u32 = 1;
const TIMED_RUNS: u32 = 5;

const COL_WIDTH: usize = 8; // width for each threshold column (header + data)

fn threshold_label(t: usize) -> String {
    if t == usize::MAX {
        "MAX".to_string()
    } else if t >= 1_000_000 {
        format!("{}M", t / 1_000_000)
    } else if t >= 1_024 {
        format!("{}k", t / 1_024)
    } else {
        t.to_string()
    }
}

fn main() {
    println!("PAR_THRESHOLD benchmark (lower threshold = parallel used at smaller key counts)\n");
    let header_cols: String = THRESHOLDS
        .iter()
        .map(|&t| format!(" {:>width$} |", threshold_label(t), width = COL_WIDTH - 2))
        .collect();
    println!("Key count  |{} Best", header_cols);
    let sep_len = 12 + header_cols.len() + 5;
    println!("{}", "-".repeat(sep_len));

    for &n in KEY_COUNTS {
        let keys: Vec<u64> = (0..n as u64)
            .map(|i| i.wrapping_mul(0x9e37_79b9_7f4a_7c15))
            .collect();

        // Only run thresholds where threshold <= n (or MAX for serial); larger thresholds are serial anyway.
        let indices_to_run: Vec<usize> = THRESHOLDS
            .iter()
            .enumerate()
            .filter(|(_, t)| **t == usize::MAX || **t <= n)
            .map(|(i, _)| i)
            .collect();

        let mut times = vec![f64::NAN; THRESHOLDS.len()];
        for &i in &indices_to_run {
            let threshold = THRESHOLDS[i];
            for _ in 0..WARMUP_RUNS {
                let _ = LeveledMphf::new_with_par_threshold(&keys, SEED, OFFSET, Some(EXPANSION), threshold);
            }
            let start = Instant::now();
            for _ in 0..TIMED_RUNS {
                let _ = LeveledMphf::new_with_par_threshold(&keys, SEED, OFFSET, Some(EXPANSION), threshold);
            }
            times[i] = (start.elapsed() / TIMED_RUNS).as_secs_f64();
        }

        let serial_time = times[0];
        let best_parallel_idx = times
            .iter()
            .enumerate()
            .filter(|(i, t)| *i > 0 && n >= THRESHOLDS[*i] && t.is_finite())
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i);
        let best_label = match best_parallel_idx {
            Some(i) if times[i] < serial_time => format!("parallel ({})", threshold_label(THRESHOLDS[i])),
            _ => "serial".to_string(),
        };

        let times_str: String = times
            .iter()
            .map(|t| {
                if t.is_finite() {
                    format!(" {:>width$.2} |", t * 1000.0, width = COL_WIDTH - 2)
                } else {
                    format!(" {:>width$} |", "-", width = COL_WIDTH - 2)
                }
            })
            .collect();
        println!("{:>9}  |{} {}", n, times_str, &best_label);
    }

    println!("\nInterpretation: Best = 'parallel' when the best parallel run beats serial; else 'serial'.");
}
