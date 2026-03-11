//! Benchmark different PAR_THRESHOLD values to find where parallelization pays off.
//!
//! Run with: cargo run --example bench_par_threshold --release --features parallel
//!
//! Without the `parallel` feature, every threshold behaves the same (serial only).

use better_mphf::LeveledMphf;
use std::time::Instant;

const SEED: u64 = 0xa0761d6478bd642f;
const OFFSET: u64 = 0xe7037ed1a0b428db;
const EXPANSION: f64 = 1.5;

/// Key counts to test (spanning below and above typical thresholds).
const KEY_COUNTS: &[usize] = &[
    512, 1024, 2_048, 4_096, 6_000, 8_192, 12_000, 16_384, 24_000, 32_768, 65_536, 131_072,
];

/// Thresholds to try: usize::MAX = effectively serial; lower = parallel kicks in earlier.
const THRESHOLDS: &[usize] = &[
    usize::MAX, // always serial
    16_384,
    8_192,
    4_096,
    2_048,
    1_024,
    512,
];

const WARMUP_RUNS: u32 = 1;
const TIMED_RUNS: u32 = 5;

fn main() {
    println!("PAR_THRESHOLD benchmark (lower threshold = parallel used at smaller key counts)\n");
    println!("Key count  | Serial (MAX) | 16k   | 8k    | 4k    | 2k    | 1k    | 512   | Best");
    println!("-----------|--------------|-------|-------|-------|-------|-------|-------|------");

    for &n in KEY_COUNTS {
        let keys: Vec<u64> = (0..n as u64)
            .map(|i| i.wrapping_mul(0x9e37_79b9_7f4a_7c15))
            .collect();

        let mut times = Vec::with_capacity(THRESHOLDS.len());
        for &threshold in THRESHOLDS {
            // Warmup
            for _ in 0..WARMUP_RUNS {
                let _ = LeveledMphf::new_with_par_threshold(&keys, SEED, OFFSET, Some(EXPANSION), threshold);
            }

            let start = Instant::now();
            for _ in 0..TIMED_RUNS {
                let _ = LeveledMphf::new_with_par_threshold(&keys, SEED, OFFSET, Some(EXPANSION), threshold);
            }
            let elapsed = start.elapsed() / TIMED_RUNS;
            times.push(elapsed.as_secs_f64());
        }

        let best_idx = times
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let best_label = if THRESHOLDS[best_idx] == usize::MAX {
            "serial"
        } else {
            match THRESHOLDS[best_idx] {
                16_384 => "16k",
                8_192 => "8k",
                4_096 => "4k",
                2_048 => "2k",
                1_024 => "1k",
                512 => "512",
                _ => "?",
            }
        };

        let row = format!(
            "{:>9}  | {:>10.4}ms | {:>5.2} | {:>5.2} | {:>5.2} | {:>5.2} | {:>5.2} | {:>5.2} | {}",
            n,
            times[0] * 1000.0,
            times[1] * 1000.0,
            times[2] * 1000.0,
            times[3] * 1000.0,
            times[4] * 1000.0,
            times[5] * 1000.0,
            times[6] * 1000.0,
            best_label,
        );
        println!("{}", row);
    }

    println!("\nInterpretation: when a threshold column beats 'Serial (MAX)', parallelization helps at that key count.");
}
