//! Benchmark different expansion factor (γ) values and auto-tuned construction.
//!
//! Run with: cargo run --example bench_gamma --release
//!
//! Compares fixed γ (1.3, 1.5, 1.7, 2.0) vs auto-tuned (adaptive_gamma: γ = gamma_base * (2 - n_remaining/n_original)).

use better_mphf::LeveledMphf;
use std::time::Instant;

const SEED: u64 = 0xa0761d6478bd642f;
const OFFSET: u64 = 0xe7037ed1a0b428db;

/// Key counts to test.
const KEY_COUNTS: &[usize] = &[
    1_000, 4_096, 8_192, 16_384, 32_768, 65_536, 131_072, 262_144,
];

/// Fixed expansion factors to compare.
const GAMMAS: &[f64] = &[1.3, 1.5, 1.7, 2.0];

const WARMUP_RUNS: u32 = 1;
const TIMED_RUNS: u32 = 5;

fn main() {
    println!("Expansion factor (γ) benchmark\n");
    println!(
        "{:>9} | {:>8} | {:>8} | {:>8} | {:>8} | {:>10} | Best",
        "Keys", "γ=1.3", "γ=1.5", "γ=1.7", "γ=2.0", "auto-tuned"
    );
    println!("{}", "-".repeat(75));

    for &n in KEY_COUNTS {
        let keys: Vec<u64> = (0..n as u64)
            .map(|i| i.wrapping_mul(0x9e37_79b9_7f4a_7c15))
            .collect();

        let mut times = Vec::with_capacity(GAMMAS.len() + 1);

        for &gamma in GAMMAS {
            for _ in 0..WARMUP_RUNS {
                let _ = LeveledMphf::new(&keys, SEED, OFFSET, gamma);
            }
            let start = Instant::now();
            for _ in 0..TIMED_RUNS {
                let _ = LeveledMphf::new(&keys, SEED, OFFSET, gamma);
            }
            times.push((start.elapsed() / TIMED_RUNS).as_secs_f64());
        }

        {
            for _ in 0..WARMUP_RUNS {
                let _ = LeveledMphf::new_auto_tuned(&keys, SEED, OFFSET);
            }
            let start = Instant::now();
            for _ in 0..TIMED_RUNS {
                let _ = LeveledMphf::new_auto_tuned(&keys, SEED, OFFSET);
            }
            times.push((start.elapsed() / TIMED_RUNS).as_secs_f64());
        }

        let best_idx = times
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let best_label = if best_idx < GAMMAS.len() {
            format!("γ={}", GAMMAS[best_idx])
        } else {
            "auto-tuned".to_string()
        };

        println!(
            "{:>9} | {:>6.2}ms | {:>6.2}ms | {:>6.2}ms | {:>6.2}ms | {:>8.2}ms | {}",
            n,
            times[0] * 1000.0,
            times[1] * 1000.0,
            times[2] * 1000.0,
            times[3] * 1000.0,
            times[4] * 1000.0,
            best_label,
        );
    }

    println!("\nLower γ = smaller tables but more retries; auto-tuned uses adaptive_gamma so γ grows as keys drain.");
}
