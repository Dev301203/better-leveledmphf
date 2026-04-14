//! Shared helpers for the standalone objective benchmark runners.

pub const SEED: u64 = 0xa0761d6478bd642f;
pub const OFFSET: u64 = 0xe7037ed1a0b428db;

pub fn make_keys(n: usize) -> Vec<u64> {
    (0..n as u64)
        .map(|i| i.wrapping_mul(0x9e37_79b9_7f4a_7c15))
        .collect()
}

pub fn median_min_max(samples: &mut [f64]) -> (f64, f64, f64) {
    if samples.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = samples.len();
    let min = samples[0];
    let max = samples[n - 1];
    let med = if n % 2 == 1 {
        samples[n / 2]
    } else {
        (samples[n / 2 - 1] + samples[n / 2]) / 2.0
    };
    (med, min, max)
}

pub fn mean(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        0.0
    } else {
        samples.iter().sum::<f64>() / samples.len() as f64
    }
}

pub fn sample_after_warmup_raw(warmup: u32, timed: u32, mut one_build: impl FnMut()) -> Vec<f64> {
    use std::time::Instant;

    for _ in 0..warmup {
        one_build();
    }
    let mut samples = Vec::with_capacity(timed as usize);
    for _ in 0..timed {
        let t0 = Instant::now();
        one_build();
        samples.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    samples
}
