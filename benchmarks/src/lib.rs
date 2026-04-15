//! Shared helpers for the standalone objective benchmark runners.

pub const SEED: u64 = 0xa0761d6478bd642f;
pub const OFFSET: u64 = 0xe7037ed1a0b428db;
const GOLDEN: u64 = 0x9e37_79b9_7f4a_7c15;
const CLUSTER_BITS: u64 = 6;
const CLUSTER_SIZE: u64 = 1 << CLUSTER_BITS;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyMode {
    Multiplicative,
    Sequential,
    SplitmixRandom,
    Clustered,
    HighBitHeavy,
}

impl KeyMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Multiplicative => "multiplicative",
            Self::Sequential => "sequential",
            Self::SplitmixRandom => "splitmix-random",
            Self::Clustered => "clustered",
            Self::HighBitHeavy => "high-bit-heavy",
        }
    }
}

impl std::str::FromStr for KeyMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "multiplicative" | "mult" => Ok(Self::Multiplicative),
            "sequential" | "seq" => Ok(Self::Sequential),
            "splitmix-random" | "splitmix_random" | "splitmix" | "random" => {
                Ok(Self::SplitmixRandom)
            }
            "clustered" | "cluster" => Ok(Self::Clustered),
            "high-bit-heavy" | "high_bit_heavy" | "highbits" => Ok(Self::HighBitHeavy),
            _ => Err(format!(
                "unknown key mode '{s}' (expected multiplicative|sequential|splitmix-random|clustered|high-bit-heavy)"
            )),
        }
    }
}

#[inline(always)]
fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(GOLDEN);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

pub fn seed_offset_from_index(seed_index: u64) -> (u64, u64) {
    let seed_mix = seed_index.wrapping_mul(GOLDEN);
    let seed = splitmix64(SEED ^ seed_mix);
    let offset = splitmix64(OFFSET ^ seed_mix.rotate_left(17));
    (seed, offset)
}

pub fn make_keys_with_mode(n: usize, key_mode: KeyMode, seed: u64, offset: u64) -> Vec<u64> {
    let mut keys = Vec::with_capacity(n);
    let offset_odd = offset | 1;

    for i in 0..n as u64 {
        let key = match key_mode {
            KeyMode::Multiplicative => i.wrapping_mul(GOLDEN).wrapping_add(offset),
            KeyMode::Sequential => i.wrapping_add(offset),
            KeyMode::SplitmixRandom => splitmix64(seed.wrapping_add(i.wrapping_mul(offset_odd))),
            KeyMode::Clustered => {
                let cluster = i >> CLUSTER_BITS;
                let intra = i & (CLUSTER_SIZE - 1);
                let perm_cluster = cluster.wrapping_mul(GOLDEN).wrapping_add(seed ^ offset);
                (perm_cluster << CLUSTER_BITS).wrapping_add(intra)
            }
            KeyMode::HighBitHeavy => {
                let x = i.wrapping_add(offset);
                let hi = x.rotate_left(32);
                let lo = splitmix64(seed ^ x) & 0xff;
                (hi & !0xff) | lo
            }
        };
        keys.push(key);
    }

    keys
}

pub fn make_keys(n: usize) -> Vec<u64> {
    make_keys_with_mode(n, KeyMode::Multiplicative, SEED, OFFSET)
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
