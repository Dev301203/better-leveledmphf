use core::panic;

use crate::hash::SplitMix64;
use crate::bitset::{RankedBitSet, BitSet};
use crate::bitset::CACHE_LINE_BITS;

struct LevelMeta {
    offset_multiplier: u64,
    num_slots: usize,
    cumulative_rank: usize,
    keys_placed: usize,
}

pub struct LevelStats {
    pub level: usize,
    pub num_slots: usize,
    pub keys_placed: usize,
    pub fill_factor: f64,
}

pub struct LeveledMphf {
    splitmix: SplitMix64,
    level_meta: Vec<LevelMeta>,
    bitsets: Vec<RankedBitSet>,
}

#[inline(always)]
fn fastrange(hash: u64, n: usize) -> usize {
    ((hash as u128 * n as u128) >> 64) as usize
}

impl LeveledMphf {
    #[inline(always)]
    pub fn lookup(&self, key: u64) -> usize {
        debug_assert!(!self.level_meta.is_empty(), "lookup called on empty MPHF");

        // lvl 0 peeled out since cumulative_rank is always 0
        // SAFETY: level_meta and bitsets are non-empty
        let meta0 = unsafe { self.level_meta.get_unchecked(0) };
        let hash = self.splitmix.hash(key, meta0.offset_multiplier);
        let bit_idx = fastrange(hash, meta0.num_slots);
        if let Some(rank) = unsafe { self.bitsets.get_unchecked(0) }.rank_if_set(bit_idx) {
            return rank;
        }

        for level in 1..self.level_meta.len() {
            let meta = &self.level_meta[level];
            let hash = self.splitmix.hash(key, meta.offset_multiplier);
            let bit_idx = fastrange(hash, meta.num_slots);
            // SAFETY: bitsets.len() == level_meta.len(), and level < level_meta.len()
            if let Some(rank) = unsafe { self.bitsets.get_unchecked(level) }.rank_if_set(bit_idx) {
                return meta.cumulative_rank + rank;
            }
        }

        panic!("key {} does not exist", key);
    }

    pub fn level_stats(&self) -> Vec<LevelStats> {
        self.level_meta
            .iter()
            .enumerate()
            .map(|(i, m)| LevelStats {
                level: i,
                num_slots: m.num_slots,
                keys_placed: m.keys_placed,
                fill_factor: m.keys_placed as f64 / m.num_slots as f64,
            })
            .collect()
    }

    pub fn new(keys: &[u64], seed: u64, offset: u64, expansion_factor: f64) -> Self {
        let size = keys.len();
        let splitmix = SplitMix64::new(seed, offset);

        let mut level_meta = Vec::new();
        let mut bitsets = Vec::new();

        let mut remaining_keys = keys.to_vec();
        let mut current_offset_multiplier = 0u8;
        let mut items_placed_so_far = 0;

        // num_slots is strictly deincreasing across levels, so allocate counts once at the maximum possible size
        // reuse it every iteration by zeroing only the [..num_slots] prefix
        let max_slots = {
            let expanded = (size as f64 * expansion_factor).max(1.0) as usize;
            ((expanded + CACHE_LINE_BITS - 1) / CACHE_LINE_BITS) * CACHE_LINE_BITS
        };
        let mut counts = vec![0u8; max_slots];

        while !remaining_keys.is_empty() {
            let n = remaining_keys.len();

            // oversize to a cache-line-aligned slot count
            // also eliminates the tail edge case for small remaining sets
            let expanded = (n as f64 * expansion_factor).max(1.0) as usize;
            let num_slots = ((expanded + CACHE_LINE_BITS - 1) / CACHE_LINE_BITS) * CACHE_LINE_BITS;

            // poisson-approximated expected unique count: E[unique] = n * e^(-load)
            let load = n as f64 / num_slots as f64;
            let expected_unique = n as f64 * (-load).exp();

            let mut num_attempts = 1usize;

            loop {
                if current_offset_multiplier == 255 {
                    panic!("Exhausted all 255 offset multipliers after {} attempts! Try a different base seed.", num_attempts);
                }

                // zero out live prefixes
                counts[..num_slots].fill(0);
                // then hash and inc slot
                for &k in &remaining_keys {
                    let hash = splitmix.hash(k, current_offset_multiplier as u64);
                    let bit_idx = ((hash as u128 * num_slots as u128) >> 64) as usize;
                    counts[bit_idx] = counts[bit_idx].saturating_add(1);
                }

                let unique_count = counts[..num_slots].iter().filter(|&&c| c == 1).count();

                // reject if the hash performed worse than the poisson expectation
                if (unique_count as f64) < expected_unique {
                    num_attempts += 1;
                    current_offset_multiplier += 1;
                    continue;
                }

                // otherwise good load, commit this level
                let mut bs = BitSet::new(num_slots);
                let mut next_keys = Vec::with_capacity(n - unique_count);

                for &k in &remaining_keys {
                    let hash = splitmix.hash(k, current_offset_multiplier as u64);
                    let bit_idx = ((hash as u128 * num_slots as u128) >> 64) as usize;

                    if counts[bit_idx] == 1 {
                        bs.set(bit_idx);
                    } else {
                        next_keys.push(k);
                    }
                }

                level_meta.push(LevelMeta {
                    offset_multiplier: current_offset_multiplier as u64,
                    num_slots,
                    cumulative_rank: items_placed_so_far,
                    keys_placed: unique_count,
                });
                bitsets.push(RankedBitSet::new(bs));

                items_placed_so_far += unique_count;
                remaining_keys = next_keys;
                current_offset_multiplier += 1;

                break;
            }
        }

        Self {
            splitmix,
            level_meta,
            bitsets,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // lookup is a bijection onto [0, n).
    fn assert_bijection(mphf: &LeveledMphf, keys: &[u64]) {
        let n = keys.len();
        let mut outputs: Vec<usize> = keys.iter().map(|&k| mphf.lookup(k)).collect();
        outputs.sort_unstable();
        let expected: Vec<usize> = (0..n).collect();
        assert_eq!(outputs, expected, "lookup results are not a bijection onto [0, {n})");
    }

    const SEED: u64   = 0xa0761d6478bd642f;
    const OFFSET: u64 = 0xe7037ed1a0b428db;

    #[test]
    fn single_key() {
        let mphf = LeveledMphf::new(&[42], SEED, OFFSET, 1.5);
        assert_eq!(mphf.lookup(42), 0);
    }

    #[test]
    fn small_sequential_keys() {
        let keys: Vec<u64> = (0..20).collect();
        let mphf = LeveledMphf::new(&keys, SEED, OFFSET, 1.5);
        assert_bijection(&mphf, &keys);
    }

    #[test]
    fn large_sequential_keys() {
        let keys: Vec<u64> = (0..2000).collect();
        let mphf = LeveledMphf::new(&keys, SEED, OFFSET, 1.5);
        assert_bijection(&mphf, &keys);
    }

    #[test]
    fn sparse_keys() {
        let keys: Vec<u64> = (0..500u64).map(|i| i.wrapping_mul(0xdeadbeef)).collect();
        let mphf = LeveledMphf::new(&keys, SEED, OFFSET, 1.5);
        assert_bijection(&mphf, &keys);
    }


    // cargo test -- --ignored
    #[test]
    #[ignore]
    fn stress_large_scale() {
        const N: usize = 62_500_000;
        let keys: Vec<u64> = (0..N as u64).collect();
        let mphf = LeveledMphf::new(&keys, SEED, OFFSET, 1.5);

        let mut seen = vec![false; N];
        for &k in &keys {
            let idx = mphf.lookup(k);
            assert!(idx < N, "index {idx} out of bounds for key {k}");
            assert!(!seen[idx], "duplicate index {idx} for key {k}");
            seen[idx] = true;
        }
    }
}
