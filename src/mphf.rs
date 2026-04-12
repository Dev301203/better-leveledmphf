use core::panic;

#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(feature = "parallel")]
use std::cell::RefCell;

use crate::bitset::CACHE_LINE_BITS;
use crate::bitset::{BitSet, RankedBitSet};
use crate::hash::SplitMix64;

// Minimum number of keys before we spawn parallel work; below this, overhead dominates.
// TODO: Test to find a good limit
const PAR_THRESHOLD: usize = 131_072;

// Maximum γ when auto-tuning (2 * gamma_base); used for buffer sizing.
const GAMMA_MAX: f64 = 3.0;

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

// one level attempt - count half (classify is seq_classify_pass after Poisson acceptance)
// sequential counting, unique count
fn seq_count_pass(
    seq_counts: &mut [u8],
    remaining_keys: &[u64],
    num_slots: usize,
    om: u64,
    splitmix: &SplitMix64,
) -> usize {
    // counting pass
    // sequential path uses seq_counts
    seq_counts.fill(0);
    for &k in remaining_keys {
        let bit_idx = fastrange(splitmix.hash(k, om), num_slots);
        seq_counts[bit_idx] = seq_counts[bit_idx].saturating_add(1);
    }
    seq_counts.iter().filter(|&&c| c == 1).count()
}

// one level attempt - classify half (only after count passes Poisson gate)
// commit pass
// classify keys into placed (unique slot) vs survivors (collision)
// set bits and build next_keys
fn seq_classify_pass(
    seq_counts: &[u8],
    remaining_keys: &[u64],
    num_slots: usize,
    unique_count: usize,
    om: u64,
    splitmix: &SplitMix64,
) -> (Vec<u64>, BitSet) {
    let mut bs = BitSet::new(num_slots);
    let mut next_keys = Vec::with_capacity(remaining_keys.len() - unique_count);
    for &k in remaining_keys {
        let bit_idx = fastrange(splitmix.hash(k, om), num_slots);
        if seq_counts[bit_idx] == 1 {
            bs.set(bit_idx);
        } else {
            next_keys.push(k);
        }
    }
    (next_keys, bs)
}

#[cfg(feature = "parallel")]
thread_local! {
    // reused per worker across retries and levels; resize is a no-op after first level (num_slots strictly decreasing).
    static COUNT_THREAD_BUF: RefCell<Vec<u8>> = RefCell::new(Vec::new());
}

#[cfg(feature = "parallel")]
// one level attempt - count half (classify is par_classify_pass after Poisson acceptance)
// parallel counting (thread-local buffer, clone out for reduce), parallel unique count
fn par_count_pass(
    remaining_keys: &[u64],
    n: usize,
    num_slots: usize,
    om: u64,
    splitmix: &SplitMix64,
) -> (usize, Vec<u8>) {
    // counting pass
    // each worker fills its thread-local buffer, clones out for tree reduce.
    let chunk_size = n.div_ceil(rayon::current_num_threads()).max(1);
    let counts = remaining_keys
        .par_chunks(chunk_size)
        .map(|chunk| {
            COUNT_THREAD_BUF.with(|cell| {
                let mut buf = cell.borrow_mut();
                buf.resize(num_slots, 0);
                buf.fill(0);
                for &k in chunk {
                    let bit_idx = fastrange(splitmix.hash(k, om), num_slots);
                    buf[bit_idx] = buf[bit_idx].saturating_add(1);
                }
                buf.clone()
            })
        })
        .reduce(
            || vec![0u8; num_slots],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x = x.saturating_add(*y);
                }
                a
            },
        );
    let unique_count = counts.par_iter().filter(|&&c| c == 1).count();
    (unique_count, counts)
}

#[cfg(feature = "parallel")]
// one level attempt - classify half (only after count passes Poisson gate)
// parallel commit pass
// classify keys into placed (unique slot) vs survivors (collision)
// thread-local vecs, then set bits sequentially and concat survivors.
fn par_classify_pass(
    remaining_keys: &[u64],
    n: usize,
    num_slots: usize,
    unique_count: usize,
    om: u64,
    splitmix: &SplitMix64,
    counts: &[u8],
) -> (Vec<u64>, BitSet) {
    let mut bs = BitSet::new(num_slots);
    let chunk_size = n.div_ceil(rayon::current_num_threads()).max(1);
    let (placed_chunks, survivor_chunks): (Vec<Vec<usize>>, Vec<Vec<u64>>) = remaining_keys
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut placed = Vec::new();
            let mut survivors = Vec::new();
            for &k in chunk {
                let bit_idx = fastrange(splitmix.hash(k, om), num_slots);
                if counts[bit_idx] == 1 {
                    placed.push(bit_idx);
                } else {
                    survivors.push(k);
                }
            }
            (placed, survivors)
        })
        .unzip();

    // setting collected indices is O(unique_count).
    for placed in &placed_chunks {
        for &bit_idx in placed {
            bs.set(bit_idx);
        }
    }
    let mut next_keys = Vec::with_capacity(n - unique_count);
    for chunk in survivor_chunks {
        next_keys.extend(chunk);
    }
    (next_keys, bs)
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
        Self::new_with_par_threshold(keys, seed, offset, Some(expansion_factor), PAR_THRESHOLD)
    }

    // like [`new`](Self::new) but with adaptive γ per level: γ = 1.5 * (2 - n_remaining/n_original).
    pub fn new_auto_tuned(keys: &[u64], seed: u64, offset: u64) -> Self {
        Self::new_with_par_threshold(keys, seed, offset, None, PAR_THRESHOLD)
    }

    // same as [`new`](Self::new) but with a configurable parallelization threshold
    // `expansion_factor`: `Some(g)` = fixed γ; `None` = adaptive γ per level
    // use `usize::MAX` for `par_threshold` to force serial construction
    pub fn new_with_par_threshold(
        keys: &[u64],
        seed: u64,
        offset: u64,
        expansion_factor: Option<f64>,
        par_threshold: usize,
    ) -> Self {
        let size = keys.len();
        let splitmix = SplitMix64::new(seed, offset);

        let mut level_meta = Vec::new();
        let mut bitsets = Vec::new();

        let mut remaining_keys = keys.to_vec();
        let mut current_offset_multiplier = 0u8;
        let mut items_placed_so_far = 0;

        let gamma_base = expansion_factor.unwrap_or(1.5);
        let buffer_gamma = expansion_factor.unwrap_or(GAMMA_MAX);
        // num_slots is strictly decreasing across levels, so allocate counts once at the maximum possible size
        // reuse it every iteration by zeroing only the [..num_slots] prefix
        let max_slots = {
            let expanded = (size as f64 * buffer_gamma).max(1.0) as usize;
            ((expanded + CACHE_LINE_BITS - 1) / CACHE_LINE_BITS) * CACHE_LINE_BITS
        };
        let mut seq_counts = vec![0u8; max_slots];

        while !remaining_keys.is_empty() {
            let n = remaining_keys.len();
            let gamma = match expansion_factor {
                Some(g) => g,
                None => {
                    let ratio = n as f64 / size as f64;
                    gamma_base * (2.0 - ratio)
                }
            };
            // oversize to a cache-line-aligned slot count
            // also eliminates the tail edge case for small remaining sets
            let expanded = (n as f64 * gamma).max(1.0) as usize;
            let num_slots = ((expanded + CACHE_LINE_BITS - 1) / CACHE_LINE_BITS) * CACHE_LINE_BITS;

            // poisson-approximated expected unique count: E[unique] = n * e^(-load)
            let load = n as f64 / num_slots as f64;
            let expected_unique = n as f64 * (-load).exp();

            let mut num_attempts = 1usize;

            loop {
                if current_offset_multiplier == 255 {
                    panic!(
                        "Exhausted all 255 offset multipliers after {} attempts! Try a different base seed.",
                        num_attempts
                    );
                }

                let om = current_offset_multiplier as u64;

                // count pass first
                // classify only after Poisson acceptance (rejected attempts skip BitSet + next_keys)
                #[cfg(feature = "parallel")]
                let (unique_count, next_keys, bs) = if n >= par_threshold {
                    let (unique_count, counts) =
                        par_count_pass(&remaining_keys, n, num_slots, om, &splitmix);
                    // reject if the hash performed worse than the poisson expectation
                    if (unique_count as f64) < expected_unique {
                        num_attempts += 1;
                        current_offset_multiplier += 1;
                        continue;
                    }
                    let (next_keys, bs) = par_classify_pass(
                        &remaining_keys,
                        n,
                        num_slots,
                        unique_count,
                        om,
                        &splitmix,
                        &counts,
                    );
                    (unique_count, next_keys, bs)
                } else {
                    let unique_count = seq_count_pass(
                        &mut seq_counts[..num_slots],
                        &remaining_keys,
                        num_slots,
                        om,
                        &splitmix,
                    );
                    // reject if the hash performed worse than the poisson expectation
                    if (unique_count as f64) < expected_unique {
                        num_attempts += 1;
                        current_offset_multiplier += 1;
                        continue;
                    }
                    let (next_keys, bs) = seq_classify_pass(
                        &seq_counts[..num_slots],
                        &remaining_keys,
                        num_slots,
                        unique_count,
                        om,
                        &splitmix,
                    );
                    (unique_count, next_keys, bs)
                };
                #[cfg(not(feature = "parallel"))]
                let (unique_count, next_keys, bs) = {
                    let unique_count = seq_count_pass(
                        &mut seq_counts[..num_slots],
                        &remaining_keys,
                        num_slots,
                        om,
                        &splitmix,
                    );
                    // reject if the hash performed worse than the poisson expectation
                    if (unique_count as f64) < expected_unique {
                        num_attempts += 1;
                        current_offset_multiplier += 1;
                        continue;
                    }
                    let (next_keys, bs) = seq_classify_pass(
                        &seq_counts[..num_slots],
                        &remaining_keys,
                        num_slots,
                        unique_count,
                        om,
                        &splitmix,
                    );
                    (unique_count, next_keys, bs)
                };

                level_meta.push(LevelMeta {
                    offset_multiplier: om,
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
        assert_eq!(
            outputs, expected,
            "lookup results are not a bijection onto [0, {n})"
        );
    }

    const SEED: u64 = 0xa0761d6478bd642f;
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

    #[test]
    fn parallel_path_correct() {
        let keys: Vec<u64> = (0..10_000u64)
            .map(|i| i.wrapping_mul(0xcafe_babe_dead_beef))
            .collect();
        let mphf = LeveledMphf::new(&keys, SEED, OFFSET, 1.5);
        assert_bijection(&mphf, &keys);
    }

    #[test]
    fn auto_tuned_correct() {
        let keys: Vec<u64> = (0..5_000u64)
            .map(|i| i.wrapping_mul(0x1234_5678_9abc_def0))
            .collect();
        let mphf = LeveledMphf::new_auto_tuned(&keys, SEED, OFFSET);
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
