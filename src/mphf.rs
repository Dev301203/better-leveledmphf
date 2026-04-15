use std::collections::HashSet;

#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(feature = "parallel")]
use std::cell::RefCell;

use crate::bitset::CACHE_LINE_BITS;
use crate::bitset::{BitSet, RankedBitSet};
use crate::hash::SplitMix64;

// Minimum number of keys before we spawn parallel work; below this, overhead dominates on current benchmarks
const PAR_THRESHOLD: usize = 131_072;

// Default γ base for auto-tuned construction.
const AUTO_TUNED_GAMMA_BASE: f64 = 1.5;

// Upper bound on total retry attempts consumed across all levels and retries
const MAX_OFFSET_ATTEMPTS: u64 = 1024;

struct LevelMeta {
    retry_seed: u64,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildError {
    EmptyKeys,
    DuplicateKey { key: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AutoGammaShape {
    Pow05,
    Pow10,
    Pow15,
    Piecewise,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastRangeMode {
    Low32,
    High32,
    Mul64,
}

#[derive(Debug, Clone, Copy)]
pub struct BuildOptions {
    expansion_factor: Option<f64>,
    auto_tuned_gamma_base: f64,
    auto_tuned_gamma_shape: AutoGammaShape,
    fastrange_mode: FastRangeMode,
    par_threshold: usize,
}

/// Defaults to auto-tuned construction with `AUTO_TUNED_GAMMA_BASE` and the crate default parallel threshold
impl Default for BuildOptions {
    fn default() -> Self {
        Self {
            expansion_factor: None,
            auto_tuned_gamma_base: AUTO_TUNED_GAMMA_BASE,
            auto_tuned_gamma_shape: AutoGammaShape::Pow10,
            fastrange_mode: FastRangeMode::Low32,
            par_threshold: PAR_THRESHOLD,
        }
    }
}

impl BuildOptions {
    pub fn fixed_gamma(mut self, expansion_factor: f64) -> Self {
        self.expansion_factor = Some(expansion_factor);
        self
    }

    pub fn auto_tuned(mut self) -> Self {
        self.expansion_factor = None;
        self
    }

    pub fn auto_tuned_gamma_base(mut self, gamma_base: f64) -> Self {
        self.auto_tuned_gamma_base = gamma_base;
        self
    }

    pub fn auto_tuned_gamma_shape(mut self, shape: AutoGammaShape) -> Self {
        self.auto_tuned_gamma_shape = shape;
        self
    }

    pub fn fastrange_mode(mut self, mode: FastRangeMode) -> Self {
        self.fastrange_mode = mode;
        self
    }

    pub fn par_threshold(mut self, par_threshold: usize) -> Self {
        self.par_threshold = par_threshold;
        self
    }
}

pub struct LeveledMphf {
    splitmix: SplitMix64,
    fastrange_mode: FastRangeMode,
    level_meta: Vec<LevelMeta>,
    bitsets: Vec<RankedBitSet>,
}

#[inline(always)]
fn fastrange_mul64(hash: u64, n: usize) -> usize {
    ((hash as u128 * n as u128) >> 64) as usize
}

#[inline(always)]
fn fastrange(hash: u64, n: usize, mode: FastRangeMode) -> usize {
    match mode {
        FastRangeMode::Low32 => {
            if n <= u32::MAX as usize {
                ((((hash as u32) as u64) * (n as u32 as u64)) >> 32) as usize
            } else {
                fastrange_mul64(hash, n)
            }
        }
        FastRangeMode::High32 => {
            if n <= u32::MAX as usize {
                (((((hash >> 32) as u32) as u64) * (n as u32 as u64)) >> 32) as usize
            } else {
                fastrange_mul64(hash, n)
            }
        }
        FastRangeMode::Mul64 => fastrange_mul64(hash, n),
    }
}

#[inline(always)]
fn round_slots(raw_slots: usize) -> usize {
    raw_slots.div_ceil(CACHE_LINE_BITS) * CACHE_LINE_BITS
}

#[inline(always)]
fn auto_gamma(gamma_base: f64, ratio: f64, shape: AutoGammaShape) -> f64 {
    let ratio = ratio.clamp(0.0, 1.0);
    match shape {
        AutoGammaShape::Pow05 => gamma_base * (1.0 + ratio.sqrt()),
        AutoGammaShape::Pow10 => gamma_base * (1.0 + ratio),
        AutoGammaShape::Pow15 => gamma_base * (1.0 + ratio * ratio.sqrt()),
        AutoGammaShape::Piecewise => {
            if ratio >= 0.75 {
                gamma_base * 2.0
            } else if ratio >= 0.5 {
                gamma_base * 1.8
            } else if ratio >= 0.25 {
                gamma_base * 1.55
            } else {
                gamma_base * 1.35
            }
        }
    }
}

// one level attempt - count half (classify is seq_classify_pass after Poisson acceptance)
// sequential counting, unique count
fn seq_count_pass(
    seq_counts: &mut [u8],
    bucket_ids: &mut [u32],
    remaining_keys: &[u64],
    num_slots: usize,
    retry_seed: u64,
    splitmix: &SplitMix64,
    fastrange_mode: FastRangeMode,
) -> usize {
    // counting pass
    // sequential path uses seq_counts and caches bucket ids so accepted attempts do not rehash
    debug_assert!(
        num_slots <= u32::MAX as usize,
        "num_slots {} exceeds cached bucket id range",
        num_slots
    );
    seq_counts.fill(0);
    for (bucket_id, &k) in bucket_ids.iter_mut().zip(remaining_keys.iter()) {
        let bit_idx = fastrange(
            splitmix.hash_with_retry_seed(k, retry_seed),
            num_slots,
            fastrange_mode,
        );
        *bucket_id = bit_idx as u32;
        seq_counts[bit_idx] = seq_counts[bit_idx].saturating_add(1);
    }
    seq_counts.iter().filter(|&&c| c == 1).count()
}

// one level attempt - classify half (only after count passes Poisson gate)
// commit pass
// partition `remaining_keys` in place (write head + truncate to survivor count)
// singleton slot keys only set bits
// collision keys are gathered toward lower indices as we scan, then the vec length is set to the survivor count
fn seq_classify_pass_inplace(
    seq_counts: &[u8],
    bucket_ids: &[u32],
    remaining_keys: &mut Vec<u64>,
    num_slots: usize,
    unique_count: usize,
) -> BitSet {
    let mut bs = BitSet::new(num_slots);
    let n = remaining_keys.len();
    let mut w = 0usize;
    for i in 0..n {
        let k = remaining_keys[i];
        let bit_idx = bucket_ids[i] as usize;
        if seq_counts[bit_idx] == 1 {
            bs.set(bit_idx);
        } else {
            remaining_keys[w] = k;
            w += 1;
        }
    }
    debug_assert_eq!(w, n - unique_count);
    remaining_keys.truncate(w);
    bs
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
    chunk_size: usize,
    num_slots: usize,
    retry_seed: u64,
    splitmix: &SplitMix64,
    fastrange_mode: FastRangeMode,
) -> (usize, Vec<u8>) {
    // counting pass
    // each worker fills its thread-local buffer, clones out for tree reduce.
    let counts = remaining_keys
        .par_chunks(chunk_size)
        .map(|chunk| {
            COUNT_THREAD_BUF.with(|cell| {
                let mut buf = cell.borrow_mut();
                buf.resize(num_slots, 0);
                buf.fill(0);
                for &k in chunk {
                    let bit_idx =
                        fastrange(
                            splitmix.hash_with_retry_seed(k, retry_seed),
                            num_slots,
                            fastrange_mode,
                        );
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
    chunk_size: usize,
    n: usize,
    num_slots: usize,
    unique_count: usize,
    retry_seed: u64,
    splitmix: &SplitMix64,
    counts: &[u8],
    fastrange_mode: FastRangeMode,
) -> (Vec<u64>, BitSet) {
    let mut bs = BitSet::new(num_slots);
    let (placed_chunks, survivor_chunks): (Vec<Vec<usize>>, Vec<Vec<u64>>) = remaining_keys
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut placed = Vec::new();
            let mut survivors = Vec::new();
            for &k in chunk {
                let bit_idx = fastrange(
                    splitmix.hash_with_retry_seed(k, retry_seed),
                    num_slots,
                    fastrange_mode,
                );
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
    fn validate_keys(keys: &[u64]) -> Result<(), BuildError> {
        if keys.is_empty() {
            return Err(BuildError::EmptyKeys);
        }

        let mut seen = HashSet::with_capacity(keys.len());
        for &key in keys {
            if !seen.insert(key) {
                return Err(BuildError::DuplicateKey { key });
            }
        }

        Ok(())
    }

    fn build_unchecked(keys: &[u64], seed: u64, offset: u64, options: BuildOptions) -> Self {
        #[cfg(not(feature = "parallel"))]
        let _ = options.par_threshold;

        assert!(
            !keys.is_empty(),
            "LeveledMphf constructors require at least one key"
        );

        let size = keys.len();
        let splitmix = SplitMix64::new(seed, offset);

        let mut level_meta = Vec::new();
        let mut bitsets = Vec::new();

        let mut remaining_keys = keys.to_vec();
        let mut total_attempts = 0u64;
        let mut items_placed_so_far = 0;

        let gamma_base = options.auto_tuned_gamma_base;
        let buffer_gamma = options
            .expansion_factor
            .unwrap_or(options.auto_tuned_gamma_base * 2.0);
        // num_slots is strictly decreasing across levels, so allocate counts once at the maximum possible size
        // reuse it every iteration by zeroing only the [..num_slots] prefix
        let max_slots = {
            let expanded = (size as f64 * buffer_gamma).max(1.0) as usize;
            round_slots(expanded)
        };
        let mut seq_counts = vec![0u8; max_slots];
        let mut seq_bucket_ids = vec![0u32; size];

        while !remaining_keys.is_empty() {
            let n = remaining_keys.len();
            let gamma = match options.expansion_factor {
                Some(g) => g,
                None => {
                    let ratio = n as f64 / size as f64;
                    auto_gamma(gamma_base, ratio, options.auto_tuned_gamma_shape)
                }
            };
            // oversize to a cache-line-aligned slot count
            // also eliminates the tail edge case for small remaining sets
            let expanded = (n as f64 * gamma).max(1.0) as usize;
            let num_slots = round_slots(expanded);

            // poisson-approximated expected unique count: E[unique] = n * e^(-load)
            let load = n as f64 / num_slots as f64;
            let expected_unique = n as f64 * (-load).exp();

            let level_index = level_meta.len() as u64;
            let mut attempt_in_level = 0u64;
            loop {
                if total_attempts >= MAX_OFFSET_ATTEMPTS {
                    panic!(
                        "construction stalled after {} total attempts; try a different seed",
                        MAX_OFFSET_ATTEMPTS
                    );
                }

                let retry_seed = splitmix.retry_seed(attempt_in_level, level_index);
                total_attempts += 1;

                // count pass first
                // classify only after Poisson acceptance (rejected attempts skip BitSet + next_keys)
                #[cfg(feature = "parallel")]
                // only large remaining_key sets pay Rayon overhead
                let do_parallel = n >= options.par_threshold;
                #[cfg(not(feature = "parallel"))]
                // no Rayon in dependency graph
                let do_parallel = false;

                if do_parallel {
                    #[cfg(feature = "parallel")]
                    {
                        let chunk_size = n.div_ceil(rayon::current_num_threads()).max(1);
                        let (unique_count, counts) = par_count_pass(
                            &remaining_keys,
                            chunk_size,
                            num_slots,
                            retry_seed,
                            &splitmix,
                            options.fastrange_mode,
                        );
                        // reject if the hash performed worse than the poisson expectation
                        if (unique_count as f64) < expected_unique {
                            attempt_in_level += 1;
                            continue;
                        }
                        let (next_keys, bs) = par_classify_pass(
                            &remaining_keys,
                            chunk_size,
                            n,
                            num_slots,
                            unique_count,
                            retry_seed,
                            &splitmix,
                            &counts,
                            options.fastrange_mode,
                        );

                        level_meta.push(LevelMeta {
                            retry_seed,
                            num_slots,
                            cumulative_rank: items_placed_so_far,
                            keys_placed: unique_count,
                        });
                        bitsets.push(RankedBitSet::new(bs));

                        items_placed_so_far += unique_count;
                        remaining_keys = next_keys;
                        break;
                    }
                    #[cfg(not(feature = "parallel"))]
                    {
                        unreachable!("parallel path only when the parallel feature is enabled")
                    }
                } else {
                    // sequential count + classify
                    let unique_count = seq_count_pass(
                        &mut seq_counts[..num_slots],
                        &mut seq_bucket_ids[..n],
                        &remaining_keys,
                        num_slots,
                        retry_seed,
                        &splitmix,
                        options.fastrange_mode,
                    );
                    // reject if the hash performed worse than the poisson expectation
                    if (unique_count as f64) < expected_unique {
                        attempt_in_level += 1;
                        continue;
                    }
                    let bs = seq_classify_pass_inplace(
                        &seq_counts[..num_slots],
                        &seq_bucket_ids[..n],
                        &mut remaining_keys,
                        num_slots,
                        unique_count,
                    );

                    level_meta.push(LevelMeta {
                        retry_seed,
                        num_slots,
                        cumulative_rank: items_placed_so_far,
                        keys_placed: unique_count,
                    });
                    bitsets.push(RankedBitSet::new(bs));

                    items_placed_so_far += unique_count;
                    break;
                }
            }
        }

        Self {
            splitmix,
            fastrange_mode: options.fastrange_mode,
            level_meta,
            bitsets,
        }
    }

    #[inline(always)]
    pub fn lookup(&self, key: u64) -> usize {
        debug_assert!(!self.level_meta.is_empty(), "lookup called on empty MPHF");

        // lvl 0 peeled out since cumulative_rank is always 0
        // SAFETY: level_meta and bitsets are non-empty
        let meta0 = unsafe { self.level_meta.get_unchecked(0) };
        let hash = self.splitmix.hash_with_retry_seed(key, meta0.retry_seed);
        let bit_idx = fastrange(hash, meta0.num_slots, self.fastrange_mode);
        if let Some(rank) = unsafe { self.bitsets.get_unchecked(0) }.rank_if_set(bit_idx) {
            return rank;
        }

        for level in 1..self.level_meta.len() {
            let meta = &self.level_meta[level];
            let hash = self.splitmix.hash_with_retry_seed(key, meta.retry_seed);
            let bit_idx = fastrange(hash, meta.num_slots, self.fastrange_mode);
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
        Self::build_unchecked(
            keys,
            seed,
            offset,
            BuildOptions::default().fixed_gamma(expansion_factor),
        )
    }

    // like [`new`](Self::new) but with adaptive γ per level controlled by BuildOptions::auto_tuned_gamma_shape.
    pub fn new_auto_tuned(keys: &[u64], seed: u64, offset: u64) -> Self {
        Self::build_unchecked(keys, seed, offset, BuildOptions::default().auto_tuned())
    }

    pub fn new_with_options(keys: &[u64], seed: u64, offset: u64, options: BuildOptions) -> Self {
        Self::build_unchecked(keys, seed, offset, options)
    }

    /// Validates that `keys` is non-empty and duplicate-free before construction.
    /// This duplicate check allocates O(n) extra memory, so trusted large inputs should prefer [`new`](Self::new).
    pub fn try_new(
        keys: &[u64],
        seed: u64,
        offset: u64,
        expansion_factor: f64,
    ) -> Result<Self, BuildError> {
        Self::try_new_with_options(
            keys,
            seed,
            offset,
            BuildOptions::default().fixed_gamma(expansion_factor),
        )
    }

    /// Validates that `keys` is non-empty and duplicate-free before construction.
    /// This duplicate check allocates O(n) extra memory, so trusted large inputs should prefer [`new_auto_tuned`](Self::new_auto_tuned).
    pub fn try_new_auto_tuned(keys: &[u64], seed: u64, offset: u64) -> Result<Self, BuildError> {
        Self::try_new_with_options(keys, seed, offset, BuildOptions::default().auto_tuned())
    }

    /// Validates that `keys` is non-empty and duplicate-free before construction.
    /// This duplicate check allocates O(n) extra memory, so trusted large inputs should prefer [`new_with_options`](Self::new_with_options).
    pub fn try_new_with_options(
        keys: &[u64],
        seed: u64,
        offset: u64,
        options: BuildOptions,
    ) -> Result<Self, BuildError> {
        Self::validate_keys(keys)?;
        Ok(Self::build_unchecked(keys, seed, offset, options))
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
    fn empty_keys_rejected() {
        let err = LeveledMphf::try_new(&[], SEED, OFFSET, 1.5)
            .err()
            .expect("empty input should be rejected");
        assert_eq!(err, BuildError::EmptyKeys);
    }

    #[test]
    fn duplicate_keys_rejected() {
        let err = LeveledMphf::try_new(&[1, 2, 2, 3], SEED, OFFSET, 1.5)
            .err()
            .expect("duplicate input should be rejected");
        assert_eq!(err, BuildError::DuplicateKey { key: 2 });
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
