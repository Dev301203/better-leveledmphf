pub const CACHE_LINE_BYTES: usize = 64;
pub const CACHE_LINE_BITS: usize = CACHE_LINE_BYTES * 8;


#[repr(C, align(64))]
#[derive(Clone, Default)]
struct CacheLine {
    line: [u64; CACHE_LINE_BYTES / std::mem::size_of::<u64>()],
}

pub(crate) struct BitSet {
    data: Vec<CacheLine>,
    pub(crate) size: usize,
}

impl BitSet {
    pub(crate) fn new(size: usize) -> Self {
        // ceil of size / CACHE_LINE_BITS
        // (a + b - 1) / b
        let num_cache_lines = (size + CACHE_LINE_BITS - 1) / CACHE_LINE_BITS;
        BitSet {
            data: vec![CacheLine::default(); num_cache_lines],
            size,
        }
    }

    #[inline(always)]
    pub(crate) fn set(&mut self, idx: usize) {
        debug_assert!(
            idx < self.size,
            "index {idx} out of bounds for BitSet of size {}",
            self.size
        );
        let cache_line_idx = idx >> 9;
        let u64_idx = (idx & 511) >> 6;
        let bit_idx = idx & 63;
        let mask = 1u64 << bit_idx;
        self.data[cache_line_idx].line[u64_idx] |= mask;
    }

    #[inline(always)]
    pub fn get(&self, idx: usize) -> bool {
        debug_assert!(
            idx < self.size,
            "index {idx} out of bounds for BitSet of size {}",
            self.size
        );
        let cache_line_idx = idx >> 9;
        let u64_idx = (idx & 511) >> 6;
        let bit_idx = idx & 63;
        let mask = 1u64 << bit_idx;
        self.data[cache_line_idx].line[u64_idx] & mask != 0
    }
}

pub(crate) struct RankedBitSet {
    pub bs: BitSet,
    rank_prefix: Vec<usize>,
}

impl RankedBitSet {
    pub(crate) fn new(bs: BitSet) -> Self {
        assert!(
            bs.data.as_ptr() as usize % CACHE_LINE_BYTES == 0,
            "BitSet vec is not cache aligned"
        );

        // rank_prefix[i] is number of set bits in all cache lines before i
        // one entry per cache line
        let mut rank_prefix = Vec::with_capacity(bs.data.len());
        let mut cumulative = 0usize;
        for line in &bs.data {
            rank_prefix.push(cumulative);
            for &word in &line.line {
                cumulative += word.count_ones() as usize;
            }
        }

        RankedBitSet { bs, rank_prefix }
    }

    #[inline(always)]
    pub(crate) fn get(&self, idx: usize) -> bool {
        self.bs.get(idx)
    }

    // returns Some(rank) if the bit at idx is set, None otherwise
    // Fuses the get + rank into a single index decomposition and word load.
    #[inline(always)]
    pub(crate) fn rank_if_set(&self, idx: usize) -> Option<usize> {
        let cache_line_idx = idx >> 9;
        let u64_idx = (idx & 511) >> 6;
        let bit_idx = idx & 63;

        // SAFETY: `idx` comes from fastrange(hash, bs.size), so idx < bs.size.
        // bs.data.len() == ceil(bs.size / 512), so cache_line_idx = idx / 512 < bs.data.len().
        // rank_prefix.len() == bs.data.len() by construction in RankedBitSet::new.
        let cache_line = unsafe { self.bs.data.get_unchecked(cache_line_idx) };
        let word = cache_line.line[u64_idx]; // u64_idx = (idx & 511) >> 6 < 8, always in bounds

        if word & (1u64 << bit_idx) == 0 {
            return None;
        }

        let mut count = unsafe { *self.rank_prefix.get_unchecked(cache_line_idx) };
        for i in 0..u64_idx {
            count += cache_line.line[i].count_ones() as usize; // i < u64_idx <= 7 < 8
        }
        let partial_mask = (1u64 << bit_idx) - 1;
        count += (word & partial_mask).count_ones() as usize;
        Some(count)
    }

    #[inline(always)]
    pub(crate) fn rank(&self, idx: usize) -> usize {
        debug_assert!(
            idx < self.bs.size,
            "index {idx} out of bounds for RankedBitSet of size {}",
            self.bs.size
        );
        let cache_line_idx = idx >> 9;
        let u64_idx = (idx & 511) >> 6;
        let bit_idx = idx & 63;

        // prefix popcount from prior cache lines
        let mut count = self.rank_prefix[cache_line_idx];

        // full u64s within this cache line
        for i in 0..u64_idx {
            count += self.bs.data[cache_line_idx].line[i].count_ones() as usize;
        }

        // partial u64: bits 0..bit_idx (exclude bit_idx itself)
        // (1u64 << bit_idx) - 1 is safe: shift is at most 63, no overflow
        let partial_mask = (1u64 << bit_idx) - 1;
        count += (self.bs.data[cache_line_idx].line[u64_idx] & partial_mask).count_ones() as usize;

        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitset_all_unset_initially() {
        let bs = BitSet::new(600);
        for i in 0..600 {
            assert!(!bs.get(i), "bit {i} should start unset");
        }
    }

    #[test]
    fn bitset_set_and_get_roundtrip() {
        let mut bs = BitSet::new(600);
        let positions = [0, 1, 63, 64, 511, 512, 599];
        for &p in &positions {
            bs.set(p);
        }
        for i in 0..600 {
            assert_eq!(bs.get(i), positions.contains(&i), "mismatch at bit {i}");
        }
    }

    // consecutive CacheLines are actually 64-byte aligned and laid out contiguously
    #[test]
    fn bitset_cache_line_layout() {
        let bs = BitSet::new(CACHE_LINE_BITS * 2);

        let ptr0 = bs.data[0].line.as_ptr() as usize;
        let ptr1 = bs.data[1].line.as_ptr() as usize;

        // 64-byte aligned.
        assert_eq!(ptr0 % CACHE_LINE_BYTES, 0);
        assert_eq!(ptr1 % CACHE_LINE_BYTES, 0);

        // two cache lines must be exactly CACHE_LINE_BYTES apart
        assert_eq!(ptr1 - ptr0, CACHE_LINE_BYTES);

        // or, the last u64 of cache line 0 is immediately before the first u64 of cache line 1.
        let last_word_of_line0 = &bs.data[0].line[bs.data[0].line.len() - 1] as *const u64 as usize;
        assert_eq!(ptr1 - last_word_of_line0, std::mem::size_of::<u64>());
    }
    #[test]
    fn rank_counts_preceding_set_bits() {
        let mut bs = BitSet::new(100);
        bs.set(3);
        bs.set(7);
        bs.set(10);
        let rbs = RankedBitSet::new(bs);
        assert_eq!(rbs.rank(3), 0);  // nothing before index 3
        assert_eq!(rbs.rank(7), 1);  // {3}
        assert_eq!(rbs.rank(10), 2); // {3, 7}
        assert_eq!(rbs.rank(11), 3); // {3, 7, 10}
        assert_eq!(rbs.rank(99), 3); // still only three set bits
    }

    // rank must remain correct when set bits straddle the cache-line boundary.
    #[test]
    fn rank_across_cache_line_boundary() {
        let boundary = CACHE_LINE_BITS; // 512
        let mut bs = BitSet::new(boundary * 2);
        bs.set(boundary - 12); // in cache line 0
        bs.set(boundary - 1);  // last bit of cache line 0
        bs.set(boundary + 7);  // in cache line 1
        let rbs = RankedBitSet::new(bs);

        assert_eq!(rbs.rank(boundary - 12), 0); // nothing before it
        assert_eq!(rbs.rank(boundary - 1), 1);  // {boundary-12}
        assert_eq!(rbs.rank(boundary), 2);       // {boundary-12, boundary-1}
        assert_eq!(rbs.rank(boundary + 7), 2);  // same two; boundary+7 not yet counted
        assert_eq!(rbs.rank(boundary + 8), 3);  // now includes boundary+7
    }
}
