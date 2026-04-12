#[derive(Debug, Clone, Copy)]
pub(crate) struct SplitMix64 {
    seed: u64,
    offset: u64,
}

impl SplitMix64 {
    pub(crate) fn new(seed: u64, offset: u64) -> Self {
        SplitMix64 { seed, offset }
    }

    pub(crate) fn hash(&self, key: u64, num_offsets: u64) -> u64 {
        let mut z = key
            .wrapping_add(self.seed)
            .wrapping_add(num_offsets.wrapping_mul(self.offset))
            .wrapping_add(0x9e3779b97f4a7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
}
