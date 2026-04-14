#[derive(Debug, Clone, Copy)]
pub(crate) struct SplitMix64 {
    base_seed: u64,
    retry_stream: u64,
}

impl SplitMix64 {
    pub(crate) fn new(base_seed: u64, retry_stream: u64) -> Self {
        SplitMix64 {
            base_seed,
            retry_stream,
        }
    }

    #[inline(always)]
    fn mix(mut z: u64) -> u64 {
        z = z.wrapping_add(0x9e3779b97f4a7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    #[inline(always)]
    pub(crate) fn retry_seed(&self, attempt: u64, stream: u64) -> u64 {
        Self::mix(
            self.base_seed ^ self.retry_stream ^ stream ^ attempt.wrapping_mul(0x9e3779b97f4a7c15),
        )
    }

    #[inline(always)]
    pub(crate) fn hash_with_retry_seed(&self, key: u64, retry_seed: u64) -> u64 {
        Self::mix(key ^ retry_seed)
    }
}
