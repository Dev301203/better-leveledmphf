# Better Leveled MPHF

Leveled minimal perfect hash function in Rust. Hashes `n` keys to `[0, n)` with no collisions

Keys are hashed into a bitset level by level. Keys that land in a unique slot are placed there, while the rest cascade to the next level. Lookup checks each level until it finds the key's slot and returns the cumulative rank.

## Implementation Details

- Fast range reduction instead of modulo
- Rank blocks are cache-aligned; bitsets are oversized to the nearest cache line
- Fused `rank_if_set`, single cache-line load does the get and popcount together, bails early on miss
- Level 0 peeled out since cumulative rank is always 0
- Poisson acceptance, construction rejects a hash function if unique-landing count falls below `n·e^(−load)`
- SplitMix64 hash, giga fast and random enough

## Usage

```rust
let keys: Vec<u64> = (0..1_000_000).collect();
let mphf = LeveledMphf::new(&keys, seed, offset, 1.5);
let idx = mphf.lookup(42); // in [0, n)
```
