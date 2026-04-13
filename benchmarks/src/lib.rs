//! Shared helpers for `bench_compare` (serial) and `bench_compare_parallel` binaries.

use better_mphf::LeveledMphf;
use boomphf::Mphf;
#[cfg(feature = "parallel")]
use rayon::ThreadPool;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

pub const SEED: u64 = 0xa0761d6478bd642f;
pub const OFFSET: u64 = 0xe7037ed1a0b428db;

pub const NS_GRID: &[usize] = &[
    512, 1_024, 2_048, 4_096, 8_192, 16_384, 32_768, 65_536, 131_072, 262_144, 524_288, 1_048_576,
    2_097_152,
];

pub const NS_LARGE: &[usize] = &[
    1_048_576, 2_097_152, 4_194_304, 8_388_608, 16_777_216, 33_554_432,
];

pub const GAMMAS: &[f64] = &[1.15, 1.25, 1.35, 1.5, 1.65, 1.8, 2.0, 2.25, 2.5];

pub const THREAD_COUNTS: &[usize] = &[1, 2, 4, 8, 16, 24, 32];

/// Warmup / timed **construction** runs for n < 1M (grid cells).
pub const WARMUP_GRID: u32 = 2;
pub const TIMED_GRID: u32 = 7;

/// Warmup / timed construction runs for n ≥ 1M (appendix + any large row).
pub const WARMUP_LARGE: u32 = 2;
pub const TIMED_LARGE: u32 = 7;

pub const LOOKUP_TIMED: u32 = 5;

pub const GAMMA_LARGE_N: f64 = 1.5;

pub fn make_keys(n: usize) -> Vec<u64> {
    (0..n as u64)
        .map(|i| i.wrapping_mul(0x9e37_79b9_7f4a_7c15))
        .collect()
}

/// When both `MPHF_BENCH_WARMUP` and `MPHF_BENCH_TIMED` are set (integers), they override
/// per-cell warmup/timed counts for **all** sizes (useful on Slurm for e.g. 5 / 21).
pub fn warmup_timed_for_n(n: usize) -> (u32, u32) {
    if let (Ok(ws), Ok(ts)) = (
        std::env::var("MPHF_BENCH_WARMUP"),
        std::env::var("MPHF_BENCH_TIMED"),
    ) {
        if let (Ok(wu), Ok(td)) = (ws.parse::<u32>(), ts.parse::<u32>()) {
            return (wu, td.max(1));
        }
    }
    if n >= 1_048_576 {
        (WARMUP_LARGE, TIMED_LARGE)
    } else {
        (WARMUP_GRID, TIMED_GRID)
    }
}

/// Each timed iteration measured separately; returns **(median_ms, min_ms, max_ms)**.
pub fn sample_after_warmup(
    warmup: u32,
    timed: u32,
    mut one_build: impl FnMut(),
) -> (f64, f64, f64) {
    for _ in 0..warmup {
        one_build();
    }
    let mut samples = Vec::with_capacity(timed as usize);
    for _ in 0..timed {
        let t0 = Instant::now();
        one_build();
        samples.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    median_min_max(&mut samples)
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

/// Fixed width for `format_mmm_wide` / `format_timing_na` so table columns line up.
pub const MMM_CELL_WIDTH: usize = 28;

pub fn format_mmm_wide(med: f64, min: f64, max: f64) -> String {
    let inner = format!("{med:>10.2} [{min:.2}–{max:.2}]");
    format!("{inner:>w$}", w = MMM_CELL_WIDTH)
}

pub fn format_timing_na() -> String {
    format!("{:>w$}", "n/a", w = MMM_CELL_WIDTH)
}

pub fn timing_compare_table_rule() -> String {
    "-".repeat(10 + 3 + MMM_CELL_WIDTH * 3 + 3 + 8 + 3 + 8)
}

pub fn print_compare_timing_header(bmf_col: &str, boom_col: &str) {
    let w = MMM_CELL_WIDTH;
    println!(
        "{:>10} | {:>w$} | {:>w$} | {:>w$} | {:>8} | {:>8}",
        "n",
        bmf_col,
        boom_col,
        "bbhash(C++)",
        "bmf-lk",
        "boom-lk",
        w = w
    );
    println!("{}", timing_compare_table_rule());
}

pub fn time_better_serial_mmm(
    keys: &[u64],
    gamma: f64,
    warmup: u32,
    timed: u32,
) -> (f64, f64, f64) {
    let keys = keys;
    let gamma = gamma;
    sample_after_warmup(warmup, timed, || {
        let _ = LeveledMphf::new_with_par_threshold(keys, SEED, OFFSET, Some(gamma), usize::MAX);
    })
}

#[cfg(feature = "parallel")]
pub fn time_better_parallel_mmm(
    pool: &ThreadPool,
    keys: &[u64],
    gamma: f64,
    warmup: u32,
    timed: u32,
) -> (f64, f64, f64) {
    let keys = keys;
    let gamma = gamma;
    pool.install(|| {
        sample_after_warmup(warmup, timed, || {
            let _ = LeveledMphf::new(keys, SEED, OFFSET, gamma);
        })
    })
}

pub fn time_boom_serial_mmm(keys: &[u64], gamma: f64, warmup: u32, timed: u32) -> (f64, f64, f64) {
    let keys = keys;
    let gamma = gamma;
    sample_after_warmup(warmup, timed, || {
        let _ = Mphf::new(gamma, keys);
    })
}

#[cfg(feature = "parallel")]
pub fn time_boom_parallel_mmm(
    pool: &ThreadPool,
    keys: &[u64],
    gamma: f64,
    warmup: u32,
    timed: u32,
) -> (f64, f64, f64) {
    let keys = keys;
    let gamma = gamma;
    pool.install(|| {
        sample_after_warmup(warmup, timed, || {
            let _ = Mphf::new_parallel(gamma, keys, None);
        })
    })
}

pub fn lookup_better_serial_ms(keys: &[u64], gamma: f64) -> f64 {
    let mphf = LeveledMphf::new_with_par_threshold(keys, SEED, OFFSET, Some(gamma), usize::MAX);
    let t0 = Instant::now();
    for _ in 0..LOOKUP_TIMED {
        for &k in keys {
            black_box(mphf.lookup(k));
        }
    }
    t0.elapsed().as_secs_f64() * 1000.0 / LOOKUP_TIMED as f64
}

#[cfg(feature = "parallel")]
pub fn lookup_better_parallel_ms(pool: &ThreadPool, keys: &[u64], gamma: f64) -> f64 {
    let mphf = pool.install(|| LeveledMphf::new(keys, SEED, OFFSET, gamma));
    let t0 = Instant::now();
    for _ in 0..LOOKUP_TIMED {
        for &k in keys {
            black_box(mphf.lookup(k));
        }
    }
    t0.elapsed().as_secs_f64() * 1000.0 / LOOKUP_TIMED as f64
}

pub fn lookup_boom_ms(keys: &[u64], gamma: f64) -> f64 {
    let mphf = Mphf::new(gamma, keys);
    let t0 = Instant::now();
    for _ in 0..LOOKUP_TIMED {
        for &k in keys {
            black_box(mphf.hash(&k));
        }
    }
    t0.elapsed().as_secs_f64() * 1000.0 / LOOKUP_TIMED as f64
}

#[cfg(feature = "parallel")]
pub fn lookup_boom_parallel_ms(pool: &ThreadPool, keys: &[u64], gamma: f64) -> f64 {
    let mphf = pool.install(|| Mphf::new_parallel(gamma, keys, None));
    let t0 = Instant::now();
    for _ in 0..LOOKUP_TIMED {
        for &k in keys {
            black_box(mphf.hash(&k));
        }
    }
    t0.elapsed().as_secs_f64() * 1000.0 / LOOKUP_TIMED as f64
}

pub fn resolve_bbhash_exe() -> Option<PathBuf> {
    let cpp = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cpp");
    let native = cpp.join("bbhash_bench");
    if native.exists() {
        return Some(native);
    }
    #[cfg(windows)]
    {
        let win_exe = cpp.join("bbhash_bench.exe");
        if win_exe.exists() {
            return Some(win_exe);
        }
    }
    None
}

pub fn run_bbhash_subprocess_once(
    exe: &Path,
    nelem: usize,
    nthreads: usize,
    gamma: f64,
) -> Option<f64> {
    let out = Command::new(exe)
        .arg(nelem.to_string())
        .arg(nthreads.to_string())
        .arg(format!("{gamma:.6}"))
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let line = String::from_utf8_lossy(&out.stdout);
    let line = line.trim();
    let last = line.lines().last().unwrap_or(line);
    let mut parts = last.split(',');
    let _ne = parts.next()?;
    let _nt = parts.next()?;
    let _g = parts.next()?;
    let ms_str = parts.next()?;
    ms_str.parse().ok()
}

pub fn run_bbhash_subprocess_mmm(
    exe: &Path,
    nelem: usize,
    nthreads: usize,
    gamma: f64,
    warmup: u32,
    timed: u32,
) -> Option<(f64, f64, f64)> {
    for _ in 0..warmup {
        let _ = run_bbhash_subprocess_once(exe, nelem, nthreads, gamma)?;
    }
    let mut samples = Vec::with_capacity(timed as usize);
    for _ in 0..timed {
        samples.push(run_bbhash_subprocess_once(exe, nelem, nthreads, gamma)?);
    }
    Some(median_min_max(&mut samples))
}

pub fn hostname_hint() -> String {
    use std::env::var;
    if let Ok(h) = var("HOSTNAME") {
        return h;
    }
    if let Ok(h) = var("COMPUTERNAME") {
        return h;
    }
    // Slurm compute step (often unset in non-interactive jobs)
    if let Ok(h) = var("SLURMD_NODENAME") {
        return h;
    }
    #[cfg(unix)]
    {
        if let Ok(s) = std::fs::read_to_string("/etc/hostname") {
            let h = s.trim();
            if !h.is_empty() {
                return h.to_string();
            }
        }
    }
    "(unknown host)".into()
}
