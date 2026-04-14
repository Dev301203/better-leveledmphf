use mphf_benchmarks::{make_keys, OFFSET, SEED};
use std::env;
use std::process;

const CACHE_LINE_BITS: usize = 512;
const GOLDEN: u64 = 0x9e3779b97f4a7c15;

#[derive(Clone, Copy)]
enum HashFamily {
    Additive,
    XorRemixed,
}

impl HashFamily {
    fn name(self) -> &'static str {
        match self {
            Self::Additive => "additive",
            Self::XorRemixed => "xor-remixed",
        }
    }
}

struct Config {
    n: usize,
    gamma: f64,
    attempts: usize,
    print_attempts: usize,
}

fn usage() -> ! {
    eprintln!(
        "usage: diagnose_hash_family --n <usize> --gamma <f64> [--attempts <usize>] [--print-attempts <usize>]"
    );
    process::exit(2);
}

fn parse_flag<T: std::str::FromStr>(args: &[String], flag: &str, default: Option<T>) -> T {
    if let Some(i) = args.iter().position(|a| a == flag) {
        return args
            .get(i + 1)
            .unwrap_or_else(|| usage())
            .parse::<T>()
            .unwrap_or_else(|_| usage());
    }
    default.unwrap_or_else(|| usage())
}

fn parse_args() -> Config {
    let args: Vec<String> = env::args().collect();
    Config {
        n: parse_flag(&args, "--n", None),
        gamma: parse_flag(&args, "--gamma", None),
        attempts: parse_flag(&args, "--attempts", Some(64)),
        print_attempts: parse_flag(&args, "--print-attempts", Some(16)),
    }
}

#[inline(always)]
fn fastrange(hash: u64, n: usize) -> usize {
    ((hash as u128 * n as u128) >> 64) as usize
}

#[inline(always)]
fn splitmix_scalar(mut z: u64) -> u64 {
    z = z.wrapping_add(GOLDEN);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

#[inline(always)]
fn hash_additive(key: u64, om: u64) -> u64 {
    let mut z = key
        .wrapping_add(SEED)
        .wrapping_add(om.wrapping_mul(OFFSET))
        .wrapping_add(GOLDEN);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

#[inline(always)]
fn hash_xor_remixed(key: u64, om: u64) -> u64 {
    let om_hash = splitmix_scalar(om.wrapping_mul(OFFSET));
    let mut z = key.wrapping_add(SEED) ^ om_hash;
    z = z.wrapping_add(GOLDEN);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

fn hash_for_family(family: HashFamily, key: u64, om: u64) -> u64 {
    match family {
        HashFamily::Additive => hash_additive(key, om),
        HashFamily::XorRemixed => hash_xor_remixed(key, om),
    }
}

fn num_slots(n: usize, gamma: f64) -> usize {
    let expanded = (n as f64 * gamma).max(1.0) as usize;
    ((expanded + CACHE_LINE_BITS - 1) / CACHE_LINE_BITS) * CACHE_LINE_BITS
}

fn count_unique(keys: &[u64], num_slots: usize, family: HashFamily, om: u64, counts: &mut [u8]) -> usize {
    counts[..num_slots].fill(0);
    for &key in keys {
        let idx = fastrange(hash_for_family(family, key, om), num_slots);
        counts[idx] = counts[idx].saturating_add(1);
    }
    counts[..num_slots].iter().filter(|&&c| c == 1).count()
}

fn lag1_correlation(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return f64::NAN;
    }
    let xs = &values[..values.len() - 1];
    let ys = &values[1..];
    let mean_x = xs.iter().sum::<f64>() / xs.len() as f64;
    let mean_y = ys.iter().sum::<f64>() / ys.len() as f64;
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let dx = x - mean_x;
        let dy = y - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    if var_x == 0.0 || var_y == 0.0 {
        f64::NAN
    } else {
        cov / (var_x.sqrt() * var_y.sqrt())
    }
}

fn main() {
    let cfg = parse_args();
    let keys = make_keys(cfg.n);
    let num_slots = num_slots(cfg.n, cfg.gamma);
    let load = cfg.n as f64 / num_slots as f64;
    let expected_unique = cfg.n as f64 * (-load).exp();
    let expected_ratio = expected_unique / cfg.n as f64;
    let mut counts = vec![0u8; num_slots];

    println!(
        "n={},gamma={:.6},num_slots={},load={:.6},expected_unique={:.3},expected_ratio={:.6}",
        cfg.n, cfg.gamma, num_slots, load, expected_unique, expected_ratio
    );

    for family in [HashFamily::Additive, HashFamily::XorRemixed] {
        let mut uniques = Vec::with_capacity(cfg.attempts);
        let mut first_accept = None;
        let mut longest_reject_streak = 0usize;
        let mut current_reject_streak = 0usize;

        for om in 0..cfg.attempts as u64 {
            let unique_count = count_unique(&keys, num_slots, family, om, &mut counts);
            let accept = unique_count as f64 >= expected_unique;
            if accept {
                if first_accept.is_none() {
                    first_accept = Some(om);
                }
                current_reject_streak = 0;
            } else {
                current_reject_streak += 1;
                longest_reject_streak = longest_reject_streak.max(current_reject_streak);
            }
            uniques.push(unique_count as f64);

            if om < cfg.print_attempts as u64 {
                println!(
                    "family={},om={},unique_count={},unique_ratio={:.6},accepted={}",
                    family.name(),
                    om,
                    unique_count,
                    unique_count as f64 / cfg.n as f64,
                    accept
                );
            }
        }

        let mean_unique = uniques.iter().sum::<f64>() / uniques.len() as f64;
        let min_unique = uniques.iter().copied().fold(f64::INFINITY, f64::min);
        let max_unique = uniques.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lag1 = lag1_correlation(&uniques);
        println!(
            "summary,family={},attempts={},first_accept={},longest_reject_streak={},mean_unique={:.3},min_unique={:.3},max_unique={:.3},lag1_corr={:.6}",
            family.name(),
            cfg.attempts,
            first_accept
                .map(|v| v.to_string())
                .unwrap_or_else(|| "none".to_string()),
            longest_reject_streak,
            mean_unique,
            min_unique,
            max_unique,
            lag1
        );
    }
}
