use boomphf::Mphf;
use mphf_benchmarks::{make_keys, mean, median_min_max, sample_after_warmup_raw};
use std::env;
use std::hint::black_box;
use std::process;
use std::time::Instant;

struct Config {
    n: usize,
    gamma: f64,
    warmup: u32,
    timed: u32,
    lookup_timed: u32,
}

fn usage() -> ! {
    eprintln!(
        "usage: bench_boom_runner --n <usize> --gamma <f64> [--warmup <u32>] [--timed <u32>] [--lookup-timed <u32>]"
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
        warmup: parse_flag(&args, "--warmup", Some(2)),
        timed: parse_flag(&args, "--timed", Some(7)),
        lookup_timed: parse_flag(&args, "--lookup-timed", Some(5)),
    }
}

fn lookup_samples(keys: &[u64], gamma: f64, timed: u32) -> Vec<f64> {
    let mphf = Mphf::new(gamma, keys);
    let mut samples = Vec::with_capacity(timed as usize);
    for _ in 0..timed {
        let t0 = Instant::now();
        for &k in keys {
            black_box(mphf.hash(&k));
        }
        samples.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    samples
}

fn emit_raw_rows(phase: &str, n: usize, gamma: f64, samples: &[f64]) {
    for (trial, ms) in samples.iter().enumerate() {
        println!("boomphf-rust,{phase},{n},{gamma:.6},1,{trial},{ms:.6}");
    }
}

fn emit_summary_row(phase: &str, n: usize, gamma: f64, samples: &[f64]) {
    let mut sorted = samples.to_vec();
    let (med, min, max) = median_min_max(&mut sorted);
    let avg = mean(samples);
    eprintln!(
        "summary,boomphf-rust,{phase},{n},{gamma:.6},median_ms={med:.6},min_ms={min:.6},max_ms={max:.6},mean_ms={avg:.6}"
    );
}

fn main() {
    let cfg = parse_args();
    let keys = make_keys(cfg.n);

    println!("impl,phase,n,gamma,threads,trial,ms");

    let build_samples = sample_after_warmup_raw(cfg.warmup, cfg.timed, || {
        let _ = Mphf::new(cfg.gamma, &keys);
    });
    emit_raw_rows("build", cfg.n, cfg.gamma, &build_samples);
    emit_summary_row("build", cfg.n, cfg.gamma, &build_samples);

    let lookup_samples = lookup_samples(&keys, cfg.gamma, cfg.lookup_timed);
    emit_raw_rows("lookup", cfg.n, cfg.gamma, &lookup_samples);
    emit_summary_row("lookup", cfg.n, cfg.gamma, &lookup_samples);
}
