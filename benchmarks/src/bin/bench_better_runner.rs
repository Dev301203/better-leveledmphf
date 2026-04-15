use better_mphf::{BuildOptions, FastRangeMode, LeveledMphf};
use mphf_benchmarks::{
    make_keys_with_mode, mean, median_min_max, sample_after_warmup_raw, seed_offset_from_index,
    KeyMode,
};
use std::env;
use std::hint::black_box;
use std::process;
use std::time::Instant;

const DEFAULT_AUTO_GAMMA_BASE: f64 = 1.5;
const DEFAULT_KEY_MODE: &str = "multiplicative";
const DEFAULT_FASTRANGE_MODE: &str = "mul64";

struct Config {
    n: usize,
    gamma: Option<f64>,
    auto_gamma: bool,
    gamma_base: f64,
    warmup: u32,
    timed: u32,
    lookup_timed: u32,
    seed: u64,
    offset: u64,
    seed_index: u64,
    key_mode: KeyMode,
    fastrange_mode: FastRangeMode,
}

fn usage() -> ! {
    eprintln!(
        "usage: bench_better_runner --n <usize> (--gamma <f64> | --auto-gamma [--gamma-base <f64>]) [--seed-index <u64>] [--seed <u64>] [--offset <u64>] [--key-mode <multiplicative|sequential|splitmix-random|clustered|high-bit-heavy>] [--fastrange-mode <low32|high32|mul64>] [--warmup <u32>] [--timed <u32>] [--lookup-timed <u32>]"
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

fn parse_opt_flag<T: std::str::FromStr>(args: &[String], flag: &str) -> Option<T> {
    args.iter().position(|a| a == flag).map(|i| {
        args.get(i + 1)
            .unwrap_or_else(|| usage())
            .parse::<T>()
            .unwrap_or_else(|_| usage())
    })
}

fn parse_fastrange_mode(s: &str) -> FastRangeMode {
    match s {
        "low32" => FastRangeMode::Low32,
        "high32" => FastRangeMode::High32,
        "mul64" | "full64" => FastRangeMode::Mul64,
        _ => usage(),
    }
}

fn fastrange_mode_label(mode: FastRangeMode) -> &'static str {
    match mode {
        FastRangeMode::Low32 => "low32",
        FastRangeMode::High32 => "high32",
        FastRangeMode::Mul64 => "mul64",
    }
}

fn parse_args() -> Config {
    let args: Vec<String> = env::args().collect();
    let gamma = args
        .iter()
        .position(|a| a == "--gamma")
        .map(|_| parse_flag(&args, "--gamma", None));
    let auto_gamma = args.iter().any(|a| a == "--auto-gamma");
    if auto_gamma == gamma.is_some() {
        usage();
    }

    let seed_index: u64 = parse_flag(&args, "--seed-index", Some(0));
    let (default_seed, default_offset) = seed_offset_from_index(seed_index);
    let seed = parse_opt_flag(&args, "--seed").unwrap_or(default_seed);
    let offset = parse_opt_flag(&args, "--offset").unwrap_or(default_offset);

    let key_mode = parse_flag(&args, "--key-mode", Some(DEFAULT_KEY_MODE.to_string()))
        .parse::<KeyMode>()
        .unwrap_or_else(|_| usage());
    let fastrange_mode = parse_fastrange_mode(&parse_flag(
        &args,
        "--fastrange-mode",
        Some(DEFAULT_FASTRANGE_MODE.to_string()),
    ));

    Config {
        n: parse_flag(&args, "--n", None),
        gamma,
        auto_gamma,
        gamma_base: parse_flag(&args, "--gamma-base", Some(DEFAULT_AUTO_GAMMA_BASE)),
        warmup: parse_flag(&args, "--warmup", Some(2)),
        timed: parse_flag(&args, "--timed", Some(7)),
        lookup_timed: parse_flag(&args, "--lookup-timed", Some(5)),
        seed,
        offset,
        seed_index,
        key_mode,
        fastrange_mode,
    }
}

fn build_options(cfg: &Config) -> BuildOptions {
    let opts = BuildOptions::default()
        .par_threshold(usize::MAX)
        .fastrange_mode(cfg.fastrange_mode);
    if cfg.auto_gamma {
        opts.auto_tuned_gamma_base(cfg.gamma_base)
    } else {
        opts.fixed_gamma(cfg.gamma.unwrap())
    }
}

fn impl_name(cfg: &Config) -> &'static str {
    if cfg.auto_gamma {
        "better-mphf-auto"
    } else {
        "better-mphf"
    }
}

fn gamma_label(cfg: &Config) -> f64 {
    cfg.gamma.unwrap_or(0.0)
}

fn lookup_samples(keys: &[u64], cfg: &Config, timed: u32) -> Vec<f64> {
    let mphf = LeveledMphf::new_with_options(keys, cfg.seed, cfg.offset, build_options(cfg));
    let mut samples = Vec::with_capacity(timed as usize);
    for _ in 0..timed {
        let t0 = Instant::now();
        for &k in keys {
            black_box(mphf.lookup(k));
        }
        samples.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    samples
}

fn emit_raw_rows(impl_name: &str, phase: &str, n: usize, gamma: f64, cfg: &Config, samples: &[f64]) {
    let fastrange_mode = fastrange_mode_label(cfg.fastrange_mode);
    for (trial, ms) in samples.iter().enumerate() {
        println!(
            "{impl_name},{phase},{n},{gamma:.6},1,{trial},{ms:.6},{},{},{},{},{}",
            cfg.seed,
            cfg.offset,
            cfg.seed_index,
            cfg.key_mode.as_str(),
            fastrange_mode
        );
    }
}

fn emit_summary_row(impl_name: &str, phase: &str, n: usize, gamma: f64, cfg: &Config, samples: &[f64]) {
    let mut sorted = samples.to_vec();
    let (med, min, max) = median_min_max(&mut sorted);
    let avg = mean(samples);
    eprintln!(
        "summary,{impl_name},{phase},{n},{gamma:.6},median_ms={med:.6},min_ms={min:.6},max_ms={max:.6},mean_ms={avg:.6},seed={},offset={},seed_index={},key_mode={},fastrange_mode={}",
        cfg.seed,
        cfg.offset,
        cfg.seed_index,
        cfg.key_mode.as_str(),
        fastrange_mode_label(cfg.fastrange_mode)
    );
}

fn main() {
    let cfg = parse_args();
    let keys = make_keys_with_mode(cfg.n, cfg.key_mode, cfg.seed, cfg.offset);
    let impl_name = impl_name(&cfg);
    let gamma = gamma_label(&cfg);

    println!("impl,phase,n,gamma,threads,trial,ms,seed,offset,seed_index,key_mode,fastrange_mode");

    let build_samples = sample_after_warmup_raw(cfg.warmup, cfg.timed, || {
        let _ = LeveledMphf::new_with_options(&keys, cfg.seed, cfg.offset, build_options(&cfg));
    });
    emit_raw_rows(impl_name, "build", cfg.n, gamma, &cfg, &build_samples);
    emit_summary_row(impl_name, "build", cfg.n, gamma, &cfg, &build_samples);

    let lookup_samples = lookup_samples(&keys, &cfg, cfg.lookup_timed);
    emit_raw_rows(impl_name, "lookup", cfg.n, gamma, &cfg, &lookup_samples);
    emit_summary_row(impl_name, "lookup", cfg.n, gamma, &cfg, &lookup_samples);
}
