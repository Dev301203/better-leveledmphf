// Standalone BooPHF runner with the same key stream and timing protocol as the Rust runners.
//
// Usage:
//   bbhash_runner --n <nelem> --gamma <gamma> [--warmup <u32>] [--timed <u32>] [--lookup-timed <u32>] [--threads <int>]
//
// CSV to stdout:
//   impl,phase,n,gamma,threads,trial,ms
// Summary lines go to stderr so the CSV stays clean.

#include "BooPHF.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

static uint64_t rust_style_key(uint64_t i) {
    return i * UINT64_C(0x9e3779b97f4a7c15);
}

struct Config {
    size_t n = 0;
    double gamma = 0.0;
    int threads = 1;
    unsigned warmup = 2;
    unsigned timed = 7;
    unsigned lookup_timed = 5;
};

[[noreturn]] static void usage() {
    std::cerr << "usage: bbhash_runner --n <nelem> --gamma <gamma> [--warmup <u32>] [--timed <u32>] [--lookup-timed <u32>] [--threads <int>]\n";
    std::exit(2);
}

static bool match_flag(const std::string& arg, const char* flag) {
    return arg == flag;
}

template <typename T>
struct FlagParser;

template <>
struct FlagParser<size_t> {
    static size_t parse(const char* value) {
        return static_cast<size_t>(std::strtoull(value, nullptr, 10));
    }
};

template <>
struct FlagParser<unsigned> {
    static unsigned parse(const char* value) {
        return static_cast<unsigned>(std::strtoul(value, nullptr, 10));
    }
};

template <>
struct FlagParser<int> {
    static int parse(const char* value) {
        return std::atoi(value);
    }
};

template <>
struct FlagParser<double> {
    static double parse(const char* value) {
        return std::strtod(value, nullptr);
    }
};

template <typename T>
static T parse_flag(int argc, char** argv, const char* flag, bool required, T default_value) {
    for (int i = 1; i < argc; ++i) {
        if (match_flag(argv[i], flag)) {
            if (i + 1 >= argc) {
                usage();
            }
            return FlagParser<T>::parse(argv[i + 1]);
        }
    }
    if (required) {
        usage();
    }
    return default_value;
}

static Config parse_args(int argc, char** argv) {
    Config cfg;
    cfg.n = parse_flag<size_t>(argc, argv, "--n", true, 0);
    cfg.gamma = parse_flag<double>(argc, argv, "--gamma", true, 0.0);
    cfg.warmup = parse_flag<unsigned>(argc, argv, "--warmup", false, 2);
    cfg.timed = parse_flag<unsigned>(argc, argv, "--timed", false, 7);
    cfg.lookup_timed = parse_flag<unsigned>(argc, argv, "--lookup-timed", false, 5);
    cfg.threads = parse_flag<int>(argc, argv, "--threads", false, 1);
    if (cfg.threads < 1) {
        cfg.threads = 1;
    }
    return cfg;
}

static std::vector<uint64_t> make_keys(size_t n) {
    std::vector<uint64_t> keys;
    keys.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        keys.push_back(rust_style_key(static_cast<uint64_t>(i)));
    }
    return keys;
}

static void print_raw_rows(const char* phase, size_t n, double gamma, int threads, const std::vector<double>& samples) {
    for (size_t i = 0; i < samples.size(); ++i) {
        std::cout << "bbhash-cpp," << phase << ',' << n << ',' << std::fixed << std::setprecision(6)
                  << gamma << ',' << threads << ',' << i << ',' << samples[i] << "\n";
    }
}

static void print_summary(const char* phase, size_t n, double gamma, const std::vector<double>& samples) {
    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    const double min = sorted.empty() ? 0.0 : sorted.front();
    const double max = sorted.empty() ? 0.0 : sorted.back();
    const double median = sorted.empty()
        ? 0.0
        : (sorted.size() % 2 == 1
            ? sorted[sorted.size() / 2]
            : 0.5 * (sorted[sorted.size() / 2 - 1] + sorted[sorted.size() / 2]));
    const double mean = sorted.empty()
        ? 0.0
        : std::accumulate(sorted.begin(), sorted.end(), 0.0) / static_cast<double>(sorted.size());
    std::cerr << "summary,bbhash-cpp," << phase << ',' << n << ',' << std::fixed << std::setprecision(6)
              << gamma << ",median_ms=" << median << ",min_ms=" << min << ",max_ms=" << max
              << ",mean_ms=" << mean << "\n";
}

int main(int argc, char** argv) {
    const Config cfg = parse_args(argc, argv);
    const std::vector<uint64_t> keys = make_keys(cfg.n);

    std::cout << "impl,phase,n,gamma,threads,trial,ms\n";

    using clock = std::chrono::steady_clock;
    for (unsigned i = 0; i < cfg.warmup; ++i) {
        boomphf::mphf<uint64_t, boomphf::SingleHashFunctor<uint64_t>> mphf(
            cfg.n, keys, cfg.threads, cfg.gamma,
            false,
            false,
            0.0f);
        (void) mphf;
    }

    std::vector<double> build_samples;
    build_samples.reserve(cfg.timed);
    for (unsigned i = 0; i < cfg.timed; ++i) {
        auto t0 = clock::now();
        boomphf::mphf<uint64_t, boomphf::SingleHashFunctor<uint64_t>> mphf(
            cfg.n, keys, cfg.threads, cfg.gamma,
            false,
            false,
            0.0f);
        auto t1 = clock::now();
        build_samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        (void) mphf;
    }
    print_raw_rows("build", cfg.n, cfg.gamma, cfg.threads, build_samples);
    print_summary("build", cfg.n, cfg.gamma, build_samples);

    boomphf::mphf<uint64_t, boomphf::SingleHashFunctor<uint64_t>> lookup_mphf(
        cfg.n, keys, cfg.threads, cfg.gamma,
        false,
        false,
        0.0f);
    std::vector<double> lookup_samples;
    lookup_samples.reserve(cfg.lookup_timed);
    for (unsigned i = 0; i < cfg.lookup_timed; ++i) {
        volatile uint64_t sink = 0;
        auto t0 = clock::now();
        for (uint64_t key : keys) {
            sink ^= lookup_mphf.lookup(key);
        }
        auto t1 = clock::now();
        lookup_samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        if (sink == std::numeric_limits<uint64_t>::max()) {
            std::cerr << "black_box=" << sink << '\n';
        }
    }
    print_raw_rows("lookup", cfg.n, cfg.gamma, cfg.threads, lookup_samples);
    print_summary("lookup", cfg.n, cfg.gamma, lookup_samples);

    return 0;
}
