// Standalone BooPHF runner with the same key-stream and timing protocol as the Rust runners.
//
// Usage:
//   bbhash_runner --n <nelem> --gamma <gamma>
//     [--seed-index <u64>] [--seed <u64>] [--offset <u64>]
//     [--key-mode <multiplicative|sequential|splitmix-random|clustered|high-bit-heavy>]
//     [--warmup <u32>] [--timed <u32>] [--lookup-timed <u32>] [--threads <int>]
//
// CSV to stdout:
//   impl,phase,n,gamma,threads,trial,ms,seed,offset,seed_index,key_mode,fastrange_mode
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
#include <utility>
#include <vector>

static constexpr uint64_t DEFAULT_SEED = UINT64_C(0xa0761d6478bd642f);
static constexpr uint64_t DEFAULT_OFFSET = UINT64_C(0xe7037ed1a0b428db);
static constexpr uint64_t GOLDEN = UINT64_C(0x9e3779b97f4a7c15);
static constexpr uint64_t CLUSTER_BITS = 6;
static constexpr uint64_t CLUSTER_SIZE = UINT64_C(1) << CLUSTER_BITS;

enum class KeyMode {
    Multiplicative,
    Sequential,
    SplitmixRandom,
    Clustered,
    HighBitHeavy,
};

static const char* key_mode_name(KeyMode mode) {
    switch (mode) {
        case KeyMode::Multiplicative:
            return "multiplicative";
        case KeyMode::Sequential:
            return "sequential";
        case KeyMode::SplitmixRandom:
            return "splitmix-random";
        case KeyMode::Clustered:
            return "clustered";
        case KeyMode::HighBitHeavy:
            return "high-bit-heavy";
    }
    return "multiplicative";
}

static KeyMode parse_key_mode(const std::string& raw) {
    if (raw == "multiplicative" || raw == "mult") {
        return KeyMode::Multiplicative;
    }
    if (raw == "sequential" || raw == "seq") {
        return KeyMode::Sequential;
    }
    if (raw == "splitmix-random" || raw == "splitmix_random" || raw == "splitmix" || raw == "random") {
        return KeyMode::SplitmixRandom;
    }
    if (raw == "clustered" || raw == "cluster") {
        return KeyMode::Clustered;
    }
    if (raw == "high-bit-heavy" || raw == "high_bit_heavy" || raw == "highbits") {
        return KeyMode::HighBitHeavy;
    }
    std::cerr << "unknown --key-mode '" << raw
              << "' (expected multiplicative|sequential|splitmix-random|clustered|high-bit-heavy)\n";
    std::exit(2);
}

static inline uint64_t rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> (64 - r));
}

static inline uint64_t splitmix64(uint64_t z) {
    z = z + GOLDEN;
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static std::pair<uint64_t, uint64_t> seed_offset_from_index(uint64_t seed_index) {
    const uint64_t seed_mix = seed_index * GOLDEN;
    const uint64_t seed = splitmix64(DEFAULT_SEED ^ seed_mix);
    const uint64_t offset = splitmix64(DEFAULT_OFFSET ^ rotl64(seed_mix, 17));
    return {seed, offset};
}

struct Config {
    size_t n = 0;
    double gamma = 0.0;
    int threads = 1;
    unsigned warmup = 2;
    unsigned timed = 7;
    unsigned lookup_timed = 5;
    uint64_t seed = DEFAULT_SEED;
    uint64_t offset = DEFAULT_OFFSET;
    uint64_t seed_index = 0;
    KeyMode key_mode = KeyMode::Multiplicative;
};

[[noreturn]] static void usage() {
    std::cerr
        << "usage: bbhash_runner --n <nelem> --gamma <gamma> [--seed-index <u64>] [--seed <u64>] [--offset <u64>] [--key-mode <multiplicative|sequential|splitmix-random|clustered|high-bit-heavy>] [--warmup <u32>] [--timed <u32>] [--lookup-timed <u32>] [--threads <int>]\n";
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

template <>
struct FlagParser<unsigned long long> {
    static unsigned long long parse(const char* value) {
        return std::strtoull(value, nullptr, 10);
    }
};

template <>
struct FlagParser<std::string> {
    static std::string parse(const char* value) {
        return std::string(value);
    }
};

template <typename T>
static T parse_flag(int argc, char** argv, const char* flag, bool required, const T& default_value) {
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
    cfg.seed_index = static_cast<uint64_t>(
        parse_flag<unsigned long long>(argc, argv, "--seed-index", false, 0ULL));

    const auto defaults = seed_offset_from_index(cfg.seed_index);
    cfg.seed = static_cast<uint64_t>(parse_flag<unsigned long long>(
        argc,
        argv,
        "--seed",
        false,
        static_cast<unsigned long long>(defaults.first)));
    cfg.offset = static_cast<uint64_t>(parse_flag<unsigned long long>(
        argc,
        argv,
        "--offset",
        false,
        static_cast<unsigned long long>(defaults.second)));

    const std::string key_mode_raw = parse_flag<std::string>(
        argc,
        argv,
        "--key-mode",
        false,
        std::string("multiplicative"));
    cfg.key_mode = parse_key_mode(key_mode_raw);

    if (cfg.threads < 1) {
        cfg.threads = 1;
    }
    return cfg;
}

static std::vector<uint64_t> make_keys(const Config& cfg) {
    std::vector<uint64_t> keys;
    keys.reserve(cfg.n);
    const uint64_t offset_odd = cfg.offset | UINT64_C(1);

    for (size_t i = 0; i < cfg.n; ++i) {
        const uint64_t idx = static_cast<uint64_t>(i);
        uint64_t key = 0;
        switch (cfg.key_mode) {
            case KeyMode::Multiplicative:
                key = idx * GOLDEN + cfg.offset;
                break;
            case KeyMode::Sequential:
                key = idx + cfg.offset;
                break;
            case KeyMode::SplitmixRandom:
                key = splitmix64(cfg.seed + idx * offset_odd);
                break;
            case KeyMode::Clustered: {
                const uint64_t cluster = idx >> CLUSTER_BITS;
                const uint64_t intra = idx & (CLUSTER_SIZE - 1);
                const uint64_t perm_cluster = cluster * GOLDEN + (cfg.seed ^ cfg.offset);
                key = (perm_cluster << CLUSTER_BITS) + intra;
                break;
            }
            case KeyMode::HighBitHeavy: {
                const uint64_t x = idx + cfg.offset;
                const uint64_t hi = rotl64(x, 32);
                const uint64_t lo = splitmix64(cfg.seed ^ x) & UINT64_C(0xff);
                key = (hi & ~UINT64_C(0xff)) | lo;
                break;
            }
        }
        keys.push_back(key);
    }

    return keys;
}

static void print_raw_rows(const char* phase, const Config& cfg, const std::vector<double>& samples) {
    for (size_t i = 0; i < samples.size(); ++i) {
        std::cout << "bbhash-cpp," << phase << ',' << cfg.n << ',' << std::fixed << std::setprecision(6)
                  << cfg.gamma << ',' << cfg.threads << ',' << i << ',' << samples[i] << ','
                  << cfg.seed << ',' << cfg.offset << ',' << cfg.seed_index << ',' << key_mode_name(cfg.key_mode)
                  << ",na\n";
    }
}

static void print_summary(const char* phase, const Config& cfg, const std::vector<double>& samples) {
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
    std::cerr << "summary,bbhash-cpp," << phase << ',' << cfg.n << ',' << std::fixed << std::setprecision(6)
              << cfg.gamma << ",median_ms=" << median << ",min_ms=" << min << ",max_ms=" << max
              << ",mean_ms=" << mean << ",seed=" << cfg.seed << ",offset=" << cfg.offset
              << ",seed_index=" << cfg.seed_index << ",key_mode=" << key_mode_name(cfg.key_mode) << "\n";
}

int main(int argc, char** argv) {
    const Config cfg = parse_args(argc, argv);
    const std::vector<uint64_t> keys = make_keys(cfg);

    std::cout
        << "impl,phase,n,gamma,threads,trial,ms,seed,offset,seed_index,key_mode,fastrange_mode\n";

    using clock = std::chrono::steady_clock;
    for (unsigned i = 0; i < cfg.warmup; ++i) {
        boomphf::mphf<uint64_t, boomphf::SingleHashFunctor<uint64_t>> mphf(
            cfg.n,
            keys,
            cfg.threads,
            cfg.gamma,
            false,
            false,
            0.0f);
        (void)mphf;
    }

    std::vector<double> build_samples;
    build_samples.reserve(cfg.timed);
    for (unsigned i = 0; i < cfg.timed; ++i) {
        auto t0 = clock::now();
        boomphf::mphf<uint64_t, boomphf::SingleHashFunctor<uint64_t>> mphf(
            cfg.n,
            keys,
            cfg.threads,
            cfg.gamma,
            false,
            false,
            0.0f);
        auto t1 = clock::now();
        build_samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        (void)mphf;
    }
    print_raw_rows("build", cfg, build_samples);
    print_summary("build", cfg, build_samples);

    boomphf::mphf<uint64_t, boomphf::SingleHashFunctor<uint64_t>> lookup_mphf(
        cfg.n,
        keys,
        cfg.threads,
        cfg.gamma,
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
    print_raw_rows("lookup", cfg, lookup_samples);
    print_summary("lookup", cfg, lookup_samples);

    return 0;
}
