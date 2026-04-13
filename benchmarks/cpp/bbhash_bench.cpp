// Minimal timing harness for BBHash (BooPHF). Same key stream as Rust benches:
// key_i = i * 0x9e3779b97f4a7c15 (unsigned wrap).
//
// Usage: bbhash_bench <nelem> <nthreads> <gamma>
// Prints CSV: nelem,nthreads,gamma,construction_ms

#include "BooPHF.h"
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

static uint64_t rust_style_key(uint64_t i) {
	return i * UINT64_C(0x9e3779b97f4a7c15);
}

int main(int argc, char **argv) {
	if (argc != 4) {
		std::cerr << "usage: bbhash_bench <nelem> <nthreads> <gamma>\n";
		return 1;
	}

	size_t nelem = std::strtoull(argv[1], nullptr, 10);
	int nthreads = std::atoi(argv[2]);
	double gamma = std::strtod(argv[3], nullptr);

	if (nelem == 0) {
		std::cout << "0," << nthreads << "," << gamma << ",0\n";
		return 0;
	}
	if (nthreads < 1)
		nthreads = 1;

	std::vector<uint64_t> keys;
	keys.reserve(nelem);
	for (size_t i = 0; i < nelem; ++i)
		keys.push_back(rust_style_key(static_cast<uint64_t>(i)));

	using clock = std::chrono::steady_clock;
	auto t0 = clock::now();
	boomphf::mphf<uint64_t, boomphf::SingleHashFunctor<uint64_t>> mphf(
		nelem, keys, nthreads, gamma,
		/*writeEach=*/false,
		/*progress=*/false,
		/*perc_elem_loaded=*/0.0f);
	auto t1 = clock::now();
	double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

	(void)mphf;
	std::cout << nelem << "," << nthreads << "," << gamma << "," << ms << "\n";
	return 0;
}
