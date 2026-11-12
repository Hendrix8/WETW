// MESSI-DTW (DaiSy, unmodified library) on one collection: one thread, exact k-NN, same 20 timed queries as Fig. 5.
// Writes JSON with build and per-query seconds and the returned ids. Inputs are raw float32 (N*L) files.
// usage: messi_run db.f32 q.f32 N L NQ r k out_prefix threads
#include "../lib/algos/Messi.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
using clk = std::chrono::steady_clock;
static double since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }
static std::vector<float> rd(const char *p, size_t n) {
    std::vector<float> v(n); FILE *f = fopen(p, "rb"); if (!f) { perror(p); exit(1); }
    if (fread(v.data(), 4, n, f) != n) { fprintf(stderr, "short read %s\n", p); exit(1); } fclose(f); return v;
}
int main(int c, char **a) {
    if (c < 10) { fprintf(stderr, "usage\n"); return 1; }
    size_t N = atol(a[3]), L = atol(a[4]), NQ = atol(a[5]); int r = atoi(a[6]); daisy::idx_t k = atol(a[7]); int th = atoi(a[9]);
    auto db = rd(a[1], N * L); auto q = rd(a[2], NQ * L);
    daisy::Messi m(daisy::DistanceType::DTW);
    m.setNumThreads(th); m.setWarpingWindow(r);
    auto t0 = clk::now(); m.buildIndex(db.data(), N, L); double tb = since(t0);
    std::vector<daisy::idx_t> I(NQ * k); std::vector<float> D(NQ * k); std::vector<double> sec(NQ);
    for (size_t i = 0; i < NQ; i++) {  // one query per call: per-query time
        t0 = clk::now(); m.searchIndex(q.data() + i * L, 1, k, I.data() + i * k, D.data() + i * k); sec[i] = since(t0);
        fprintf(stderr, "query %zu %.3f s\n", i, sec[i]);
    }
    char p[4096]; snprintf(p, sizeof p, "%s.json", a[8]); FILE *o = fopen(p, "w");
    fprintf(o, "{\"N\": %zu, \"L\": %zu, \"r\": %d, \"k\": %llu, \"threads\": %d, \"build_s\": %.3f, \"seconds\": [", N, L, r, (unsigned long long)k, th, tb);
    for (size_t i = 0; i < NQ; i++) fprintf(o, "%s%.6f", i ? ", " : "", sec[i]);
    fprintf(o, "], \"ids\": ["); for (size_t i = 0; i < NQ * k; i++) fprintf(o, "%s%llu", i ? ", " : "", (unsigned long long)I[i]);
    fprintf(o, "], \"dist\": ["); for (size_t i = 0; i < NQ * k; i++) fprintf(o, "%s%.8g", i ? ", " : "", D[i]);
    fprintf(o, "]}\n"); fclose(o); return 0;
}
