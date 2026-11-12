/*
 * Per-query timing of exact search and of shortlist verification for the AudioMNIST scalability figure (user, Nov 4:
 * every method timed on thalia, one query per pinned physical core, many queries at once). The cascade is
 * e2e_v2/ucr_usp.c in mode usp (UCR-Suite lower bounds + EAPrunedDTW), copied unchanged; differences:
 *   - envelopes by a streaming deque in O(L) (identical values), database envelopes in parallel (offline, reported);
 *   - queries in parallel (OpenMP, one thread per query, per-thread buffers); each query's wall time is measured by
 *     its own thread (CLOCK_MONOTONIC) together with its thread CPU time, so contention is visible;
 *   - exact mode, nested database sizes in one scan: the scan visits rows in order, so when it reaches row n the
 *     state (best-so-far, pruning decisions, elapsed time) is exactly that of an exact search over the first n rows:
 *     neighbours and elapsed time are snapshotted per requested size;
 *   - verify mode: order file int32 [nq, N] (a method's ranking of the whole database) and a list of shortlist
 *     sizes; each size is verified as its own search over the first m ranked candidates (as e2e2.py), timed alone;
 *   - one file per finished query in <out>.parts/ (restart skips them).
 * usage: ucr_time exact  db.npy q.npy nq wfrac k out sizes(3250,..,26000) threads
 *        ucr_time verify db.npy q.npy nq wfrac k out limits(520,..) threads order.bin
 * out_N<size>.nn.bin / .dist.bin / .json (exact) or out_m<limit>.* (verify): neighbours, distances, per-query
 * seconds (wall and thread CPU).
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <omp.h>
#include <sched.h>

#define INF 1e20

static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}
static double cpu_now(void) {
    struct timespec t; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}
/* minimal .npy loader for C-contiguous little-endian float32/float64 2-D arrays */
static double *load_npy(const char *fp, long *rows, long *cols) {
    FILE *f = fopen(fp, "rb");
    if (!f) { perror(fp); exit(1); }
    unsigned char magic[10];
    if (fread(magic, 1, 10, f) != 10) { fprintf(stderr, "bad npy %s\n", fp); exit(1); }
    long hlen;
    if (magic[6] == 1) hlen = magic[8] | (magic[9] << 8);
    else { unsigned char b[2]; if (fread(b, 1, 2, f) != 2) exit(1); hlen = magic[8] | (magic[9] << 8) | (b[0] << 16) | (b[1] << 24); }
    char *hdr = malloc(hlen + 1);
    if (fread(hdr, 1, hlen, f) != (size_t)hlen) exit(1);
    hdr[hlen] = 0;
    int is64 = strstr(hdr, "<f8") != NULL;
    if (!is64 && !strstr(hdr, "<f4")) { fprintf(stderr, "%s: need <f4/<f8: %s\n", fp, hdr); exit(1); }
    if (strstr(hdr, "'fortran_order': True")) { fprintf(stderr, "%s: fortran order\n", fp); exit(1); }
    char *s = strstr(hdr, "'shape': (");
    s += 10;
    *rows = strtol(s, &s, 10);
    while (*s == ',' || *s == ' ') s++;
    *cols = (*s == ')') ? 1 : strtol(s, &s, 10);
    long n = (*rows) * (*cols);
    double *out = malloc(sizeof(double) * n);
    if (is64) { if (fread(out, 8, n, f) != (size_t)n) exit(1); }
    else {
        float *tmp = malloc(sizeof(float) * n);
        if (fread(tmp, 4, n, f) != (size_t)n) { fprintf(stderr, "short read %s\n", fp); exit(1); }
        for (long i = 0; i < n; i++) out[i] = tmp[i];
        free(tmp);
    }
    free(hdr); fclose(f);
    return out;
}

/* min/max envelope under a Sakoe-Chiba radius r: l[i] = min t[i-r..i+r], u[i] = max (clipped), by monotone deques
 * (Lemire 2009), O(len); dq: int workspace of 2*len */
static void envelope(const double *t, int len, int r, double *l, double *u, int *dq) {
    int *mx = dq, *mn = dq + len, hx = 0, tx = 0, hn = 0, tn = 0;
    for (int j = 0; j < len + r; j++) {
        if (j < len) {
            while (tx > hx && t[mx[tx - 1]] <= t[j]) tx--;
            mx[tx++] = j;
            while (tn > hn && t[mn[tn - 1]] >= t[j]) tn--;
            mn[tn++] = j;
        }
        int i = j - r;  /* window of i is [i-r, i+r] = [j-2r, j] */
        if (i >= 0) {
            while (mx[hx] < i - r) hx++;
            while (mn[hn] < i - r) hn++;
            u[i] = t[mx[hx]]; l[i] = t[mn[hn]];
        }
    }
}

static inline double min2(double a, double b) { return a < b ? a : b; }
static inline double dist2(double a, double b) { return (a - b) * (a - b); }

/* LB_Kim hierarchy (UCR Suite), early abandoned at bsf */
static double lb_kim(const double *q, const double *c, int len, double bsf) {
    double lb = dist2(c[0], q[0]) + dist2(c[len - 1], q[len - 1]);
    if (lb >= bsf) return lb;
    double d = min2(dist2(c[1], q[0]), dist2(c[0], q[1]));
    d = min2(d, dist2(c[1], q[1]));
    lb += d; if (lb >= bsf) return lb;
    d = min2(dist2(c[len - 2], q[len - 1]), dist2(c[len - 1], q[len - 2]));
    d = min2(d, dist2(c[len - 2], q[len - 2]));
    lb += d; if (lb >= bsf) return lb;
    d = min2(dist2(c[0], q[2]), dist2(c[1], q[2]));
    d = min2(d, dist2(c[2], q[2])); d = min2(d, dist2(c[2], q[1])); d = min2(d, dist2(c[2], q[0]));
    lb += d; if (lb >= bsf) return lb;
    d = min2(dist2(c[len - 1], q[len - 3]), dist2(c[len - 2], q[len - 3]));
    d = min2(d, dist2(c[len - 3], q[len - 3])); d = min2(d, dist2(c[len - 3], q[len - 2])); d = min2(d, dist2(c[len - 3], q[len - 1]));
    lb += d;
    return lb;
}

/* LB_Keogh of candidate c against query envelope (EQ), query order `ord` */
static double lb_keogh_eq(const int *ord, const double *c, const double *uo, const double *lo,
                          double *cb, int len, double bsf) {
    double lb = 0;
    for (int i = 0; i < len && lb < bsf; i++) {
        double x = c[ord[i]], d = 0;
        if (x > uo[i]) d = dist2(x, uo[i]);
        else if (x < lo[i]) d = dist2(x, lo[i]);
        lb += d; cb[ord[i]] = d;
    }
    return lb;
}

/* LB_Keogh of the (reordered) query against candidate envelope (EC) */
static double lb_keogh_ec(const int *ord, const double *qo, const double *cu, const double *cl,
                          double *cb, int len, double bsf) {
    double lb = 0;
    for (int i = 0; i < len && lb < bsf; i++) {
        double x = qo[i], d = 0;
        if (x > cu[ord[i]]) d = dist2(x, cu[ord[i]]);
        else if (x < cl[ord[i]]) d = dist2(x, cl[ord[i]]);
        lb += d; cb[ord[i]] = d;
    }
    return lb;
}

/* EAPrunedDTW with the cumulative-bound cutoff of UCR-MON; buffers hold 2 * (m + 1) doubles */
static double dtw_eap(const double *lines, const double *cols, const double *cb, int m, int w, double bsf, double *buf) {
    const long nb = m;
    for (long k = 0; k < 2 * (nb + 1); k++) buf[k] = INF;
    long c = 1, p = nb + 2;
    buf[c - 1] = 0;
    long next_start = 0, pruning_point = 0;
    double UB = (w + 1 < m) ? bsf - cb[w + 1] : bsf;
    for (long i = 0; i < nb; i++) {
        long t = c; c = p; p = t;
        const double li = lines[i];
        const long jStop = (i + w + 1 < nb) ? i + w + 1 : nb;
        long jStart = i > w ? i - w : 0;
        if (next_start > jStart) jStart = next_start;
        long next_pp = jStart, j = jStart;
        next_start = jStart;
        buf[c + j - 1] = INF;
        double cost = INF;
        for (; j == next_start && j < pruning_point; ++j) {
            const double d = dist2(li, cols[j]);
            cost = min2(buf[p + j - 1], buf[p + j]) + d;
            buf[c + j] = cost;
            if (cost <= UB) next_pp = j + 1; else ++next_start;
        }
        for (; j < pruning_point; ++j) {
            const double d = dist2(li, cols[j]);
            cost = min2(cost, min2(buf[p + j - 1], buf[p + j])) + d;
            buf[c + j] = cost;
            if (cost <= UB) next_pp = j + 1;
        }
        if (j < jStop) {
            const double d = dist2(li, cols[j]);
            if (j == next_start) {
                cost = buf[p + j - 1] + d;
                buf[c + j] = cost;
                if (cost <= UB) next_pp = j + 1; else return INF;
            } else {
                cost = min2(cost, buf[p + j - 1]) + d;
                buf[c + j] = cost;
                if (cost <= UB) next_pp = j + 1;
            }
            ++j;
        } else if (j == next_start) return INF;
        for (; j == next_pp && j < jStop; ++j) {
            cost = cost + dist2(li, cols[j]);
            buf[c + j] = cost;
            if (cost <= UB) ++next_pp;
        }
        pruning_point = next_pp;
        UB = (i + w + 1 < nb) ? bsf - cb[i + w + 1] : bsf;
    }
    return pruning_point != nb ? INF : buf[c + nb - 1];
}

/* LB_Webb of query a against candidate b (natural order), early abandoned at ub; ua/la: envelopes of a,
 * lua: lower envelope of ua, ula: upper envelope of la; same for b */
static double lb_webb(const double *a, const double *ua, const double *la, const double *lua, const double *ula,
                      const double *b, const double *ub_, const double *lb_, const double *lub, const double *ulb,
                      int len, int w, double ub) {
    double lb = 0;
    long fa = w, fb = w;  /* freeCountAbove, freeCountBelow */
    for (long i = 0; i < len && lb <= ub; ++i) {
        const double ai = a[i], ubi = ub_[i];
        if (ai > ubi) { lb += dist2(ai, ubi); fb = ubi >= ula[i] ? fb + 1 : 0; }
        else {
            const double lbi = lb_[i];
            if (ai < lbi) { lb += dist2(ai, lbi); fa = lbi <= lua[i] ? fa + 1 : 0; }
            else { fa++; fb++; }
        }
        if (i >= w) {
            const long j = i - w;
            const double bj = b[j], uaj = ua[j];
            if (bj > uaj) {
                if (fa > 2 * w) lb += dist2(bj, uaj);
                else { const double x = ulb[j]; if (bj > x && x >= uaj) lb += dist2(bj, uaj) - dist2(x, uaj); }
            } else {
                const double laj = la[j];
                if (bj < laj) {
                    if (fb > 2 * w) lb += dist2(bj, laj);
                    else { const double x = lub[j]; if (bj < x && x <= laj) lb += dist2(bj, laj) - dist2(x, laj); }
                }
            }
        }
    }
    for (long j = len - w; j < len && lb <= ub; j++) {
        const double bj = b[j], uaj = ua[j];
        if (bj > uaj) {
            if (j >= len - fa + w) lb += dist2(bj, uaj);
            else { const double x = ulb[j]; if (bj > x && x >= uaj) lb += dist2(bj, uaj) - dist2(x, uaj); }
        } else {
            const double laj = la[j];
            if (bj < laj) {
                if (j >= len - fb + w) lb += dist2(bj, laj);
                else { const double x = lub[j]; if (bj < x && x <= laj) lb += dist2(bj, laj) - dist2(x, laj); }
            }
        }
    }
    return lb;
}

/* k-best bookkeeping (k small): sorted ascending arrays */
static void kbest_insert(double *bd, int *bi, int k, double d, int id) {
    if (d >= bd[k - 1]) return;
    int p = k - 1;
    while (p > 0 && bd[p - 1] > d) { bd[p] = bd[p - 1]; bi[p] = bi[p - 1]; p--; }
    bd[p] = d; bi[p] = id;
}

static int cmp_absq_r(const void *a, const void *b, void *q) {
    double x = fabs(((const double *)q)[*(const int *)a]), y = fabs(((const double *)q)[*(const int *)b]);
    return (x < y) - (x > y);
}

int main(int argc, char **argv) {
    if (argc < 10 || (strcmp(argv[1], "exact") && strcmp(argv[1], "verify")) || (!strcmp(argv[1], "verify") && argc < 11)) {
        fprintf(stderr, "usage: %s exact|verify db.npy q.npy nq wfrac k out sizes threads [order.bin]\n", argv[0]); return 1;
    }
    int verify = !strcmp(argv[1], "verify");
    const char *mode = getenv("CONTROL_LB") ? getenv("CONTROL_LB") : "usp";
    int lbk = !strcmp(mode,"lbk"), lbw = !strcmp(mode,"lbw");
    if (!lbk && !lbw && strcmp(mode,"usp")) return 1;
    long N, L, NQ, LQ;
    double t0 = now();
    double *db = load_npy(argv[2], &N, &L);
    double *Q = load_npy(argv[3], &NQ, &LQ);
    if (LQ != L) { fprintf(stderr, "length mismatch\n"); return 1; }
    int nq = atoi(argv[4]); if (nq > NQ) nq = NQ;
    double wfrac = atof(argv[5]);
    int K = atoi(argv[6]);
    long sizes[64]; int ns = 0;
    for (char *t = strtok(argv[8], ","); t && ns < 64; t = strtok(NULL, ",")) sizes[ns++] = atol(t);
    for (int i = 1; i < ns; i++) if (sizes[i] <= sizes[i - 1]) { fprintf(stderr, "sizes must increase\n"); return 1; }
    if (ns == 0 || sizes[ns - 1] > N) { fprintf(stderr, "bad sizes\n"); return 1; }
    long NSCAN = verify ? N : sizes[ns - 1];
    omp_set_num_threads(atoi(argv[9]));
    int r = (int)floor(wfrac * L) - 1;  /* dtaidistance window=int(wfrac L): max shift W-1, as ucr_usp */
    int *order = NULL;
    if (verify) {
        FILE *fo = fopen(argv[10], "rb"); if (!fo) { perror(argv[10]); return 1; }
        order = malloc(sizeof(int) * (long)nq * N);
        if (fread(order, sizeof(int), (long)nq * N, fo) != (size_t)((long)nq * N)) { fprintf(stderr, "short order file\n"); return 1; }
        fclose(fo);
    }
    char parts[4096], fp[4096];
    snprintf(parts, sizeof parts, "%s.parts", argv[7]); mkdir(parts, 0775);
    double t_load = now() - t0;
    t0 = now();
    double *DU = malloc(sizeof(double) * N * L), *DL = malloc(sizeof(double) * N * L);
    if (!DU || !DL) { fprintf(stderr, "out of memory\n"); return 1; }
    #pragma omp parallel
    {
        int *dq = malloc(sizeof(int) * 2 * L);
        #pragma omp for schedule(static)
        for (long i = 0; i < N; i++) envelope(db + i * L, L, r, DL + i * L, DU + i * L, dq);
        free(dq);
    }
    double *DUL=NULL, *DLU=NULL;
    if (lbw) {
        DUL=malloc(sizeof(double)*N*L); DLU=malloc(sizeof(double)*N*L);
        if (!DUL || !DLU) return 1;
        #pragma omp parallel
        {
            int *dq=malloc(sizeof(int)*2*L); double *tmp=malloc(sizeof(double)*L);
            #pragma omp for schedule(static)
            for (long c=0;c<N;c++) {
                envelope(DL+c*L,L,r,tmp,DUL+c*L,dq);
                envelope(DU+c*L,L,r,DLU+c*L,tmp,dq);
            }
            free(dq);free(tmp);
        }
    }
    double t_env = now() - t0;
    fprintf(stderr, "loaded N=%ld L=%ld r=%d in %.0fs, envelopes %.0fs\n", N, L, r, t_load, t_env);

    int *NN = malloc(sizeof(int) * ns * nq * K); double *DD = malloc(sizeof(double) * ns * nq * K);
    double *TW = malloc(sizeof(double) * ns * nq), *TC = malloc(sizeof(double) * ns * nq), *FD = malloc(sizeof(double) * ns * nq);
    #pragma omp parallel
    {
        int *ord = malloc(sizeof(int) * L), *dq = malloc(sizeof(int) * 2 * L);
        double *qo = malloc(sizeof(double) * L), *uo = malloc(sizeof(double) * L), *lo = malloc(sizeof(double) * L);
        double *qu = malloc(sizeof(double) * L), *ql = malloc(sizeof(double) * L);
        double *cb = malloc(sizeof(double) * L), *cb1 = malloc(sizeof(double) * L), *cb2 = malloc(sizeof(double) * L);
        double *buf = malloc(sizeof(double) * 2 * (L + 1));
        double *qlu=malloc(sizeof(double)*L), *qul=malloc(sizeof(double)*L), *tmp=malloc(sizeof(double)*L);
        int bi[K]; double bd[K];
        #pragma omp for schedule(dynamic, 1)
        for (int qi = 0; qi < nq; qi++) {
            char qp[4200]; snprintf(qp, sizeof qp, "%s/q%d.bin", parts, qi);
            FILE *pf = fopen(qp, "rb");
            if (pf) {
                int ok = 1;
                for (int s = 0; s < ns; s++) {
                    long o = (long)s * nq + qi;
                    ok = ok && fread(NN + o * K, sizeof(int), K, pf) == (size_t)K && fread(DD + o * K, sizeof(double), K, pf) == (size_t)K
                            && fread(TW + o, sizeof(double), 1, pf) == 1 && fread(TC + o, sizeof(double), 1, pf) == 1
                            && fread(FD + o, sizeof(double), 1, pf) == 1;
                }
                fclose(pf);
                if (ok) continue;
            }
            const double *q = Q + (long)qi * L;
            for (int s = 0; s < (verify ? ns : 1); s++) {  /* verify: each shortlist size is its own timed search */
                long limit = verify ? sizes[s] : NSCAN, n_dtw = 0;
                const int *visit = verify ? order + (long)qi * N : NULL;
                double tw = now(), tc = cpu_now();
                envelope(q, L, r, ql, qu, dq);
                if(lbw) { envelope(ql,L,r,tmp,qul,dq); envelope(qu,L,r,qlu,tmp,dq); }
                for (int i = 0; i < L; i++) ord[i] = i;
                qsort_r(ord, L, sizeof(int), cmp_absq_r, (void *)q);
                for (int i = 0; i < L; i++) { qo[i] = q[ord[i]]; uo[i] = qu[ord[i]]; lo[i] = ql[ord[i]]; }
                for (int i = 0; i < K; i++) { bd[i] = INF; bi[i] = -1; }
                int si = 0;
                for (long ci = 0; ci < limit; ci++) {
                    if (!verify && ci == sizes[si]) {  /* exact: the prefix of size sizes[si] is complete */
                        long o = (long)si * nq + qi;
                        memcpy(NN + o * K, bi, sizeof(int) * K); memcpy(DD + o * K, bd, sizeof(double) * K);
                        TW[o] = now() - tw; TC[o] = cpu_now() - tc; FD[o] = (double)n_dtw / ci;
                        si++;
                    }
                    long c = visit ? visit[ci] : ci;
                    const double *x = db + c * L;
                    double bsf = bd[K - 1];
                    if(lbk) {
                        if(lb_keogh_eq(ord,x,uo,lo,cb1,L,bsf)>=bsf) continue;
                        cb[L-1]=cb1[L-1];
                        for(int i=L-2;i>=0;i--) cb[i]=cb[i+1]+cb1[i];
                    } else if(lbw) {
                        if(lb_webb(q,qu,ql,qlu,qul,x,DU+c*L,DL+c*L,DLU+c*L,DUL+c*L,L,r,bsf)>=bsf) continue;
                        memset(cb,0,sizeof(double)*L);
                    } else {
                    if (lb_kim(q, x, L, bsf) >= bsf) continue;
                    double lb1 = lb_keogh_eq(ord, x, uo, lo, cb1, L, bsf);
                    if (lb1 >= bsf) continue;
                    double lb2 = lb_keogh_ec(ord, qo, DU + c * L, DL + c * L, cb2, L, bsf);
                    if (lb2 >= bsf) continue;
                    const double *src = lb1 > lb2 ? cb1 : cb2;
                    cb[L - 1] = src[L - 1];
                    for (int i = L - 2; i >= 0; i--) cb[i] = cb[i + 1] + src[i];
                    }
                    double d = dtw_eap(q, x, cb, L, r, bsf, buf);
                    n_dtw++;
                    if (d < bsf) kbest_insert(bd, bi, K, d, (int)c);
                }
                long o = (long)(verify ? s : ns - 1) * nq + qi;
                memcpy(NN + o * K, bi, sizeof(int) * K); memcpy(DD + o * K, bd, sizeof(double) * K);
                TW[o] = now() - tw; TC[o] = cpu_now() - tc; FD[o] = (double)n_dtw / limit;
            }
            char tmp[4300]; snprintf(tmp, sizeof tmp, "%s.tmp", qp);
            pf = fopen(tmp, "wb");
            for (int s = 0; s < ns; s++) {
                long o = (long)s * nq + qi;
                fwrite(NN + o * K, sizeof(int), K, pf); fwrite(DD + o * K, sizeof(double), K, pf);
                fwrite(TW + o, sizeof(double), 1, pf); fwrite(TC + o, sizeof(double), 1, pf); fwrite(FD + o, sizeof(double), 1, pf);
            }
            fclose(pf); rename(tmp, qp);
            fprintf(stderr, "query %d on cpu %d: %.1fs wall %.1fs cpu (last size)\n", qi, sched_getcpu(),
                    TW[(long)(ns - 1) * nq + qi], TC[(long)(ns - 1) * nq + qi]);
        }
        free(ord); free(dq); free(qo); free(uo); free(lo); free(qu); free(ql); free(cb); free(cb1); free(cb2); free(buf); free(qlu); free(qul); free(tmp);
    }
    for (int s = 0; s < ns; s++) {
        long o = (long)s * nq;
        snprintf(fp, sizeof fp, "%s_%s%ld.nn.bin", argv[7], verify ? "m" : "N", sizes[s]); FILE *f = fopen(fp, "wb");
        fwrite(NN + o * K, sizeof(int), (long)nq * K, f); fclose(f);
        snprintf(fp, sizeof fp, "%s_%s%ld.dist.bin", argv[7], verify ? "m" : "N", sizes[s]); f = fopen(fp, "wb");
        fwrite(DD + o * K, sizeof(double), (long)nq * K, f); fclose(f);
        snprintf(fp, sizeof fp, "%s_%s%ld.json", argv[7], verify ? "m" : "N", sizes[s]); f = fopen(fp, "w");
        fprintf(f, "{\"mode\": \"%s\", \"cascade\": \"usp\", \"N\": %ld, \"L\": %ld, \"r\": %d, \"wfrac\": %g, \"k\": %d, \"size\": %ld,"
                   " \"n_queries\": %d, \"threads\": %d, \"db_envelope_s\": %.3f, \"seconds\": [", argv[1], verify ? N : sizes[s], L, r, wfrac, K,
                sizes[s], nq, atoi(argv[9]), t_env);
        for (int qi = 0; qi < nq; qi++) fprintf(f, "%s%.6f", qi ? ", " : "", TW[o + qi]);
        fprintf(f, "], \"cpu_seconds\": [");
        for (int qi = 0; qi < nq; qi++) fprintf(f, "%s%.6f", qi ? ", " : "", TC[o + qi]);
        fprintf(f, "], \"dtw_fraction\": [");
        for (int qi = 0; qi < nq; qi++) fprintf(f, "%s%.6f", qi ? ", " : "", FD[o + qi]);
        fprintf(f, "]}\n"); fclose(f);
    }
    printf("TIMING_DONE %s %d queries\n", argv[1], nq);
    return 0;
}
