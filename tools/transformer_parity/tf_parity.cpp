// Parity and timing harness for the Compression Transformer numeric decisions.
//
// Build twice, once against a reference libopenzl.a and once against the
// candidate, then compare the dumps:
//   tf_parity dump <list> <out.bin>     feature + decision records per input
//   tf_parity cmp <a.bin> <b.bin>       byte-compare records, report counts
//   tf_parity time <list> <reps> [w]    ns/byte of decide8/16_fast/32_fast
// <list> holds one raw file path per line (u16 frames). Every file is cut
// into 1..64 KiB slices and fed as 1-, 2- and 4-byte numeric streams, plus
// derived byte planes, bit splits and deltas. Synthetic edge cases follow.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "openzl/compress/selectors/transformer/generic_numeric_decide.h"
#include "openzl/compress/selectors/transformer/numeric_extract.h"

namespace {

struct Scratch {
    std::vector<uint32_t> values;
    std::array<uint32_t, TRS_NUMERIC_MATCH4_TABLE_ENTRIES> match4{};
    std::array<TRS_NumericKmvEntry, TRS_NUMERIC_KMV_K> kmv{};
    std::array<uint64_t, TRS_NUMERIC_CARDINALITY_SCRATCH_WORDS> bitmap{};
    std::array<uint64_t, TRS_NUMERIC_SORTED_GAP_BUFFER_ENTRIES> sortedGap{};
    TRS_NumericExtractWorkspace ws(size_t n)
    {
        if (values.size() < n)
            values.resize(n);
        return { values.data(),    values.size(), match4.data(),
                 match4.size(),    kmv.data(),    kmv.size(),
                 bitmap.data(),    bitmap.size(), sortedGap.data(),
                 sortedGap.size() };
    }
};

struct Input {
    int width;
    std::vector<uint8_t> bytes; // 8-byte aligned storage via copy below
};

std::string readFile(const std::string& p)
{
    std::ifstream f(p, std::ios::binary);
    std::stringstream s;
    s << f.rdbuf();
    return s.str();
}

uint64_t rng(uint64_t& s)
{
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
}

void addSlices(
        std::vector<Input>& out,
        const std::vector<uint8_t>& src,
        int width,
        uint64_t& seed)
{
    size_t pos = 0;
    while (pos + (size_t)width <= src.size()) {
        size_t len = 1 + rng(seed) % 65536;
        len        = std::min(len, src.size() - pos);
        len -= len % (size_t)width;
        if (len == 0)
            break;
        out.push_back({ width,
                        std::vector<uint8_t>(
                                src.begin() + (long)pos,
                                src.begin() + (long)(pos + len)) });
        pos += len;
    }
}

void addDerived(std::vector<Input>& out, const std::string& raw, uint64_t& seed)
{
    std::vector<uint8_t> bytes(raw.begin(), raw.end());
    addSlices(out, bytes, 1, seed);
    addSlices(out, bytes, 2, seed);
    addSlices(out, bytes, 4, seed);
    size_t const n16 = bytes.size() / 2;
    std::vector<uint16_t> v(n16);
    memcpy(v.data(), bytes.data(), n16 * 2);
    std::vector<uint8_t> lo(n16), hi(n16), nib(n16), top(n16), dlo(n16);
    std::vector<uint8_t> d16(n16 * 2), x0(n16 / 3 * 2), u32(n16 / 2 * 4);
    for (size_t i = 0; i < n16; ++i) {
        lo[i]  = (uint8_t)v[i];
        hi[i]  = (uint8_t)(v[i] >> 8);
        nib[i] = (uint8_t)(v[i] & 15);
        top[i] = (uint8_t)(v[i] >> 12);
        dlo[i] = (uint8_t)(v[i] - (i ? v[i - 1] : 0));
        uint16_t d = (uint16_t)(v[i] - (i ? v[i - 1] : 0));
        memcpy(&d16[i * 2], &d, 2);
    }
    for (size_t i = 0; i + 2 < n16 && i / 3 < n16 / 3; i += 3)
        memcpy(&x0[i / 3 * 2], &v[i], 2);
    for (size_t i = 0; i + 1 < n16; i += 2) {
        uint32_t w = (uint32_t)v[i] | ((uint32_t)(v[i + 1] & 0x3FF) << 16);
        memcpy(&u32[i / 2 * 4], &w, 4);
    }
    for (auto* p : { &lo, &hi, &nib, &top, &dlo })
        addSlices(out, *p, 1, seed);
    addSlices(out, d16, 2, seed);
    addSlices(out, x0, 2, seed);
    addSlices(out, u32, 4, seed);
}

void addSynthetic(std::vector<Input>& out)
{
    uint64_t s = 0x9E3779B97F4A7C15ull;
    std::vector<size_t> sizes;
    for (size_t n = 1; n <= 300; ++n)
        sizes.push_back(n);
    for (size_t n : { 511,   512,   513,   1023,  1024,  4095,  4096,
                      4097,  8191,  8192,  65535, 65536, 65537, 65543,
                      65544, 65545, 70000, 131072, 262144 })
        sizes.push_back(n);
    for (int w : { 1, 2, 4 }) {
        for (size_t n : sizes) {
            for (int shape = 0; shape < 9; ++shape) {
                std::vector<uint8_t> b(n * (size_t)w);
                uint64_t run = 0, val = 0;
                for (size_t i = 0; i < b.size(); ++i) {
                    uint64_t r = rng(s);
                    switch (shape) {
                        case 0: b[i] = 0; break;
                        case 1: b[i] = 0xFF; break;
                        case 2: b[i] = (uint8_t)r; break;
                        case 3: b[i] = (uint8_t)(r & 1); break;
                        case 4: b[i] = (uint8_t)(r % 3 == 0 ? 0x80 : 0x7F); break;
                        case 5:
                            if (run == 0) {
                                val = r;
                                run = 1 + (r >> 40) % 40;
                            }
                            b[i] = (uint8_t)val;
                            --run;
                            break;
                        case 6: b[i] = (uint8_t)(i * 7); break;
                        case 7: b[i] = (uint8_t)((r & 0xF0) | 0x8); break;
                        default: b[i] = (uint8_t)(i % 251 == 0 ? r : 0); break;
                    }
                }
                out.push_back({ w, std::move(b) });
            }
        }
    }
}

std::vector<Input> buildCorpus(const char* list)
{
    std::vector<Input> out;
    std::ifstream L(list);
    std::string p;
    uint64_t seed = 12345;
    while (std::getline(L, p))
        if (!p.empty())
            addDerived(out, readFile(p), seed);
    addSynthetic(out);
    return out;
}

void put(std::string& rec, const void* p, size_t n)
{
    rec.append((const char*)p, n);
}

void putCore(std::string& rec, const TRS_GenericNumericDecisionCore& c)
{
    put(rec, &c.best_op, sizeof(c.best_op));
    put(rec, &c.best_index, sizeof(c.best_index));
    put(rec, &c.score_count, sizeof(c.score_count));
    put(rec, c.scores, sizeof(float) * (size_t)c.score_count);
    put(rec, c.raw_scores, sizeof(float) * (size_t)c.score_count);
    put(rec, c.logits, sizeof(float) * (size_t)c.score_count);
}

// Fields consumed by the num16/num32 fast decisions (others stay zero).
void putFast(std::string& rec, const TRS_NumericFeatures& f)
{
    put(rec, &f, sizeof(f));
}

const char* const kKinds[] = { "features", "decide8", "decide16",
                               "decide16_fast", "decide32", "decide32_fast" };

std::string record(const Input& in, Scratch& sc, uint32_t* kindMask)
{
    // Aligned copy: num32 paths require natural alignment.
    std::vector<uint64_t> aligned((in.bytes.size() + 7) / 8 + 1);
    memcpy(aligned.data(), in.bytes.data(), in.bytes.size());
    const uint8_t* d = (const uint8_t*)aligned.data();
    size_t const n   = in.bytes.size();
    TRS_NumericExtractWorkspace ws = sc.ws(n / (size_t)in.width + 1);
    std::string rec;
    put(rec, &in.width, sizeof(in.width));
    put(rec, &n, sizeof(n));
    TRS_NumericFeatures f;
    memset(&f, 0, sizeof(f));
    int ok = TRS_numericFeatures_extract_from_bytes(
            &f, d, n, (size_t)in.width, &ws);
    put(rec, &ok, sizeof(ok));
    put(rec, &f, sizeof(f));
    *kindMask = 1;
    auto dec  = [&](int kind, const TRS_GenericNumericDecision& x) {
        uint32_t tag = (uint32_t)kind;
        put(rec, &tag, sizeof(tag));
        putFast(rec, x.features);
        putCore(rec, x.core);
        *kindMask |= 1u << kind;
    };
    if (in.width == 1) {
        dec(1, TRS_generic_numeric_decide8(d, n, &ws));
    } else if (in.width == 2) {
        dec(2, TRS_generic_numeric_decide16(d, n, &ws));
        dec(3, TRS_generic_numeric_decide16_fast(d, n, &ws));
    } else {
        dec(4, TRS_generic_numeric_decide32(d, n, &ws));
        dec(5, TRS_generic_numeric_decide32_fast(d, n));
    }
    return rec;
}

int dump(const char* list, const char* outPath)
{
    std::vector<Input> corpus = buildCorpus(list);
    Scratch sc;
    FILE* o = fopen(outPath, "wb");
    std::map<int, size_t> perWidth;
    for (const Input& in : corpus) {
        uint32_t mask;
        std::string rec = record(in, sc, &mask);
        uint32_t len    = (uint32_t)rec.size();
        fwrite(&len, 4, 1, o);
        fwrite(&mask, 4, 1, o);
        fwrite(rec.data(), 1, rec.size(), o);
        perWidth[in.width]++;
    }
    fclose(o);
    printf("inputs %zu (w1 %zu, w2 %zu, w4 %zu)\n",
           corpus.size(),
           perWidth[1],
           perWidth[2],
           perWidth[4]);
    return 0;
}

int cmp(const char* a, const char* b)
{
    std::string A = readFile(a), B = readFile(b);
    size_t pa = 0, pb = 0, n = 0, bad = 0;
    std::map<std::string, size_t> checked;
    while (pa < A.size() && pb < B.size()) {
        uint32_t la, lb, ma, mb;
        memcpy(&la, &A[pa], 4);
        memcpy(&ma, &A[pa + 4], 4);
        memcpy(&lb, &B[pb], 4);
        memcpy(&mb, &B[pb + 4], 4);
        bool same = la == lb && ma == mb
                && memcmp(&A[pa + 8], &B[pb + 8], la) == 0;
        for (int k = 0; k < 6; ++k)
            if (ma & (1u << k))
                checked[kKinds[k]]++;
        if (!same) {
            if (bad < 10)
                fprintf(stderr, "mismatch at record %zu\n", n);
            ++bad;
        }
        pa += 8 + la;
        pb += 8 + lb;
        ++n;
    }
    if (pa != A.size() || pb != B.size()) {
        fprintf(stderr, "record count differs\n");
        ++bad;
    }
    printf("records %zu mismatches %zu\n", n, bad);
    for (auto& [k, c] : checked)
        printf("  %s: %zu\n", k.c_str(), c);
    return bad != 0;
}

int timeIt(const char* list, int reps, int onlyWidth)
{
    std::vector<Input> corpus = buildCorpus(list);
    Scratch sc;
    std::map<int, std::pair<double, size_t>> best;
    int sink = 0;
    for (int w : { 1, 2, 4 }) {
        if (onlyWidth != 0 && w != onlyWidth)
            continue;
        double bestNs = 1e300;
        size_t bytes  = 0;
        for (int r = 0; r < reps; ++r) {
            bytes   = 0;
            auto t0 = std::chrono::steady_clock::now();
            for (const Input& in : corpus) {
                if (in.width != w)
                    continue;
                const uint8_t* d = in.bytes.data();
                size_t n         = in.bytes.size();
                if (w == 4 && ((uintptr_t)d & 3))
                    continue;
                TRS_NumericExtractWorkspace ws = sc.ws(n / (size_t)w + 1);
                if (w == 1)
                    sink += TRS_generic_numeric_decide8(d, n, &ws).core.best_op;
                else if (w == 2)
                    sink += TRS_generic_numeric_decide16_fast(d, n, &ws)
                                    .core.best_op;
                else
                    sink += TRS_generic_numeric_decide32_fast(d, n)
                                    .core.best_op;
                bytes += n;
            }
            double ns = std::chrono::duration<double, std::nano>(
                                std::chrono::steady_clock::now() - t0)
                                .count();
            bestNs = std::min(bestNs, ns);
        }
        printf("decide%d: %.3f ns/byte over %zu bytes (sink %d)\n",
               w * 8,
               bestNs / (double)bytes,
               bytes,
               sink);
    }
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc >= 4 && !strcmp(argv[1], "dump"))
        return dump(argv[2], argv[3]);
    if (argc >= 4 && !strcmp(argv[1], "cmp"))
        return cmp(argv[2], argv[3]);
    if (argc >= 4 && !strcmp(argv[1], "time"))
        return timeIt(argv[2], atoi(argv[3]), argc >= 5 ? atoi(argv[4]) : 0);
    fprintf(stderr, "usage: tf_parity dump|cmp|time ...\n");
    return 2;
}
