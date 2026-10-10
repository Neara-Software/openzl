# points-tools

Helpers for trained points compressors (`.zlc`) and their benchmarks. Python scripts need `cbor2`.

- `add_narrow.py in.zlc out.zlc KEEP...` adds brute-force narrowing params to every ACE
  selector: try all successors on the first input of a file, then keep only the best ones
  (fast compress, encoder-only).
- `add_probe.py in.zlc out.zlc [--inner BYTES] [--top BYTES]` adds brute-force probe params:
  rank successors on a slice and fully try only the best (encoder-only).
- `merge_zlc.py a.zlc b.zlc out.zlc` merges the top-level brute-force candidates of two
  points-i32 compressors.
- `cdbench.cpp` benchmarks models and zstd levels on a list of chunk files. Each chunk is
  compressed and decompressed back to back; narrowing is reset when the chunk directory
  (= source file) changes. Prints per-directory CSV rows.

Build `cdbench` from a CMake build directory:

```
g++ -O2 -std=c++17 -I../include -I../src -Iinclude ../contrib/points-tools/cdbench.cpp \
    -o cdbench libopenzl.a zstd_build/lib/libzstd.a lz4_build/liblz4.a -lpthread
```

Usage: `cdbench <iters> <elt_bytes> <chunk_list> <model.zlc | zstd:LEVEL | tf:W[xS]>...`
(`CDB_TRY_CACHE=MiB` sets the tryGraph cache budget).
