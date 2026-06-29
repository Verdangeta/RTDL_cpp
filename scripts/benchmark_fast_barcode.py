#!/usr/bin/env python3
"""Benchmark barcode restoration after MSTs are available (naive vs RTDL_FAST_BARCODE).

The C++ core prints phase timings when RTDL_TIMING=1. The primary metric is
after_mst_median_sec: median time of compute_pairs_naive/fast on sorted MST edge
lists, excluding dense-matrix work, Prim/MST construction, and the TSP tail.
"""

import argparse
import ctypes
import math
import os
import pathlib
import statistics
import time

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
LIB_PATH = ROOT / "RTDLite" / "rtd_lite.so"


class BirthDeathEdges(ctypes.Structure):
    _fields_ = [
        ("a", ctypes.c_int64),
        ("b", ctypes.c_int64),
        ("c", ctypes.c_int64),
        ("d", ctypes.c_int64),
    ]


class RTDLiteResult(ctypes.Structure):
    _fields_ = [
        ("left_bars", ctypes.c_int64),
        ("right_bars", ctypes.c_int64),
        ("left_to_right", ctypes.POINTER(BirthDeathEdges)),
        ("right_to_left", ctypes.POINTER(BirthDeathEdges)),
    ]


def load_lib():
    lib = ctypes.cdll.LoadLibrary(str(LIB_PATH))
    matrix_ptr = ctypes.POINTER(ctypes.c_double)
    lib.rtd_lite_run_matrix.argtypes = [matrix_ptr, matrix_ptr, ctypes.c_int, ctypes.c_bool]
    lib.rtd_lite_run_matrix.restype = RTDLiteResult
    lib.rtd_lite_result_free.argtypes = [ctypes.POINTER(RTDLiteResult)]
    lib.rtd_lite_result_free.restype = None
    return lib


def tsp_case(n, rng):
    coords = rng.uniform(0.0, 1.0, size=(n, 2))
    diff = coords[:, None, :] - coords[None, :, :]
    r1 = np.sqrt(np.sum(diff * diff, axis=2))
    r2 = np.full((n, n), math.inf, dtype=np.float64)
    np.fill_diagonal(r2, 0.0)
    tour = list(rng.permutation(n))
    for i in range(n):
        u = int(tour[i])
        v = int(tour[(i + 1) % n])
        r2[u, v] = r2[v, u] = r1[u, v]
    return np.ascontiguousarray(r1), np.ascontiguousarray(r2)


def dense_case(n, rng):
    a = rng.uniform(0.0, 1.0, size=(n, 2))
    b = rng.uniform(0.0, 1.0, size=(n, 2))
    r1 = np.sqrt(np.sum((a[:, None, :] - a[None, :, :]) ** 2, axis=2))
    r2 = np.sqrt(np.sum((b[:, None, :] - b[None, :, :]) ** 2, axis=2))
    return np.ascontiguousarray(r1), np.ascontiguousarray(r2)


def parse_timing(stderr_text):
    timing_lines = [line for line in stderr_text.splitlines() if line.startswith("RTDL_TIMING")]
    if not timing_lines:
        raise RuntimeError(f"RTDL_TIMING line was not emitted. stderr={stderr_text!r}")

    fields = {}
    for part in timing_lines[-1].split(",")[1:]:
        key, value = part.split("=", 1)
        fields[key] = value

    if "after_mst_median_sec" not in fields and "pair_median_sec" in fields:
        fields["after_mst_median_sec"] = fields["pair_median_sec"]
    elif "pair_median_sec" not in fields and "after_mst_median_sec" in fields:
        fields["pair_median_sec"] = fields["after_mst_median_sec"]
    for key in ("setup_sec", "after_mst_median_sec", "pair_median_sec", "tsp_tail_sec", "total_inside_sec"):
        if key in fields:
            fields[key] = float(fields[key])
    for key in ("n", "fallback", "repeat"):
        fields[key] = int(fields[key])
    return fields


def capture_c_stderr(func):
    read_fd, write_fd = os.pipe()
    old_stderr = os.dup(2)
    try:
        os.dup2(write_fd, 2)
        func()
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)
        os.close(write_fd)

    chunks = []
    while True:
        chunk = os.read(read_fd, 4096)
        if not chunk:
            break
        chunks.append(chunk)
    os.close(read_fd)
    return b"".join(chunks).decode("utf-8", errors="replace")


def call_once(lib, r1, r2, fast, timing=False, inner_repeats=1):
    os.environ["RTDL_FAST_BARCODE"] = "1" if fast else "0"
    os.environ["RTDL_TIMING"] = "1" if timing else "0"
    os.environ["RTDL_BENCH_REPEAT"] = str(inner_repeats)

    def run():
        result = lib.rtd_lite_run_matrix(
            r1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            r2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            int(r1.shape[0]),
            True,
        )
        lib.rtd_lite_result_free(ctypes.byref(result))

    start = time.perf_counter()
    if timing:
        stderr_text = capture_c_stderr(run)
    else:
        run()
        stderr_text = ""
    wall = time.perf_counter() - start

    if timing:
        fields = parse_timing(stderr_text)
        fields["wall_sec"] = wall
        return fields
    return {"wall_sec": wall}


def median_field(samples, key):
    return statistics.median(sample[key] for sample in samples)


def loglog_slope(sizes, values):
    valid = [(n, v) for n, v in zip(sizes, values) if v > 0]
    if len(valid) < 2:
        return float("nan")
    xs = np.log([n for n, _ in valid])
    ys = np.log([v for _, v in valid])
    return float(np.polyfit(xs, ys, 1)[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[100, 200, 400, 800, 1600])
    parser.add_argument("--outer-repeats", type=int, default=5)
    parser.add_argument("--inner-repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--case", choices=["tsp", "dense"], default="tsp")
    parser.add_argument("--seed", type=int, default=20260618)
    args = parser.parse_args()

    lib = load_lib()
    rng = np.random.default_rng(args.seed)

    make_case = tsp_case if args.case == "tsp" else dense_case
    rows = []

    print(
        "case,n,naive_after_mst_sec,fast_after_mst_sec,after_mst_speedup,"
        "naive_setup_sec,fast_setup_sec,naive_tsp_tail_sec,fast_tsp_tail_sec,"
        "naive_wall_sec,fast_wall_sec,fast_fallback"
    )
    for n in args.sizes:
        r1, r2 = make_case(n, rng)
        for _ in range(args.warmup):
            call_once(lib, r1, r2, fast=False, timing=True, inner_repeats=max(1, args.inner_repeats // 2))
            call_once(lib, r1, r2, fast=True, timing=True, inner_repeats=max(1, args.inner_repeats // 2))

        naive_samples = [
            call_once(lib, r1, r2, fast=False, timing=True, inner_repeats=args.inner_repeats)
            for _ in range(args.outer_repeats)
        ]
        fast_samples = [
            call_once(lib, r1, r2, fast=True, timing=True, inner_repeats=args.inner_repeats)
            for _ in range(args.outer_repeats)
        ]

        naive_pair = median_field(naive_samples, "after_mst_median_sec")
        fast_pair = median_field(fast_samples, "after_mst_median_sec")
        speedup = naive_pair / fast_pair if fast_pair > 0 else float("inf")
        fast_fallback = max(sample["fallback"] for sample in fast_samples)
        rows.append((n, naive_pair, fast_pair))

        print(
            f"{args.case},{n},"
            f"{naive_pair:.9f},{fast_pair:.9f},{speedup:.3f},"
            f"{median_field(naive_samples, 'setup_sec'):.9f},{median_field(fast_samples, 'setup_sec'):.9f},"
            f"{median_field(naive_samples, 'tsp_tail_sec'):.9f},{median_field(fast_samples, 'tsp_tail_sec'):.9f},"
            f"{median_field(naive_samples, 'wall_sec'):.9f},{median_field(fast_samples, 'wall_sec'):.9f},"
            f"{fast_fallback}"
        )

    slope_naive = loglog_slope([row[0] for row in rows], [row[1] for row in rows])
    slope_fast = loglog_slope([row[0] for row in rows], [row[2] for row in rows])
    print(f"# loglog_slope_after_mst_naive={slope_naive:.3f}")
    print(f"# loglog_slope_after_mst_fast={slope_fast:.3f}")


if __name__ == "__main__":
    main()
