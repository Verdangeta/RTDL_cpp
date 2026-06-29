#!/usr/bin/env python3
"""Differential tests for RTDL_FAST_BARCODE.

The script calls the C API twice in one process: first with the naive path,
then with the Link-Cut-Tree path. It compares the final public result, including
the existing extra TSP barcode in the 2->1 direction.
"""

import ctypes
import math
import os
import pathlib

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


LIB = load_lib()


def bd_tuple(item):
    return (int(item.a), int(item.b), int(item.c), int(item.d))


def run_result(r1, r2, fast):
    os.environ["RTDL_FAST_BARCODE"] = "1" if fast else "0"
    r1 = np.ascontiguousarray(r1, dtype=np.float64)
    r2 = np.ascontiguousarray(r2, dtype=np.float64)
    n = int(r1.shape[0])
    result = LIB.rtd_lite_run_matrix(
        r1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        r2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n,
        True,
    )
    try:
        left = [bd_tuple(result.left_to_right[i]) for i in range(result.left_bars)]
        right = [bd_tuple(result.right_to_left[i]) for i in range(result.right_bars)]
        return left, right
    finally:
        LIB.rtd_lite_result_free(ctypes.byref(result))


def complete_matrix(n, rng, ties=False):
    if n == 1:
        return np.zeros((1, 1), dtype=np.float64)
    values = rng.uniform(0.1, 10.0, size=(n, n))
    matrix = (values + values.T) / 2.0
    if ties:
        matrix = np.round(matrix)
        matrix[matrix <= 0] = 1.0
    np.fill_diagonal(matrix, 0.0)
    return matrix


def sparse_connected_matrix(n, rng, edge_count, ties=False):
    matrix = np.full((n, n), math.inf, dtype=np.float64)
    np.fill_diagonal(matrix, 0.0)
    if n <= 1:
        return matrix

    edges = set()
    order = list(rng.permutation(n))
    for pos in range(1, n):
        u = order[pos]
        v = order[int(rng.integers(0, pos))]
        edges.add(tuple(sorted((int(u), int(v)))))

    max_edges = n * (n - 1) // 2
    target = min(max(edge_count, n - 1), max_edges)
    while len(edges) < target:
        u = int(rng.integers(0, n))
        v = int(rng.integers(0, n))
        if u != v:
            edges.add(tuple(sorted((u, v))))

    for u, v in edges:
        weight = float(rng.integers(1, 6) if ties else rng.uniform(0.1, 10.0))
        matrix[u, v] = matrix[v, u] = weight
    return matrix


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
    return r1, r2


def assert_same_case(name, r1, r2):
    naive = run_result(r1, r2, fast=False)
    fast = run_result(r1, r2, fast=True)
    if naive != fast:
        raise AssertionError(
            f"{name}: fast result differs\n"
            f"naive left/right={naive}\n"
            f"fast left/right={fast}"
        )


def main():
    rng = np.random.default_rng(20260618)
    cases = []

    for n in (1, 2, 3, 4, 8, 16, 32):
        cases.append((f"complete_n{n}", complete_matrix(n, rng), complete_matrix(n, rng)))
        cases.append((f"ties_complete_n{n}", complete_matrix(n, rng, ties=True), complete_matrix(n, rng, ties=True)))

    for n in (5, 10, 20, 35):
        for factor_name, edge_count in (
            ("tree", n - 1),
            ("nlogn", int(n * math.log(max(n, 2)))),
            ("n32", int(n ** 1.5)),
        ):
            cases.append((
                f"sparse_{factor_name}_n{n}",
                sparse_connected_matrix(n, rng, edge_count),
                sparse_connected_matrix(n, rng, edge_count),
            ))
            cases.append((
                f"ties_sparse_{factor_name}_n{n}",
                sparse_connected_matrix(n, rng, edge_count, ties=True),
                sparse_connected_matrix(n, rng, edge_count, ties=True),
            ))

    for n in (4, 8, 16, 32, 64):
        r1, r2 = tsp_case(n, rng)
        cases.append((f"tsp_n{n}", r1, r2))

    for name, r1, r2 in cases:
        assert_same_case(name, r1, r2)

    print(f"OK: {len(cases)} differential cases matched")


if __name__ == "__main__":
    main()
