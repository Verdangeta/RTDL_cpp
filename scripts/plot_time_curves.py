#!/usr/bin/env python3
"""Build time-vs-size plots for naive and fast RTDL barcode restoration after MSTs."""

import argparse
import math
import pathlib
import statistics

import matplotlib.pyplot as plt
import numpy as np

from benchmark_fast_barcode import call_once, dense_case, load_lib, loglog_slope, tsp_case


ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"
CONTEXT_DIR = FIG_DIR / "context"

AFTER_MST_YLABEL = "time after MSTs available, seconds"
AFTER_MST_TSP_TITLE = "TSP case: barcode restoration after MSTs are available"
AFTER_MST_DENSITY_TITLE = "Graph density cases: barcode restoration after MSTs are available"
WALL_YLABEL = "public API wall time, seconds (context)"
WALL_TSP_TITLE = "TSP case: end-to-end API wall time (context)"
WALL_DENSITY_TITLE = "Graph density cases: end-to-end API wall time (context)"


def sparse_connected_matrix(n, rng, edge_count):
    matrix = np.full((n, n), math.inf, dtype=np.float64)
    np.fill_diagonal(matrix, 0.0)
    if n <= 1:
        return np.ascontiguousarray(matrix)

    edges = set()
    order = list(rng.permutation(n))
    for pos in range(1, n):
        u = int(order[pos])
        v = int(order[int(rng.integers(0, pos))])
        edges.add(tuple(sorted((u, v))))

    max_edges = n * (n - 1) // 2
    target = min(max(edge_count, n - 1), max_edges)
    while len(edges) < target:
        u = int(rng.integers(0, n))
        v = int(rng.integers(0, n))
        if u != v:
            edges.add(tuple(sorted((u, v))))

    for u, v in edges:
        weight = float(rng.uniform(0.1, 10.0))
        matrix[u, v] = matrix[v, u] = weight
    return np.ascontiguousarray(matrix)


def graph_density_case(n, rng, density_name):
    if density_name == "tree":
        edge_count = n - 1
    elif density_name == "nlogn":
        edge_count = int(n * math.log(max(n, 2)))
    elif density_name == "n32":
        edge_count = int(n ** 1.5)
    elif density_name == "dense":
        return dense_case(n, rng)
    else:
        raise ValueError(f"Unknown density: {density_name}")

    return (
        sparse_connected_matrix(n, rng, edge_count),
        sparse_connected_matrix(n, rng, edge_count),
    )


def median(values):
    return statistics.median(values)


def timing_field(sample):
    return sample.get("after_mst_median_sec", sample["pair_median_sec"])


def measure_after_mst_case(lib, r1, r2, outer_repeats, inner_repeats, warmup):
    for _ in range(warmup):
        call_once(lib, r1, r2, fast=False, timing=True, inner_repeats=max(1, inner_repeats // 2))
        call_once(lib, r1, r2, fast=True, timing=True, inner_repeats=max(1, inner_repeats // 2))

    naive = [
        timing_field(call_once(lib, r1, r2, fast=False, timing=True, inner_repeats=inner_repeats))
        for _ in range(outer_repeats)
    ]
    fast = [
        timing_field(call_once(lib, r1, r2, fast=True, timing=True, inner_repeats=inner_repeats))
        for _ in range(outer_repeats)
    ]
    return median(naive), median(fast)


def measure_wall_case(lib, r1, r2, outer_repeats, warmup):
    for _ in range(warmup):
        call_once(lib, r1, r2, fast=False, timing=False)
        call_once(lib, r1, r2, fast=True, timing=False)

    naive = [
        call_once(lib, r1, r2, fast=False, timing=False)["wall_sec"]
        for _ in range(outer_repeats)
    ]
    fast = [
        call_once(lib, r1, r2, fast=True, timing=False)["wall_sec"]
        for _ in range(outer_repeats)
    ]
    return median(naive), median(fast)


def plot_tsp(lib, sizes, rng, outer_repeats, inner_repeats, warmup, output_path, wall_output_path=None):
    naive_times = []
    fast_times = []
    naive_wall_times = []
    fast_wall_times = []
    for n in sizes:
        r1, r2 = tsp_case(n, rng)
        naive, fast = measure_after_mst_case(lib, r1, r2, outer_repeats, inner_repeats, warmup)
        naive_times.append(naive)
        fast_times.append(fast)
        if wall_output_path is not None:
            naive_wall, fast_wall = measure_wall_case(lib, r1, r2, outer_repeats, warmup)
            naive_wall_times.append(naive_wall)
            fast_wall_times.append(fast_wall)
            print(
                f"tsp,n={n},naive_after_mst={naive:.9f},fast_after_mst={fast:.9f},after_mst_speedup={naive / fast:.3f},"
                f"naive_wall={naive_wall:.9f},fast_wall={fast_wall:.9f},wall_speedup={naive_wall / fast_wall:.3f}"
            )
        else:
            print(
                f"tsp,n={n},naive_after_mst={naive:.9f},fast_after_mst={fast:.9f},after_mst_speedup={naive / fast:.3f}"
            )

    slope_naive = loglog_slope(sizes, naive_times)
    slope_fast = loglog_slope(sizes, fast_times)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(sizes, naive_times, marker="o", label=f"naive, slope {slope_naive:.2f}")
    ax.plot(sizes, fast_times, marker="o", label=f"LCT fast, slope {slope_fast:.2f}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("n vertices")
    ax.set_ylabel(AFTER_MST_YLABEL)
    ax.set_title(AFTER_MST_TSP_TITLE)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    if wall_output_path is None:
        return

    wall_slope_naive = loglog_slope(sizes, naive_wall_times)
    wall_slope_fast = loglog_slope(sizes, fast_wall_times)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(sizes, naive_wall_times, marker="o", label=f"naive, slope {wall_slope_naive:.2f}")
    ax.plot(sizes, fast_wall_times, marker="o", label=f"LCT fast, slope {wall_slope_fast:.2f}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("n vertices")
    ax.set_ylabel(WALL_YLABEL)
    ax.set_title(WALL_TSP_TITLE)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(wall_output_path, dpi=180)
    plt.close(fig)


def plot_density_grid(
    lib,
    sizes,
    densities,
    rng,
    outer_repeats,
    inner_repeats,
    warmup,
    output_path,
    wall_output_path=None,
):
    labels = {
        "tree": "m = n - 1",
        "nlogn": "m = n log n",
        "n32": "m = n^1.5",
        "dense": "dense complete",
    }

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True, sharey=True)
    axes = axes.ravel()
    wall_results = {}
    for ax, density_name in zip(axes, densities):
        naive_times = []
        fast_times = []
        naive_wall_times = []
        fast_wall_times = []
        for n in sizes:
            r1, r2 = graph_density_case(n, rng, density_name)
            naive, fast = measure_after_mst_case(lib, r1, r2, outer_repeats, inner_repeats, warmup)
            naive_times.append(naive)
            fast_times.append(fast)
            if wall_output_path is not None:
                naive_wall, fast_wall = measure_wall_case(lib, r1, r2, outer_repeats, warmup)
                naive_wall_times.append(naive_wall)
                fast_wall_times.append(fast_wall)
                print(
                    f"{density_name},n={n},naive_after_mst={naive:.9f},"
                    f"fast_after_mst={fast:.9f},after_mst_speedup={naive / fast:.3f},"
                    f"naive_wall={naive_wall:.9f},fast_wall={fast_wall:.9f},wall_speedup={naive_wall / fast_wall:.3f}"
                )
            else:
                print(
                    f"{density_name},n={n},naive_after_mst={naive:.9f},"
                    f"fast_after_mst={fast:.9f},after_mst_speedup={naive / fast:.3f}"
                )
        if wall_output_path is not None:
            wall_results[density_name] = (naive_wall_times, fast_wall_times)

        slope_naive = loglog_slope(sizes, naive_times)
        slope_fast = loglog_slope(sizes, fast_times)
        ax.plot(sizes, naive_times, marker="o", label=f"naive {slope_naive:.2f}")
        ax.plot(sizes, fast_times, marker="o", label=f"LCT {slope_fast:.2f}")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(labels[density_name])
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    fig.supxlabel("n vertices")
    fig.supylabel(AFTER_MST_YLABEL)
    fig.suptitle(AFTER_MST_DENSITY_TITLE, y=0.99)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    if wall_output_path is None:
        return

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, density_name in zip(axes, densities):
        naive_wall_times, fast_wall_times = wall_results[density_name]
        slope_naive = loglog_slope(sizes, naive_wall_times)
        slope_fast = loglog_slope(sizes, fast_wall_times)
        ax.plot(sizes, naive_wall_times, marker="o", label=f"naive {slope_naive:.2f}")
        ax.plot(sizes, fast_wall_times, marker="o", label=f"LCT {slope_fast:.2f}")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(labels[density_name])
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    fig.supxlabel("n vertices")
    fig.supylabel(WALL_YLABEL)
    fig.suptitle(WALL_DENSITY_TITLE, y=0.99)
    fig.tight_layout()
    fig.savefig(wall_output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsp-sizes", nargs="+", type=int, default=[100, 200, 400, 800, 1200, 1600])
    parser.add_argument("--graph-sizes", nargs="+", type=int, default=[100, 200, 400, 800, 1200])
    parser.add_argument("--outer-repeats", type=int, default=3)
    parser.add_argument("--inner-repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--output-dir", type=pathlib.Path, default=FIG_DIR)
    parser.add_argument(
        "--with-wall",
        action="store_true",
        help="also benchmark and plot end-to-end wall time under docs/figures/context/",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    context_dir = args.output_dir / "context"
    if args.with_wall:
        context_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    lib = load_lib()

    tsp_path = args.output_dir / "barcode_time_tsp.png"
    density_path = args.output_dir / "barcode_time_graph_densities.png"
    tsp_wall_path = context_dir / "barcode_wall_time_tsp.png" if args.with_wall else None
    density_wall_path = context_dir / "barcode_wall_time_graph_densities.png" if args.with_wall else None

    plot_tsp(
        lib,
        args.tsp_sizes,
        rng,
        args.outer_repeats,
        args.inner_repeats,
        args.warmup,
        tsp_path,
        tsp_wall_path,
    )
    plot_density_grid(
        lib,
        args.graph_sizes,
        ["tree", "nlogn", "n32", "dense"],
        rng,
        args.outer_repeats,
        args.inner_repeats,
        args.warmup,
        density_path,
        density_wall_path,
    )

    print(f"saved,{tsp_path}")
    print(f"saved,{density_path}")
    if args.with_wall:
        print(f"saved,{tsp_wall_path}")
        print(f"saved,{density_wall_path}")


if __name__ == "__main__":
    main()
