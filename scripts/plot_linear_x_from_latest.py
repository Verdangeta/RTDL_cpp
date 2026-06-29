#!/usr/bin/env python3
"""Redraw benchmark figures from cached numbers (no C++ re-benchmark)."""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"
CONTEXT_DIR = FIG_DIR / "context"

AFTER_MST_YLABEL = "time after MSTs available, seconds"
AFTER_MST_TSP_TITLE = "TSP case: barcode restoration after MSTs are available"
AFTER_MST_DENSITY_TITLE = "Graph density cases: barcode restoration after MSTs are available"
WALL_YLABEL = "public API wall time, seconds (context)"
WALL_TSP_TITLE = "TSP case: end-to-end API wall time (context)"
WALL_DENSITY_TITLE = "Graph density cases: end-to-end API wall time (context)"

TSP_ROWS = [
    (100, 0.000133987, 0.000192311),
    (200, 0.000513563, 0.000457035),
    (400, 0.001926091, 0.000999470),
    (800, 0.007642103, 0.002128036),
    (1600, 0.032919236, 0.004585889),
    (2400, 0.081606063, 0.007164200),
    (3200, 0.154852855, 0.009904453),
    (4000, 0.261643284, 0.012604159),
]

DENSITY_ROWS = {
    "m = n - 1": [
        (100, 0.000120911, 0.000166242),
        (200, 0.000486077, 0.000379352),
        (400, 0.001698499, 0.000837106),
        (800, 0.006764520, 0.001804314),
        (1600, 0.030610938, 0.003869367),
        (2400, 0.081504612, 0.006120384),
        (3200, 0.150218961, 0.008374369),
        (4000, 0.246177314, 0.010529149),
    ],
    "m = n log n": [
        (100, 0.000119718, 0.000168388),
        (200, 0.000478443, 0.000377894),
        (400, 0.001680485, 0.000838780),
        (800, 0.006772047, 0.001785217),
        (1600, 0.030427910, 0.003745368),
        (2400, 0.078733185, 0.005817100),
        (3200, 0.149479421, 0.008068735),
        (4000, 0.242528091, 0.010262016),
    ],
    "m = n^1.5": [
        (100, 0.000133742, 0.000167808),
        (200, 0.000509500, 0.000379768),
        (400, 0.001808653, 0.000835973),
        (800, 0.006874711, 0.001805747),
        (1600, 0.030800078, 0.003727495),
        (2400, 0.077072622, 0.005822505),
        (3200, 0.147925854, 0.008086543),
        (4000, 0.244569936, 0.010333342),
    ],
    "dense complete": [
        (100, 0.000143644, 0.000200361),
        (200, 0.000548223, 0.000438036),
        (400, 0.001962218, 0.000974707),
        (800, 0.007841382, 0.002130248),
        (1600, 0.034777891, 0.004575998),
        (2400, 0.085383070, 0.007203062),
        (3200, 0.165581357, 0.009822046),
        (4000, 0.273062281, 0.012607819),
    ],
}

TSP_WALL_ROWS = [
    (100, 0.000443362, 0.000518478),
    (200, 0.001775302, 0.001737691),
    (400, 0.007295392, 0.006041646),
    (800, 0.031092316, 0.025299489),
    (1600, 0.137958407, 0.109849267),
    (2400, 0.339505017, 0.260986760),
    (3200, 0.620914191, 0.475289568),
    (4000, 0.992387034, 0.737460501),
]

WALL_DENSITY_ROWS = {
    "m = n - 1": [
        (100, 0.000404820, 0.000459127),
        (200, 0.001357459, 0.001265295),
        (400, 0.004800133, 0.003958233),
        (800, 0.021607175, 0.016581178),
        (1600, 0.129431762, 0.104494832),
        (2400, 0.323344082, 0.251203209),
        (3200, 0.622672208, 0.476887152),
        (4000, 0.946155794, 0.875699848),
    ],
    "m = n log n": [
        (100, 0.000436708, 0.000495836),
        (200, 0.001422726, 0.001344748),
        (400, 0.005129024, 0.004312761),
        (800, 0.022529595, 0.017707981),
        (1600, 0.126002885, 0.099888481),
        (2400, 0.325006254, 0.248727277),
        (3200, 0.600607678, 0.455265194),
        (4000, 0.953493804, 0.720431231),
    ],
    "m = n^1.5": [
        (100, 0.000500374, 0.000546530),
        (200, 0.001624264, 0.001511946),
        (400, 0.005712397, 0.004747681),
        (800, 0.023765899, 0.018737361),
        (1600, 0.131339394, 0.104406558),
        (2400, 0.329677053, 0.255956620),
        (3200, 0.597398616, 0.459102564),
        (4000, 0.965139471, 0.734178238),
    ],
    "dense complete": [
        (100, 0.000478491, 0.000545204),
        (200, 0.001594640, 0.001509681),
        (400, 0.005771562, 0.004795611),
        (800, 0.024639904, 0.019055530),
        (1600, 0.129702084, 0.098511465),
        (2400, 0.338506378, 0.256580219),
        (3200, 0.633484878, 0.471201882),
        (4000, 1.010028556, 0.748169333),
    ],
}


def arrays(rows):
    data = np.asarray(rows, dtype=float)
    return data[:, 0], data[:, 1], data[:, 2]


def loglog_slope(sizes, values):
    valid = [(n, v) for n, v in zip(sizes, values) if v > 0]
    if len(valid) < 2:
        return float("nan")
    xs = np.log([n for n, _ in valid])
    ys = np.log([v for _, v in valid])
    return float(np.polyfit(xs, ys, 1)[0])


def plot_tsp_log():
    ns, naive, fast = arrays(TSP_ROWS)
    slope_naive = loglog_slope(ns, naive)
    slope_fast = loglog_slope(ns, fast)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(ns, naive, marker="o", label=f"RTDL baseline, slope {slope_naive:.2f}")
    ax.plot(ns, fast, marker="o", label=f"RTDL-LCT, slope {slope_fast:.2f}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("n vertices")
    ax.set_ylabel(AFTER_MST_YLABEL)
    ax.set_title(AFTER_MST_TSP_TITLE)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output = FIG_DIR / "barcode_time_tsp.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"saved,{output}")


def plot_tsp_linear():
    ns, naive, fast = arrays(TSP_ROWS)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(ns, naive, marker="o", label="RTDL baseline")
    ax.plot(ns, fast, marker="o", label="RTDL-LCT")
    ax.set_xlabel("n vertices")
    ax.set_ylabel(AFTER_MST_YLABEL)
    ax.set_title(f"{AFTER_MST_TSP_TITLE} (linear x)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output = FIG_DIR / "barcode_time_tsp_linear_x.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"saved,{output}")


def plot_wall_tsp_log():
    ns, naive, fast = arrays(TSP_WALL_ROWS)
    slope_naive = loglog_slope(ns, naive)
    slope_fast = loglog_slope(ns, fast)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(ns, naive, marker="o", label=f"RTDL baseline, slope {slope_naive:.2f}")
    ax.plot(ns, fast, marker="o", label=f"RTDL-LCT, slope {slope_fast:.2f}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("n vertices")
    ax.set_ylabel(WALL_YLABEL)
    ax.set_title(WALL_TSP_TITLE)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output = CONTEXT_DIR / "barcode_wall_time_tsp.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"saved,{output}")


def plot_wall_tsp_linear():
    ns, naive, fast = arrays(TSP_WALL_ROWS)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(ns, naive, marker="o", label="RTDL baseline")
    ax.plot(ns, fast, marker="o", label="RTDL-LCT")
    ax.set_xlabel("n vertices")
    ax.set_ylabel(WALL_YLABEL)
    ax.set_title(f"{WALL_TSP_TITLE} (linear x)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output = CONTEXT_DIR / "barcode_wall_time_tsp_linear_x.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"saved,{output}")


def plot_density_grid_log():
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, (title, rows) in zip(axes, DENSITY_ROWS.items()):
        ns, naive, fast = arrays(rows)
        slope_naive = loglog_slope(ns, naive)
        slope_fast = loglog_slope(ns, fast)
        ax.plot(ns, naive, marker="o", label=f"RTDL baseline {slope_naive:.2f}")
        ax.plot(ns, fast, marker="o", label=f"RTDL-LCT {slope_fast:.2f}")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    fig.supxlabel("n vertices")
    fig.supylabel(AFTER_MST_YLABEL)
    fig.suptitle(AFTER_MST_DENSITY_TITLE, y=0.99)
    fig.tight_layout()
    output = FIG_DIR / "barcode_time_graph_densities.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"saved,{output}")


def plot_density_grid_linear():
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, (title, rows) in zip(axes, DENSITY_ROWS.items()):
        ns, naive, fast = arrays(rows)
        ax.plot(ns, naive, marker="o", label="RTDL baseline")
        ax.plot(ns, fast, marker="o", label="RTDL-LCT")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.supxlabel("n vertices")
    fig.supylabel(AFTER_MST_YLABEL)
    fig.suptitle(f"{AFTER_MST_DENSITY_TITLE} (linear x)", y=0.99)
    fig.tight_layout()
    output = FIG_DIR / "barcode_time_graph_densities_linear_x.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"saved,{output}")


def plot_wall_density_grid_log():
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, (title, rows) in zip(axes, WALL_DENSITY_ROWS.items()):
        ns, naive, fast = arrays(rows)
        slope_naive = loglog_slope(ns, naive)
        slope_fast = loglog_slope(ns, fast)
        ax.plot(ns, naive, marker="o", label=f"RTDL baseline {slope_naive:.2f}")
        ax.plot(ns, fast, marker="o", label=f"RTDL-LCT {slope_fast:.2f}")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    fig.supxlabel("n vertices")
    fig.supylabel(WALL_YLABEL)
    fig.suptitle(WALL_DENSITY_TITLE, y=0.99)
    fig.tight_layout()
    output = CONTEXT_DIR / "barcode_wall_time_graph_densities.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"saved,{output}")


def plot_wall_density_grid_linear():
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, (title, rows) in zip(axes, WALL_DENSITY_ROWS.items()):
        ns, naive, fast = arrays(rows)
        ax.plot(ns, naive, marker="o", label="RTDL baseline")
        ax.plot(ns, fast, marker="o", label="RTDL-LCT")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.supxlabel("n vertices")
    fig.supylabel(WALL_YLABEL)
    fig.suptitle(f"{WALL_DENSITY_TITLE} (linear x)", y=0.99)
    fig.tight_layout()
    output = CONTEXT_DIR / "barcode_wall_time_graph_densities_linear_x.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"saved,{output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-wall",
        action="store_true",
        help="also redraw wall-time context figures under docs/figures/context/",
    )
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_tsp_log()
    plot_tsp_linear()
    plot_density_grid_log()
    plot_density_grid_linear()

    if args.with_wall:
        CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
        plot_wall_tsp_log()
        plot_wall_tsp_linear()
        plot_wall_density_grid_log()
        plot_wall_density_grid_linear()


if __name__ == "__main__":
    main()
