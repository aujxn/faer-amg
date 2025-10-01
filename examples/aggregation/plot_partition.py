#!/usr/bin/env python3
"""Visualize aggregate assignments for MFEM data."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


def _load_partition(path: Path):
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"File {path} has no header row")

        coord_fields: List[str] = [name for name in reader.fieldnames if name.startswith("coord")]
        if len(coord_fields) < 2:
            raise ValueError(
                "Partition file must contain at least two coordinate columns (coord0, coord1, ...)."
            )

        nodes, aggregates = [], []
        coords = [[] for _ in coord_fields]

        for row in reader:
            nodes.append(int(row["node"]))
            aggregates.append(int(row["aggregate"]))
            for axis, field in enumerate(coord_fields):
                coords[axis].append(float(row[field]))

    return nodes, coords, aggregates


def plot_partition(csv_path: Path, save_path: Path | None = None) -> None:
    nodes, coords, aggregates = _load_partition(csv_path)

    x, y = coords[0], coords[1]
    agg_to_nodes = defaultdict(list)
    for node, agg in zip(nodes, aggregates):
        agg_to_nodes[agg].append(node)

    unique_aggs = sorted(agg_to_nodes)
    cmap = plt.cm.get_cmap("tab20", max(len(unique_aggs), 1))

    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, agg in enumerate(unique_aggs):
        indices = agg_to_nodes[agg]
        ax.scatter(
            [x[i] for i in indices],
            [y[i] for i in indices],
            s=18,
            label=f"Agg {agg}",
            color=cmap(idx % cmap.N),
        )

    ax.set_title("Aggregate Partition")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", linewidth=0.5)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
    else:
        plt.show()


def main() -> None:
    default_csv = (
        Path(__file__).resolve().parent / "anisotropy_partition.csv"
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv",
        nargs="?",
        type=Path,
        default=default_csv,
        help=f"Path to partition CSV (default: {default_csv})",
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="Optional path to save the figure instead of displaying it",
    )

    args = parser.parse_args()
    plot_partition(args.csv, args.save)


if __name__ == "__main__":
    main()
