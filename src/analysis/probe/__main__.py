"""CLI for probe analysis.

Usage:
    python -m src.analysis.probe table      <manifest_dir> [--method] [--layer] [--probes]
    python -m src.analysis.probe r2         <manifest_dir> [--method] [--layer] [--probes] [--group-by] [--output] [--no-plot]
    python -m src.analysis.probe similarity <manifest_dir> [--method] [--layer] [--probes] [--output] [--no-plot]
    python -m src.analysis.probe hoo        --results <path> [--output-dir]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.probe.helpers import add_filter_args, get_filters


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe analysis CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # table
    table_parser = subparsers.add_parser("table", help="Print probe summary table")
    add_filter_args(table_parser)

    # r2
    r2_parser = subparsers.add_parser("r2", help="Plot R² comparison")
    add_filter_args(r2_parser)
    r2_parser.add_argument("--group-by", type=str, help="Group by field (e.g., method, layer)")
    r2_parser.add_argument("--output", type=str, help="Output PNG path")
    r2_parser.add_argument("--no-plot", action="store_true", help="Text output only, skip plot")

    # similarity
    sim_parser = subparsers.add_parser("similarity", help="Plot probe similarity heatmap")
    add_filter_args(sim_parser)
    sim_parser.add_argument("--output", type=str, help="Output PNG path")
    sim_parser.add_argument("--no-plot", action="store_true", help="Text output only, skip plot")

    # hoo
    hoo_parser = subparsers.add_parser("hoo", help="Held-one-out analysis")
    hoo_parser.add_argument("--results", type=str, required=True, help="Path to hoo_evaluation_summary.json")
    hoo_parser.add_argument("--output-dir", type=str, help="Output directory for plots")

    # hoo-validation
    hoov_parser = subparsers.add_parser("hoo-validation", help="HOO validation (train→CV→test)")
    hoov_parser.add_argument("--validation", type=str, required=True, help="Path to hoo_validation_results.json")
    hoov_parser.add_argument("--hoo", type=str, required=True, help="Path to hoo_evaluation_summary.json")
    hoov_parser.add_argument("--manifest", type=str, required=True, help="Path to manifest.json")
    hoov_parser.add_argument("--output-dir", type=str, help="Output directory for plots")

    args = parser.parse_args()

    if args.command == "table":
        from src.analysis.probe.helpers import load_and_filter, format_probe_table

        filters = get_filters(args)
        _, probes = load_and_filter(**filters)
        print(format_probe_table(probes))

    elif args.command == "r2":
        from src.analysis.probe.plot_r2 import run

        filters = get_filters(args)
        run(
            **filters,
            group_by=args.group_by,
            output=Path(args.output) if args.output else None,
            no_plot=args.no_plot,
        )

    elif args.command == "similarity":
        from src.analysis.probe.plot_similarity import run

        filters = get_filters(args)
        run(
            **filters,
            output=Path(args.output) if args.output else None,
            no_plot=args.no_plot,
        )

    elif args.command == "hoo":
        from src.analysis.probe.plot_hoo import run

        run(
            results_path=Path(args.results),
            output_dir=Path(args.output_dir) if args.output_dir else None,
        )

    elif args.command == "hoo-validation":
        from src.analysis.probe.plot_hoo import run_validation

        run_validation(
            validation_path=Path(args.validation),
            hoo_path=Path(args.hoo),
            manifest_path=Path(args.manifest),
            output_dir=Path(args.output_dir) if args.output_dir else None,
        )


if __name__ == "__main__":
    main()
