"""
Aggregate evaluation_results.csv across multiple seed runs.

Usage
-----
    python evaluation/aggregate.py --dirs output/with_s3_seed*
    python evaluation/aggregate.py --dirs output/with_s3_seed42 output/with_s3_seed7 ...
    python evaluation/aggregate.py --dirs output/no_split_seed*

Output
------
    Prints mean ± std table to stdout.
    Saves aggregate_mean.csv and aggregate_std.csv to the first parent directory.
"""

import argparse
import pandas as pd
from pathlib import Path


def main(dirs):
    dfs = []
    for d in dirs:
        csv = Path(d) / "evaluation_results.csv"
        if not csv.exists():
            print(f"WARNING: {csv} not found — skipping")
            continue
        dfs.append(pd.read_csv(csv, index_col="Scenario"))
        print(f"  Loaded {csv}")

    if not dfs:
        print("No results found.")
        return

    # Numeric columns only (TP / total unseen is a string)
    numeric_cols = [c for c in dfs[0].columns if c != 'TP / total unseen']

    combined = pd.concat(
        {i: df[numeric_cols] for i, df in enumerate(dfs)},
        axis=0
    )  # MultiIndex: (run_idx, Scenario)

    mean_df = combined.groupby(level=1).mean().round(4)
    std_df  = combined.groupby(level=1).std().round(4)

    # Reorder rows to match original scenario order
    scenario_order = dfs[0].index.tolist()
    mean_df = mean_df.reindex(scenario_order)
    std_df  = std_df.reindex(scenario_order)

    # Pretty-print as "mean ± std"
    summary = mean_df.copy().astype(str)
    for col in numeric_cols:
        summary[col] = mean_df[col].map('{:.4f}'.format) + \
                       ' ± ' + std_df[col].map('{:.4f}'.format)

    print(f"\n{'='*70}")
    print(f"  Results across {len(dfs)} runs (mean ± std)")
    print(f"{'='*70}")
    print(summary.to_string())

    # Save
    out_dir = Path(dirs[0]).parent
    mean_df.to_csv(out_dir / "aggregate_mean.csv")
    std_df.to_csv(out_dir  / "aggregate_std.csv")
    summary.to_csv(out_dir / "aggregate_summary.csv")
    print(f"\nSaved → {out_dir}/aggregate_mean.csv")
    print(f"Saved → {out_dir}/aggregate_std.csv")
    print(f"Saved → {out_dir}/aggregate_summary.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirs', nargs='+', required=True,
                        help='Output directories to aggregate (supports shell glob expansion)')
    args = parser.parse_args()
    main(args.dirs)
