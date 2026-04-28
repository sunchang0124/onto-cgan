"""
Visualize evaluation_results.csv produced by evaluate.py.

Usage:
    python3.11 evaluation/visualize_results.py \
        --results output/cgan_519_diagnosis/no_split_seed42/evaluation_results.csv

Outputs (same directory as the CSV):
    plot_heatmap.png       — F1 / ROC-AUC heatmaps per mode × classifier
    plot_metrics_bar.png   — grouped bar chart (Recall, Precision, F1, ROC-AUC) per mode
    plot_pr_scatter.png    — Precision vs Recall scatter per mode × classifier
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path


# ── Cosmetic helpers ──────────────────────────────────────────────────────────

CLF_COLORS  = {'RandomForest': '#2166ac', 'KNN': '#d6604d', 'GaussianNB': '#4dac26'}
CLF_MARKERS = {'RandomForest': 'o',       'KNN': 's',       'GaussianNB': '^'}

def _short_scenario(s: str) -> str:
    s = s.replace("S2 — ", "").replace("S1 — ", "S1 ")
    # bare ontology mode → keep full name; bare decoder mode → dec_onto
    if s == 'ontology':
        return 'ontology'
    if s == 'decoder':
        return 'dec_onto'
    # ontology_xxx → xxx (strip prefix)
    if s.startswith('ontology_'):
        return s[len('ontology_'):]
    # decoder_xxx → dec_xxx
    s = s.replace('decoder_', 'dec_')
    return s


# ── Plot 1 : Heatmaps (F1 and ROC-AUC) ──────────────────────────────────────

def plot_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    metrics = [('F1', 'Blues'), ('ROC-AUC (unseen)', 'Greens')]
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    for ax, (metric, cmap) in zip(axes, metrics):
        pivot = (
            df.assign(Mode=df['Scenario'].map(_short_scenario))
              .pivot(index='Mode', columns='Classifier', values=metric)
        )
        # Sort rows by mean across classifiers descending
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

        sns.heatmap(
            pivot, ax=ax, cmap=cmap, annot=True, fmt='.3f',
            linewidths=0.5, linecolor='white',
            vmin=0, vmax=1,
            cbar_kws={'shrink': 0.7},
        )
        ax.set_title(metric, fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=9)
        plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)

    fig.suptitle('Evaluation metrics by mode & classifier', fontsize=14,
                 fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── Plot 2 : Grouped bar chart ────────────────────────────────────────────────

def plot_metrics_bar(df: pd.DataFrame, out_path: Path) -> None:
    metrics = ['Recall', 'Precision', 'F1', 'ROC-AUC (unseen)']
    classifiers = df['Classifier'].unique().tolist()

    # One row per classifier
    fig, axes = plt.subplots(len(classifiers), 1,
                             figsize=(16, 5 * len(classifiers)),
                             sharex=True)
    if len(classifiers) == 1:
        axes = [axes]

    # Sort modes by RandomForest F1 descending
    rf_order = (
        df[df['Classifier'] == 'RandomForest']
          .assign(Mode=df.loc[df['Classifier'] == 'RandomForest', 'Scenario']
                          .map(_short_scenario))
          .sort_values('F1', ascending=False)['Mode']
          .tolist()
    )

    n_modes   = len(rf_order)
    n_metrics = len(metrics)
    bar_w     = 0.18
    x         = np.arange(n_modes)
    offsets   = np.linspace(-(n_metrics - 1) / 2, (n_metrics - 1) / 2, n_metrics) * bar_w

    metric_colors = ['#4393c3', '#92c5de', '#2ca25f', '#d6604d']

    for ax, clf in zip(axes, classifiers):
        sub = (df[df['Classifier'] == clf]
               .assign(Mode=df.loc[df['Classifier'] == clf, 'Scenario']
                               .map(_short_scenario))
               .set_index('Mode')
               .reindex(rf_order))

        for i, (metric, color) in enumerate(zip(metrics, metric_colors)):
            vals = sub[metric].fillna(0).values
            bars = ax.bar(x + offsets[i], vals, bar_w,
                          label=metric, color=color, alpha=0.85, edgecolor='white')

        ax.set_title(clf, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Score')
        ax.axhline(0.5, color='grey', linewidth=0.8, linestyle='--', alpha=0.6)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_xticks(x)

    axes[-1].set_xticklabels(rf_order, rotation=40, ha='right', fontsize=9)
    fig.suptitle('Metrics by mode (sorted by RandomForest F1)',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── Plot 3 : Precision–Recall scatter ────────────────────────────────────────

CLF_SHORT = {'RandomForest': 'RF', 'KNN': 'KNN', 'GaussianNB': 'NB'}


CLF_ANNOT_OFFSET = {
    'RandomForest': ( 6,  5),
    'KNN':          ( 6, -12),
    'GaussianNB':   (-6,  5),
}


def plot_pr_scatter(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 8))

    df = df.assign(Mode=df['Scenario'].map(_short_scenario))

    for _, row in df.iterrows():
        clf  = row['Classifier']
        mode = row['Mode']
        ax.scatter(row['Recall'], row['Precision'],
                   color=CLF_COLORS[clf],
                   marker=CLF_MARKERS[clf],
                   s=55, alpha=0.85, zorder=3,
                   edgecolors='white', linewidths=0.5)
        ax.annotate(mode,
                    (row['Recall'], row['Precision']),
                    textcoords='offset points',
                    xytext=CLF_ANNOT_OFFSET[clf],
                    fontsize=7.5, alpha=0.90,
                    color=CLF_COLORS[clf])

    # F1 iso-curves
    for f1_val in [0.2, 0.4, 0.6, 0.8]:
        r = np.linspace(0.01, 1.0, 300)
        p = f1_val * r / (2 * r - f1_val)
        mask = (p >= 0) & (p <= 1)
        ax.plot(r[mask], p[mask], '--', color='grey', linewidth=0.7, alpha=0.5)
        ax.annotate(f'F1={f1_val}',
                    xy=(r[mask][-1], p[mask][-1]),
                    fontsize=8, color='grey', alpha=0.7)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('Recall (unseen disease only)', fontsize=11)
    ax.set_ylabel('Precision (unseen disease only)', fontsize=11)
    ax.set_title('Precision–Recall by mode & classifier\n'
                 '(metrics computed for the unseen disease class only)',
                 fontsize=11, fontweight='bold')
    ax.tick_params(axis='both', labelsize=9)
    ax.grid(alpha=0.3)

    legend_handles = [
        mpatches.Patch(color=CLF_COLORS[clf], label=clf)
        for clf in CLF_COLORS
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc='upper left')

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True,
                        help='Path to evaluation_results.csv')
    args = parser.parse_args()

    csv_path = Path(args.results)
    out_dir  = csv_path.parent
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} rows from {csv_path}")

    # CV all-folds file: average numeric columns across folds before plotting
    if 'fold' in df.columns:
        n_folds = df['fold'].nunique()
        numeric_cols = df.select_dtypes(include='number').columns.difference(['fold'])
        df = (df.groupby(['Classifier', 'Scenario'])[numeric_cols]
                .mean()
                .reset_index())
        print(f"CV mode: averaged {n_folds} folds")

    print(f"Modes     : {df['Scenario'].nunique()}")
    print(f"Classifiers: {df['Classifier'].unique().tolist()}")

    plot_heatmap(df,      out_dir / 'plot_heatmap.png')
    plot_metrics_bar(df,  out_dir / 'plot_metrics_bar.png')
    plot_pr_scatter(df,   out_dir / 'plot_pr_scatter.png')

    print("\nDone. All plots saved to:", out_dir)


if __name__ == '__main__':
    main()
