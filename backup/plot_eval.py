"""Evaluation plots for runs trained with train.py.

Reads <run_dir>/eval_val.npz (validation scores of the best model + training
history) and writes, per run:
    training_history.png    train/val loss and val AUC vs epoch
    roc.png                 ROC (nue efficiency vs tau efficiency) with AUC 95% CI
    score_distribution.png  classifier score for tau vs nue
and, for all runs together, into --out_dir:
    roc_comparison.png
    summary.csv             AUC (+ bootstrap 95% CI), tau eff. at fixed nue eff.

Example:
    python plot_eval.py --runs trained_model5TeV/a trained_model5TeV/b \
        --labels "A" "B" --out_dir trained_model5TeV/comparison
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

# reference palette (light): tau = signal, nue = background; runs use slots 1-3
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
GRID      = "#e4e3df"
TAU_COLOR = "#2a78d6"
NUE_COLOR = "#eb6834"
RUN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # all-pairs CVD-safe for <= 3 series
NUE_EFFS = [0.1, 0.3]  # report tau efficiency at these nue (background) efficiencies

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "font.size": 11, "legend.frameon": False,
})


def bootstrap_auc(trues, preds, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(trues), len(trues))
        if len(np.unique(trues[idx])) < 2:
            continue
        aucs.append(roc_auc_score(trues[idx], preds[idx]))
    return np.percentile(aucs, [2.5, 97.5])


def tau_eff_at(fpr, tpr, nue_eff):
    return float(np.interp(nue_eff, fpr, tpr))


def load_run(run_dir):
    d = np.load(os.path.join(run_dir, "eval_val.npz"))
    return {k: d[k] for k in d.files}


def plot_history(r, label, path):
    epochs = np.arange(1, len(r["train_losses"]) + 1)
    best = int(r["best_epoch"])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax1.plot(epochs, r["train_losses"], color=TAU_COLOR, label="Train")
    ax1.plot(epochs, r["val_losses"], color=NUE_COLOR, label="Validation")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.set_title(f"{label}: training history", loc="left")
    ax2.plot(epochs, r["val_aucs"], color=TAU_COLOR)
    ax2.set_ylabel("Validation AUC")
    ax2.set_xlabel("Epoch")
    for ax in (ax1, ax2):
        ax.axvline(best, color=INK_2, linestyle=":", linewidth=1.2)
    ax2.annotate(f"best epoch {best}", xy=(best, ax2.get_ylim()[0]), xytext=(4, 4),
                 textcoords="offset points", color=INK_2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(curves, title, path):
    """curves: list of (label, fpr, tpr, auc, ci, color)."""
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], color=INK_2, linestyle="--", linewidth=1, label="Random")
    for label, fpr, tpr, auc, ci, color in curves:
        ax.plot(fpr, tpr, color=color,
                label=f"{label}  AUC {auc:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
    ax.set_xlabel("nue efficiency (background accepted)")
    ax.set_ylabel("tau efficiency (signal accepted)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(title, loc="left")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scores(r, label, path):
    preds, trues = r["preds"], r["trues"]
    bins = np.linspace(0, 1, 31)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(preds[trues == 1], bins=bins, density=True, histtype="step", linewidth=2,
            color=TAU_COLOR, label=f"nutau (signal), n={int((trues == 1).sum())}")
    ax.hist(preds[trues == 0], bins=bins, density=True, histtype="step", linewidth=2,
            color=NUE_COLOR, label=f"nue (background), n={int((trues == 0).sum())}")
    ax.set_xlabel("Classifier score (sigmoid)")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)
    ax.set_title(f"{label}: score distribution (validation)", loc="left")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="Run directories containing eval_val.npz")
    p.add_argument("--labels", nargs="+", default=None, help="Display label per run")
    p.add_argument("--out_dir", required=True, help="Where the comparison plot and summary go")
    args = p.parse_args()

    labels = args.labels or [os.path.basename(os.path.normpath(r)) for r in args.runs]
    if len(labels) != len(args.runs):
        p.error("--labels must have one entry per --runs")
    if len(args.runs) > len(RUN_COLORS):
        p.error(f"at most {len(RUN_COLORS)} runs per comparison plot (colour-safe limit)")
    os.makedirs(args.out_dir, exist_ok=True)

    rows, curves = [], []
    for run_dir, label, color in zip(args.runs, labels, RUN_COLORS):
        r = load_run(run_dir)
        preds, trues = r["preds"], r["trues"]
        fpr, tpr, _ = roc_curve(trues, preds)
        auc = roc_auc_score(trues, preds)
        ci = bootstrap_auc(trues, preds)

        plot_history(r, label, os.path.join(run_dir, "training_history.png"))
        plot_roc([(label, fpr, tpr, auc, ci, TAU_COLOR)], f"{label}: ROC (validation)",
                 os.path.join(run_dir, "roc.png"))
        plot_scores(r, label, os.path.join(run_dir, "score_distribution.png"))
        curves.append((label, fpr, tpr, auc, ci, color))

        row = {"run": label, "dir": run_dir, "n_val": len(trues),
               "best_epoch": int(r["best_epoch"]), "auc": round(auc, 4),
               "auc_ci_low": round(ci[0], 4), "auc_ci_high": round(ci[1], 4)}
        for e in NUE_EFFS:
            row[f"tau_eff_at_nue_eff_{e}"] = round(tau_eff_at(fpr, tpr, e), 3)
        rows.append(row)

    plot_roc(curves, "ROC comparison (validation)", os.path.join(args.out_dir, "roc_comparison.png"))
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    print(summary.to_string(index=False))
    print(f"\nPer-run plots written into each run dir; comparison in {args.out_dir}")


if __name__ == "__main__":
    main()
