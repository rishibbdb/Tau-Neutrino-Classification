"""Evaluation plots for runs trained with train.py.

Reads <run_dir>/eval_val.npz (validation scores of the best model + training
history) and writes, per run:
    training_history.png    train/val loss and val AUC vs epoch
    roc.png                 ROC (nue efficiency vs tau efficiency) with AUC 95% CI
    score_distribution.png  classifier score for tau vs nue
    aux_regression.png      (runs with an aux head) predicted vs true log10 E, direction error
and, for all runs together, into --out_dir:
    roc_comparison.png
    auc_vs_truth.png        (if eval_val.npz has y_aux) AUC within bins of true cos zenith / log10 E --
                            direction/energy can't separate the flavours inside a bin
    summary.csv             AUC (+ bootstrap 95% CI), tau eff. at fixed nue eff., mean within-bin AUCs

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


def truth_kinematics(r):
    """(cos zenith, log10 E) of each val event from the saved aux truth targets, or None."""
    if "y_aux" not in r:
        return None
    y = r["y_aux"]
    log_e = y[:, 0] * float(r["aux_log10e_std"]) + float(r["aux_log10e_mean"])
    return y[:, 3], log_e


def binned_auc(trues, preds, x, edges, min_per_class=20):
    """AUC (+ bootstrap 68% band) in bins of x; NaN where a class has < min_per_class events."""
    rng = np.random.default_rng(0)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        t, p = trues[m], preds[m]
        if min((t == 1).sum(), (t == 0).sum()) < min_per_class:
            out.append((np.nan, np.nan, np.nan, int(m.sum())))
            continue
        boots = []
        for _ in range(200):
            i = rng.integers(0, len(t), len(t))
            if len(np.unique(t[i])) == 2:
                boots.append(roc_auc_score(t[i], p[i]))
        lo_b, hi_b = np.percentile(boots, [16, 84])
        out.append((roc_auc_score(t, p), lo_b, hi_b, int(m.sum())))
    return np.array(out)


def plot_auc_vs_truth(runs, path):
    """runs: list of (label, r, color) with y_aux. AUC per true cos-zenith and log10 E bin."""
    cz_edges = np.linspace(-1, 1, 9)
    all_le = np.concatenate([truth_kinematics(r)[1] for _, r, _ in runs])
    le_edges = np.quantile(all_le, np.linspace(0, 1, 7))
    le_edges[-1] += 1e-9
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, edges, k, xlabel in [(axes[0], cz_edges, 0, "true cos zenith"),
                                 (axes[1], le_edges, 1, "true log10 primary energy [GeV]")]:
        centers = 0.5 * (edges[1:] + edges[:-1])
        for i, (label, r, color) in enumerate(runs):
            x = truth_kinematics(r)[k]
            res = binned_auc(r["trues"], r["preds"], x, edges)
            offset = (i - (len(runs) - 1) / 2) * 0.012 * (edges[-1] - edges[0])
            ax.errorbar(centers + offset, res[:, 0], yerr=[res[:, 0] - res[:, 1], res[:, 2] - res[:, 0]],
                        fmt="o-", color=color, markersize=6, capsize=3, linewidth=1.8, label=label)
        ax.axhline(0.5, color=INK_2, linestyle=":", linewidth=1)
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("validation AUC within bin")
    axes[0].set_title("AUC per true direction bin", loc="left")
    axes[1].set_title("AUC per true energy bin", loc="left")
    axes[0].legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_aux(r, label, path):
    """Network's auxiliary predictions vs truth: log10 E and direction opening angle."""
    y, a = r["y_aux"], r["aux_pred"]
    mu, sd = float(r["aux_log10e_mean"]), float(r["aux_log10e_std"])
    true_le, pred_le = y[:, 0] * sd + mu, a[:, 0] * sd + mu
    d_pred = a[:, 1:4] / np.maximum(np.linalg.norm(a[:, 1:4], axis=1, keepdims=True), 1e-9)
    angle = np.degrees(np.arccos(np.clip((d_pred * y[:, 1:4]).sum(axis=1), -1, 1)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    lims = np.percentile(true_le, [0.5, 99.5])
    h = ax1.hist2d(true_le, pred_le, bins=40, range=[lims, lims], cmin=1, cmap="Blues")
    ax1.plot(lims, lims, color=INK_2, linestyle=":", linewidth=1)
    fig.colorbar(h[3], ax=ax1, label="events")
    ax1.set_xlabel("true log10 E"); ax1.set_ylabel("predicted log10 E")
    ax1.set_title(f"{label}: energy (σ = {np.std(pred_le - true_le):.3f} in log10 E)", loc="left", fontsize=10)
    for lab, col, name in [(1, TAU_COLOR, "nutau"), (0, NUE_COLOR, "nue")]:
        ax2.hist(angle[r["trues"] == lab], bins=np.linspace(0, 180, 37), histtype="step", linewidth=2,
                 density=True, color=col, label=f"{name} (median {np.median(angle[r['trues'] == lab]):.0f}°)")
    ax2.set_xlabel("angle between predicted and true direction [deg]")
    ax2.set_ylabel("density")
    ax2.set_title(f"{label}: direction error", loc="left", fontsize=10)
    ax2.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return float(np.std(pred_le - true_le)), float(np.median(angle))


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

    rows, curves, truth_runs = [], [], []
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
        kin = truth_kinematics(r)
        if kin is not None:
            truth_runs.append((label, r, color))
            row["auc_mean_in_coszen_bins"] = round(np.nanmean(binned_auc(trues, preds, kin[0], np.linspace(-1, 1, 9))[:, 0]), 4)
            le_edges = np.quantile(kin[1], np.linspace(0, 1, 7)); le_edges[-1] += 1e-9
            row["auc_mean_in_log10e_bins"] = round(np.nanmean(binned_auc(trues, preds, kin[1], le_edges)[:, 0]), 4)
        if "aux_pred" in r:
            row["aux_log10e_sigma"], row["aux_dir_median_deg"] = [
                round(v, 3) for v in plot_aux(r, label, os.path.join(run_dir, "aux_regression.png"))]
        rows.append(row)

    plot_roc(curves, "ROC comparison (validation)", os.path.join(args.out_dir, "roc_comparison.png"))
    if truth_runs:
        plot_auc_vs_truth(truth_runs, os.path.join(args.out_dir, "auc_vs_truth.png"))
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    print(summary.to_string(index=False))
    print(f"\nPer-run plots written into each run dir; comparison in {args.out_dir}")


if __name__ == "__main__":
    main()
