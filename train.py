import argparse
import copy
import importlib
import inspect
import json
import os
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t).pow(self.gamma)
        loss = focal_weight * bce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def aux_loss_fn(pred, target):
    """MSE between predicted and true auxiliary targets (energy, direction),
    ignoring NaN targets (events without truth)."""
    mask = torch.isfinite(target)
    if not mask.any():
        return pred.sum() * 0.0
    return ((pred - torch.nan_to_num(target)) ** 2)[mask].mean()


def model_supports_aux(model):
    return "return_aux" in inspect.signature(model.forward).parameters


def evaluate_with_aux(model, loader, criterion, device):
    """Validation pass. Returns classification loss, AUC, scores, labels, the
    auxiliary-regression loss (nan without an aux head), the per-event truth
    targets y_aux (None if the dataset has none) and the aux predictions."""
    from sklearn.metrics import roc_auc_score
    model.eval()
    with_aux = model_supports_aux(model)
    preds, trues, y_aux, aux_pred = [], [], [], []
    total_loss, total_aux = 0.0, 0.0

    with torch.no_grad():
        for batch in loader:
            batch   = batch.to(device)
            logits, aux = model(batch, return_aux=True) if with_aux else (model(batch), None)
            targets = batch.y.view(-1)
            total_loss += criterion(logits, targets).item() * batch.num_graphs
            if aux is not None:
                total_aux += aux_loss_fn(aux, batch.y_aux).item() * batch.num_graphs
                aux_pred.append(aux.cpu().numpy())
            if "y_aux" in batch:
                y_aux.append(batch.y_aux.cpu().numpy())
            preds.append(torch.sigmoid(logits).cpu().numpy())
            trues.append(targets.cpu().numpy())

    n = len(loader.dataset)
    preds, trues = np.concatenate(preds), np.concatenate(trues)
    auc = roc_auc_score(trues, preds) if len(np.unique(trues)) >= 2 else float("nan")
    return (total_loss / n, auc, preds, trues,
            total_aux / n if aux_pred else float("nan"),
            np.concatenate(y_aux) if y_aux else None,
            np.concatenate(aux_pred) if aux_pred else None)


def train(args):
    # flush every printed line so per-epoch stats reach the log file live
    # (stdout is block-buffered when redirected, e.g. in SLURM jobs)
    sys.stdout.reconfigure(line_buffering=True)
    t_start = time.time()

    # trainer module supplies the dataset/graph building and the model, e.g.
    # trainer_noedge (default) or trainer_domenc (per-DOM pulse encoder + edges)
    tm = importlib.import_module(args.trainer)
    N_PCA_FEATURES        = tm.N_PCA_FEATURES
    set_seed              = tm.set_seed
    build_geo_dict_kdtree = tm.build_geo_dict_kdtree
    load_or_build_dataset = tm.load_or_build_dataset
    HybridGNNClassifier   = tm.HybridGNNClassifier
    print(f"Trainer module: {args.trainer}")
    if args.aux_weight > 0 and not hasattr(tm, "N_AUX"):
        sys.exit(f"ERROR: --aux_weight needs a trainer with auxiliary truth targets "
                 f"(trainer, trainer_domenc); {args.trainer} has none")

    set_seed(42)

    if not os.path.exists(args.geo):
        print(f"ERROR: geometry file not found: {args.geo}", file=sys.stderr)
        sys.exit(1)

    geo = pd.read_csv(args.geo)
    kd_tree, string_ids_geo = build_geo_dict_kdtree(geo)
    print(f"Geometry: {len(geo)} DOMs, KDTree built")

    dataset = load_or_build_dataset(
        cache_dir                 = args.cache_dir,
        nue_dbs                   = args.nue_dbs,
        tau_dbs                   = args.tau_dbs,
        kd_tree                   = kd_tree,
        string_ids_geo            = string_ids_geo,
        max_events_per_class      = args.max_events,
        charge_threshold          = args.charge_threshold,
        **({"min_dom_charge": args.min_dom_charge} if args.min_dom_charge > 0 else {}),
        **({"min_strings": args.min_strings} if args.min_strings != 2 else {}),
    )
    if args.build_only:
        print(f"--build_only: dataset with {len(dataset)} events is cached; exiting")
        return

    indices = np.arange(len(dataset))
    labels  = np.array([dataset[i].y.item() for i in range(len(dataset))])

    train_idx, val_idx = train_test_split(
        indices, test_size=args.val_frac, stratify=labels, random_state=42
    )

    train_ds = dataset[train_idx.tolist()]
    val_ds   = dataset[val_idx.tolist()]
    print(f"Split data:  train: {len(train_ds)}  val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_kwargs = {"use_pca": False} if args.no_pca else {}  # trainer and trainer_domenc support use_pca
    if args.use_image or args.image_only:
        if "use_image" not in inspect.signature(HybridGNNClassifier.__init__).parameters:
            sys.exit(f"ERROR: --use_image/--image_only need trainer_domenc; {args.trainer} has no image branch")
        model_kwargs["use_image"] = True
        model_kwargs["use_gnn"] = not args.image_only
        print("Brightest-string image branch: " + ("image only (no GNN)" if args.image_only else "GNN + image"))
    gat_opts = {"readout": args.gat_readout, "residual": args.gat_residual, "bright_feats": args.bright_feats,
                "extra_edges": args.extra_edges, "full_graph": args.full_graph,
                "time_unit_ns": args.time_unit_ns if args.time_unit_ns > 0 else None, "pulse_act": args.pulse_act}
    non_default = (args.gat_readout != "sum" or args.gat_residual or args.bright_feats or args.extra_edges
                   or args.full_graph or args.time_unit_ns > 0 or args.pulse_act != "relu")
    if non_default:
        if "readout" not in inspect.signature(HybridGNNClassifier.__init__).parameters:
            sys.exit(f"ERROR: GAT-branch options need trainer_domenc; {args.trainer} does not support them")
        model_kwargs.update(gat_opts)
        print(f"GAT-branch options: {gat_opts}")
    use_aux = args.aux_weight > 0
    if use_aux:
        model_kwargs["n_aux"] = tm.N_AUX
        print(f"Auxiliary truth targets {tm.AUX_TARGET_NAMES}, weight {args.aux_weight}")
    model = HybridGNNClassifier(gnn_hidden_dim=64, pca_dim=N_PCA_FEATURES, fusion_dim=32,
                                **model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    if args.lr_scheduler == "plateau":
        # multiply LR by lr_factor when val loss hasn't improved for lr_patience epochs
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.lr_min)
    else:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)

    n_pos      = max(int((labels[train_idx] == 1).sum()), 1)
    n_neg      = max(int((labels[train_idx] == 0).sum()), 1)
    pos_weight = torch.tensor([n_neg / n_pos], device=device, dtype=torch.float32)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # n_pos      = max(int((labels[train_idx] == 1).sum()), 1)
    # n_neg      = max(int((labels[train_idx] == 0).sum()), 1)
    # pos_weight = torch.tensor([n_neg / n_pos], device=device, dtype=torch.float32)
    # criterion  = FocalLoss(gamma=3.0, pos_weight=None)

    best_val_auc   = -np.inf
    best_state     = None
    best_epoch     = -1
    patience       = args.patience
    epochs_no_gain = 0
    os.makedirs(args.save_dir, exist_ok=True)

    train_losses = []
    val_losses   = []
    val_aucs     = []
    lrs          = []
    train_auxs   = []
    val_auxs     = []
    start_epoch  = 0

    # checkpoint / resume: state is saved after every epoch, so a run can span several jobs
    ckpt_path = os.path.join(args.save_dir, "checkpoint.pt")
    if args.resume:
        if os.path.exists(os.path.join(args.save_dir, "eval_val.npz")):
            print(f"--resume: {args.save_dir} has already finished (eval_val.npz exists); nothing to do")
            return
        if os.path.exists(ckpt_path):
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            optimizer.load_state_dict(ck["optimizer"])
            scheduler.load_state_dict(ck["scheduler"])
            start_epoch, best_val_auc, best_epoch = ck["epoch"], ck["best_val_auc"], ck["best_epoch"]
            best_state, epochs_no_gain = ck["best_state"], ck["epochs_no_gain"]
            train_losses, val_losses, val_aucs = ck["train_losses"], ck["val_losses"], ck["val_aucs"]
            lrs, train_auxs, val_auxs = ck["lrs"], ck["train_auxs"], ck["val_auxs"]
            print(f"--resume: continuing after epoch {start_epoch} (best AUC {best_val_auc:.4f} at epoch {best_epoch})")
        else:
            print(f"--resume: no checkpoint in {args.save_dir}; starting from epoch 1")

    stopped_early = epochs_no_gain >= patience
    for epoch in range(start_epoch, args.epochs):
        if stopped_early:
            break
        t_epoch = time.time()
        model.train()
        train_loss = 0.0
        train_aux  = 0.0

        pbar = tqdm(train_loader, desc=f"Train {epoch+1:03d}", unit="batch", leave=False)
        for batch in pbar:
            batch = batch.to(device)

            optimizer.zero_grad()
            targets = batch.y.view(-1)
            if use_aux:
                logits, aux = model(batch, return_aux=True)
                cls_loss = criterion(logits, targets)
                a_loss   = aux_loss_fn(aux, batch.y_aux)
                loss     = cls_loss + args.aux_weight * a_loss
                train_aux += a_loss.item() * batch.num_graphs
            else:
                logits   = model(batch)
                cls_loss = criterion(logits, targets)
                loss     = cls_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += cls_loss.item() * batch.num_graphs   # classification part (comparable across runs)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= len(train_loader.dataset)
        train_aux  /= len(train_loader.dataset)

        val_loss, val_auc, _, _, val_aux, _, _ = evaluate_with_aux(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aucs.append(val_auc if not np.isnan(val_auc) else 0.0)
        train_auxs.append(train_aux if use_aux else np.nan)
        val_auxs.append(val_aux)

        lr_epoch = optimizer.param_groups[0]["lr"]  # LR used for this epoch
        lrs.append(lr_epoch)
        if args.lr_scheduler == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_gain = 0
        else:
            epochs_no_gain += 1

        val_auc_str = f"{val_auc:.4f}" if not np.isnan(val_auc) else "nan"
        aux_str = f" | Aux train/val: {train_aux:.4f}/{val_aux:.4f}" if use_aux else ""
        print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | AUC: {val_auc_str} "
              f"| LR: {lr_epoch:.2e} | Best: {best_val_auc:.4f}{aux_str}")

        stopped_early = epochs_no_gain >= patience
        tmp = ckpt_path + ".tmp"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch + 1,
                    "best_val_auc": best_val_auc, "best_epoch": best_epoch, "best_state": best_state,
                    "epochs_no_gain": epochs_no_gain, "train_losses": train_losses, "val_losses": val_losses,
                    "val_aucs": val_aucs, "lrs": lrs, "train_auxs": train_auxs, "val_auxs": val_auxs}, tmp)
        os.replace(tmp, ckpt_path)   # atomic: a job killed mid-write keeps the previous checkpoint

        if stopped_early:
            print(f"Early stopping at epoch {epoch+1} (best epoch {best_epoch})")
            break
        if args.max_hours > 0:
            elapsed, last = time.time() - t_start, time.time() - t_epoch
            if elapsed + 1.3 * last > args.max_hours * 3600:
                print(f"Time limit: {elapsed / 3600:.2f} h used, next epoch (~{last / 60:.1f} min) would exceed "
                      f"--max_hours {args.max_hours}; checkpoint saved after epoch {epoch+1}, continue with --resume")
                return

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), os.path.join(args.save_dir, "hybrid_weights.pt"))
    joblib.dump(dataset.pca_scaler, os.path.join(args.save_dir, "pca_scaler.joblib"))

    # validation scores of the best model + training history, for plot_eval.py
    _, _, val_preds, val_trues, _, val_y_aux, val_aux_pred = evaluate_with_aux(model, val_loader, criterion, device)
    extra = {}
    if val_y_aux is not None:      # true (std log10 E, direction) of each val event: AUC per energy/zenith bin
        extra["y_aux"] = val_y_aux
    if val_aux_pred is not None:   # network's energy/direction predictions
        extra["aux_pred"] = val_aux_pred
    np.savez(os.path.join(args.save_dir, "eval_val.npz"),
             preds=val_preds, trues=val_trues,
             train_losses=np.array(train_losses), val_losses=np.array(val_losses),
             val_aucs=np.array(val_aucs), lrs=np.array(lrs), best_epoch=best_epoch,
             train_auxs=np.array(train_auxs), val_auxs=np.array(val_auxs),
             aux_log10e_mean=getattr(tm, "AUX_LOG10E_MEAN", np.nan),
             aux_log10e_std=getattr(tm, "AUX_LOG10E_STD", np.nan), **extra)
    with open(os.path.join(args.save_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\nBest val AUC: {best_val_auc:.4f} (epoch {best_epoch})")

    fig, ax = plt.subplots(figsize=(6, 4), facecolor="black")
    ax.plot(train_losses, color="cyan", label="Train")
    ax.plot(val_losses, color="orange", label="Val")
    ax.set_xlabel("Epoch", color="white")
    ax.set_ylabel("Loss", color="white")
    ax.set_title("Training Curves", color="white")
    ax.legend(facecolor="black", labelcolor="white")
    ax.set_facecolor("black")
    ax.tick_params(colors="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "training_curves.png"), dpi=150, facecolor="black")
    plt.close()

    print(f"Saved model, scaler, and plot to {args.save_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training for PCA+GNN Tau vs Electron Neutrino Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--trainer", default="trainer_noedge",
                        help="Trainer module to import (trainer_noedge, trainer, trainer_domenc)")
    parser.add_argument("--aux_weight", type=float, default=0.0,
                        help="Weight of the auxiliary truth-regression loss (true log10 E + direction "
                             "as extra targets; trainer, trainer_domenc). 0 = off")
    parser.add_argument("--use_image", action="store_true",
                        help="Add the brightest-string image CNN branch (trainer_domenc)")
    parser.add_argument("--image_only", action="store_true",
                        help="Classifier on the brightest-string image only, no GNN (trainer_domenc)")
    parser.add_argument("--gat_readout", choices=["sum", "multi"], default="sum",
                        help="GAT readout: sum (original) or multi = sum+mean+max+attention+brightest DOM (trainer_domenc)")
    parser.add_argument("--gat_residual", action="store_true", help="Residual connections around the GAT layers (trainer_domenc)")
    parser.add_argument("--bright_feats", action="store_true",
                        help="Add 5 node features relative to the brightest DOM (trainer_domenc)")
    parser.add_argument("--extra_edges", action="store_true",
                        help="Add fine-time and causality-residual edge features (trainer_domenc)")
    parser.add_argument("--time_unit_ns", type=float, default=0.0,
                        help="Express fine time differences as asinh(dt / unit), e.g. 3.3 = detector time "
                             "resolution; 0 = original scalings (trainer_domenc)")
    parser.add_argument("--pulse_act", choices=["relu", "gelu"], default="relu",
                        help="Activation in the per-DOM pulse encoder (trainer_domenc)")
    parser.add_argument("--full_graph", action="store_true",
                        help="Fully connected DOM graph per event instead of k-NN (trainer_domenc)")
    parser.add_argument("--resume", action="store_true",
                        help="Continue from <save_dir>/checkpoint.pt (exits if the run already finished)")
    parser.add_argument("--max_hours", type=float, default=0.0,
                        help="Stop cleanly (checkpoint kept) before this wall time [h]; 0 = no limit")
    parser.add_argument("--build_only", action="store_true",
                        help="Build/cache the dataset and exit (e.g. one job before parallel trainings)")
    parser.add_argument("--no_pca", action="store_true",
                        help="Drop the PCA branch from the classifier (trainer, trainer_domenc)")
    parser.add_argument("--tau_dbs", nargs="+", required=True, help="Tau DB paths")
    parser.add_argument("--nue_dbs", nargs="+", required=True, help="Nue DB paths")
    parser.add_argument("--geo", default="/path/to/geometry_clean.csv", help="Geometry file")
    parser.add_argument("--max_events", type=int, default=None, help="Max events per class")
    parser.add_argument("--charge_threshold", type=float, default=0.1, help="Charge threshold")
    parser.add_argument("--min_strings", type=int, default=2,
                        help="Minimum strings with surviving DOMs per event (2 = original; 1 keeps single-string "
                             "events -- use with --no_pca; trainer, trainer_domenc)")
    parser.add_argument("--min_dom_charge", type=float, default=0.0,
                        help="Drop DOMs whose summed charge in the event is below this [PE] (0 = off; trainer, trainer_domenc)")
    parser.add_argument("--save_dir", default="./output_hybrid", help="Output directory")
    parser.add_argument("--cache_dir", default="./dataset_cache", help="Directory to cache the built dataset")
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--val_frac", type=float, default=0.15, help="Validation fraction")
    parser.add_argument("--patience", type=int, default=35, help="Early stopping patience")
    parser.add_argument("--lr_min", type=float, default=1e-6, help="Minimum LR (plateau and cosine schedulers)")
    parser.add_argument("--lr_scheduler", choices=["plateau", "exp"], default="plateau",
                        help="plateau: ReduceLROnPlateau on val loss; exp: x0.98 every epoch (old default)")
    parser.add_argument("--lr_factor", type=float, default=0.5, help="Plateau scheduler: LR multiplier on plateau")
    parser.add_argument("--lr_patience", type=int, default=5,
                        help="Plateau scheduler: epochs without val-loss improvement before reducing LR")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)