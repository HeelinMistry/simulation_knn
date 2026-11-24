"""
train_ensemble.py  —  multi-seed ensemble + walk-forward evaluation of the universal policy.

Reuses everything from train_policy.py (model, loss, stateful backtest) and adds:

 1. SEED ENSEMBLE   train N independent members (different init, batch order, sampled
                    positions); the ensemble's output logits are averaged, THEN the legal-action
                    mask + argmax is applied. Each member is early-stopped / checkpointed on its
                    own val Sharpe, exactly like train_policy.py.
 2. WALK-FORWARD    roll the cutoff forward; per fold train on data before val_cutoff, validate
                    on [val_cutoff, cutoff), test on [cutoff, cutoff + test window). Test
                    anchors must have their target window end before the test end (embargo = k).

Single fold (identical to the original command, just with several seeds):

    python train_ensemble.py --cutoff 2025-06-01 --val-cutoff 2025-01-01 --epochs 30 \
        --use-tf-emb --seeds 0 1 2 3 4

Walk-forward (3-month test windows, stepping 3 months, 6-month validation block):

    python train_ensemble.py --walk-forward --wf-first-cutoff 2025-06-01 --wf-end 2026-11-01 \
        --wf-step-months 3 --wf-test-months 3 --wf-val-months 6 --epochs 30 \
        --use-tf-emb --seeds 0 1 2 --buy-threshold 0.6

Output layout (out_dir/fold_<cutoff>/):
    seed_<s>/best.pt, history.csv      per-member checkpoint + curves
    ensemble.pt                        all member weights + args (reload with load_ensemble())
    backtest_per_series.csv            ensemble vs buy-and-hold, per series, per split
    summary.json                       ensemble metrics + per-member metrics, per split
and out_dir/walk_forward_summary.csv / .json across folds.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import train_policy as tp
from data.torch_dataset import (
    action_mask, apply_action_mask, load_frames, make_dataloaders,
    make_splits, n_actions, to_device,
)


# ══════════════════════════════════════════════════════════════════════════════
# Ensemble wrapper — drop-in replacement for a single model
# ══════════════════════════════════════════════════════════════════════════════

class EnsemblePolicy(nn.Module):
    """
    Same call signature as UniversalTradingPolicy, so tp.backtest / tp.eval_loss /
    tp.predict_actions work on it unchanged.

    mode="logits": mean of member logits (default).
    mode="probs" : mean of member softmax over LEGAL actions, returned as log-probs.
    """

    def __init__(self, members, mode: str = "logits", allow_short: bool = False):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.mode, self.allow_short = mode, allow_short

    def forward(self, x, tf_id, position):
        outs = [m(x, tf_id, position).float() for m in self.members]
        if self.mode == "logits":
            return torch.stack(outs).mean(0)
        mask = action_mask(position, self.allow_short)
        p = torch.stack([F.softmax(apply_action_mask(o, mask), dim=-1) for o in outs]).mean(0)
        return torch.log(p.clamp_min(1e-12))


def build_model(args, num_features: int, device) -> tp.UniversalTradingPolicy:
    return tp.UniversalTradingPolicy(
        num_features, n_actions=n_actions(args.allow_short), d_model=args.d_model,
        dropout=args.dropout, use_tf_emb=args.use_tf_emb).to(device)


def load_ensemble(path: str, device=None) -> EnsemblePolicy:
    """Reload an ensemble.pt written by this script."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(path, map_location=device)
    a = argparse_namespace(ck["args"])
    members = []
    for sd in ck["members"]:
        m = build_model(a, ck["num_features"], device)
        m.load_state_dict(sd)
        members.append(m.eval())
    return EnsemblePolicy(members, ck["mode"], a.allow_short).to(device).eval()


def argparse_namespace(d: dict):
    import argparse
    return argparse.Namespace(**d)


# ══════════════════════════════════════════════════════════════════════════════
# One ensemble member (same loop as train_policy.main)
# ══════════════════════════════════════════════════════════════════════════════

def train_member(args, seed: int, frames: dict, splits: dict, out_dir: str, device):
    """Train one seed. Returns (best_model, loaders, datasets, history_df, best_score)."""
    margs = copy.copy(args)
    margs.seed = seed
    tp.set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    loaders, datasets, sampler = make_dataloaders(
        frames, splits, batch_size=margs.batch_size, lookback_hours=margs.lookback_hours,
        allow_short=margs.allow_short, p_in_position=margs.p_in_position,
        num_workers=margs.num_workers, seed=seed)

    model = build_model(margs, datasets["train"].num_features, device)
    opt = torch.optim.AdamW(model.parameters(), lr=margs.lr, weight_decay=margs.weight_decay)
    total_steps = max(1, margs.epochs * len(loaders["train"]))
    warm = max(1, int(margs.warmup_frac * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=margs.amp and device.type == "cuda")

    ckpt = os.path.join(out_dir, "best.pt")
    best, bad, history = -float("inf"), 0, []
    for epoch in range(margs.epochs):
        t0 = time.time()
        datasets["train"].set_epoch(epoch)
        sampler.set_epoch(epoch)
        tr = tp.train_one_epoch(model, loaders["train"], opt, sched, scaler, device, margs)
        va = tp.eval_loss(model, loaders["val"], device, margs)
        vs = tp.summarise(tp.backtest(model, datasets["val"], frames, device, margs, "val"))
        score = vs["mean_strat_sharpe"] if margs.select == "val_sharpe" else -va["loss"]
        if not np.isfinite(score):
            score = -float("inf")
        history.append({"epoch": epoch, **tr, "val_loss": va["loss"],
                        "val_exp_value": va["exp_value"],
                        "val_strat_sharpe": vs["mean_strat_sharpe"],
                        "val_bh_sharpe": vs["mean_bh_sharpe"],
                        "val_time_in_market": vs["mean_time_in_market"],
                        "sec": time.time() - t0})
        flag = ""
        if score > best:
            best, bad, flag = score, 0, " ★"
            torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt)
        else:
            bad += 1
        print(f"  [seed {seed}] ep {epoch:3d} | train {tr['train_loss']:.4f} | "
              f"val loss {va['loss']:.4f} | val Sharpe {vs['mean_strat_sharpe']:+.2f} "
              f"vs B&H {vs['mean_bh_sharpe']:+.2f} | {history[-1]['sec']:.0f}s{flag}")
        if bad >= margs.patience:
            print(f"  [seed {seed}] early stop")
            break

    hist = pd.DataFrame(history)
    hist.to_csv(os.path.join(out_dir, "history.csv"), index=False)
    if not os.path.exists(ckpt):          # every epoch scored -inf (e.g. NaN Sharpe): keep last weights
        torch.save({"model": model.state_dict(), "epoch": len(history) - 1}, ckpt)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    return model.eval(), loaders, datasets, hist, best


# ══════════════════════════════════════════════════════════════════════════════
# One fold: train all seeds, evaluate members + ensemble
# ══════════════════════════════════════════════════════════════════════════════

def make_fold_splits(args, cutoff: str, val_cutoff: str, test_end, train_start):
    splits = make_splits(args.train_symbols, args.test_symbols, args.train_tfs, args.test_tfs,
                         cutoff=cutoff, val_cutoff=val_cutoff)
    for name in ("test_coin", "test_tf", "test_both"):
        if name in splits and test_end:
            splits[name] = dataclasses.replace(splits[name], end=test_end)
    if train_start:
        splits["train"] = dataclasses.replace(splits["train"], start=train_start)
    return splits


def run_fold(args, frames, cutoff, val_cutoff, test_end, train_start, fold_dir, device) -> list:
    os.makedirs(fold_dir, exist_ok=True)
    splits = make_fold_splits(args, cutoff, val_cutoff, test_end, train_start)
    print("\n" + "#" * 78 +
          f"\nFOLD  train < {val_cutoff} | val [{val_cutoff}, {cutoff}) | "
          f"test [{cutoff}, {test_end or 'end'})   seeds={args.seeds}\n" + "#" * 78)

    members, scores, eval_loaders, eval_datasets = [], [], None, None
    for seed in args.seeds:
        m, loaders, datasets, _, best = train_member(
            args, seed, frames, splits, os.path.join(fold_dir, f"seed_{seed}"), device)
        members.append(m)
        scores.append(best)
        if eval_loaders is None:                 # evaluation data is seed-independent; keep the first
            eval_loaders, eval_datasets = loaders, datasets

    ens = EnsemblePolicy(members, args.ens_mode, args.allow_short).to(device).eval()
    torch.save({"members": [m.state_dict() for m in members], "mode": args.ens_mode,
                "args": vars(args), "num_features": eval_datasets["train"].num_features,
                "feature_cols": eval_datasets["train"].feature_cols,
                "cutoff": cutoff, "val_cutoff": val_cutoff, "test_end": test_end},
               os.path.join(fold_dir, "ensemble.pt"))

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:+.3f}")
    tabs, summary, rows = [], {}, []
    for name in ("val", "test_coin", "test_tf", "test_both"):
        if name not in eval_datasets:
            continue
        ds, ld = eval_datasets[name], eval_loaders[name]
        tab = tp.backtest(ens, ds, frames, device, args, name)
        tabs.append(tab)
        ens_sum = {**tp.summarise(tab), **tp.eval_loss(ens, ld, device, args)}

        mem_sh = []
        for m in members:
            mem_sh.append(tp.summarise(tp.backtest(m, ds, frames, device, args, name))["mean_strat_sharpe"])
        summary[name] = {
            "ensemble": ens_sum,
            "members_mean_sharpe": mem_sh,
            "members_avg": float(np.nanmean(mem_sh)),
            "members_std": float(np.nanstd(mem_sh)),
            "ensemble_minus_member_avg": ens_sum["mean_strat_sharpe"] - float(np.nanmean(mem_sh)),
        }
        print(f"\n[{name}]")
        print(tab.drop(columns="split").to_string(index=False))
        print("  ensemble:", {k: round(v, 3) for k, v in ens_sum.items()})
        print(f"  members : mean {summary[name]['members_avg']:+.3f} "
              f"std {summary[name]['members_std']:.3f}  "
              f"| ensemble - member avg {summary[name]['ensemble_minus_member_avg']:+.3f}")
        rows.append({"fold_cutoff": cutoff, "val_cutoff": val_cutoff, "test_end": test_end,
                     "split": name, "n_series": len(tab),
                     "ens_sharpe": ens_sum["mean_strat_sharpe"],
                     "bh_sharpe": ens_sum["mean_bh_sharpe"],
                     "ens_minus_bh_median": ens_sum["median_sharpe_diff"],
                     "frac_beating_bh": ens_sum["frac_series_beating_bh"],
                     "time_in_market": ens_sum["mean_time_in_market"],
                     "members_sharpe_avg": summary[name]["members_avg"],
                     "members_sharpe_std": summary[name]["members_std"]})

    pd.concat(tabs).to_csv(os.path.join(fold_dir, "backtest_per_series.csv"), index=False)
    with open(os.path.join(fold_dir, "summary.json"), "w") as f:
        json.dump({"member_val_scores": scores, **summary}, f, indent=2)
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Walk-forward fold generation
# ══════════════════════════════════════════════════════════════════════════════

def _d(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d")


def make_folds(args) -> list:
    """[(cutoff, val_cutoff, test_end, train_start)]"""
    if not args.walk_forward:
        if not args.cutoff or not args.val_cutoff:
            raise SystemExit("--cutoff and --val-cutoff are required (or use --walk-forward).")
        return [(args.cutoff, args.val_cutoff, None, None)]
    end = pd.Timestamp(args.wf_end) if args.wf_end else pd.Timestamp.today().normalize()
    c = pd.Timestamp(args.wf_first_cutoff)
    folds = []
    while True:
        test_end = min(c + pd.DateOffset(months=args.wf_test_months), end)
        if (test_end - c).days < args.wf_min_test_days:
            break
        train_start = (_d(c - pd.DateOffset(months=args.wf_train_months))
                       if args.wf_train_months else None)
        folds.append((_d(c), _d(c - pd.DateOffset(months=args.wf_val_months)), _d(test_end),
                      train_start))
        c = c + pd.DateOffset(months=args.wf_step_months)
    if not folds:
        raise SystemExit("No walk-forward folds fit between --wf-first-cutoff and --wf-end.")
    return folds


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args(argv=None):
    p = tp.build_parser()
    p.description = __doc__
    for a in p._actions:                      # cutoff is only mandatory outside walk-forward
        if a.dest == "cutoff":
            a.required = False
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                   help="one ensemble member per seed (3-5 recommended)")
    p.add_argument("--ens-mode", choices=["logits", "probs"], default="logits",
                   help="average member logits (default) or legal-action probabilities")
    p.add_argument("--walk-forward", action="store_true")
    p.add_argument("--wf-first-cutoff", default="2025-06-01", help="first fold's test start")
    p.add_argument("--wf-end", default=None, help="end of usable data (default: today)")
    p.add_argument("--wf-step-months", type=int, default=3)
    p.add_argument("--wf-test-months", type=int, default=3)
    p.add_argument("--wf-val-months", type=int, default=6,
                   help="validation block length before each cutoff")
    p.add_argument("--wf-train-months", type=int, default=0,
                   help="0 = expanding train window; N = rolling window of N months")
    p.add_argument("--wf-min-test-days", type=int, default=30,
                   help="drop a trailing fold whose test window is shorter than this")
    args = p.parse_args(argv)
    tp.check_args(args)
    return args


def main():
    args = parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("--seeds must be unique.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir or os.path.join("runs", "ens_" + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    folds = make_folds(args)
    print(f"device={device}  out_dir={out_dir}  seeds={args.seeds}  folds={len(folds)}")
    for c, v, e, s in folds:
        print(f"  fold: train {s or '-'} → {v} | val {v} → {c} | test {c} → {e or 'end'}")

    # frames contain the full history and are fold-independent: load once
    first = make_fold_splits(args, *folds[0][:2], folds[0][2], folds[0][3])
    syms = sorted({s for sp in first.values() for s in sp.symbols})
    tfs = sorted({t for sp in first.values() for t in sp.timeframes})
    frames = load_frames(syms, tfs, frozen=True, horizon_hours=args.horizon_hours,
                         fee_bps=args.fee_bps, slippage_bps=args.slippage_bps)

    all_rows = []
    for cutoff, val_cutoff, test_end, train_start in folds:
        all_rows += run_fold(args, frames, cutoff, val_cutoff, test_end, train_start,
                             os.path.join(out_dir, f"fold_{cutoff}"), device)
        pd.DataFrame(all_rows).to_csv(os.path.join(out_dir, "walk_forward_summary.csv"), index=False)

    wf = pd.DataFrame(all_rows)
    print("\n" + "=" * 78 + "\nWALK-FORWARD SUMMARY (ensemble)\n" + "=" * 78)
    print(wf.drop(columns=["val_cutoff"]).to_string(index=False))
    agg = {}
    for name, g in wf.groupby("split"):
        agg[name] = {"folds": int(len(g)),
                     "mean_ens_sharpe": float(g["ens_sharpe"].mean()),
                     "mean_bh_sharpe": float(g["bh_sharpe"].mean()),
                     "frac_folds_beating_bh": float((g["ens_sharpe"] > g["bh_sharpe"]).mean()),
                     "worst_fold_ens_sharpe": float(g["ens_sharpe"].min()),
                     "mean_ens_minus_member_avg": float((g["ens_sharpe"] - g["members_sharpe_avg"]).mean())}
    print("\nacross folds:")
    for k, v in agg.items():
        print(f"  {k}: {({a: round(b, 3) for a, b in v.items()})}")
    with open(os.path.join(out_dir, "walk_forward_summary.json"), "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()

# single fold, 5 seeds, no coin identity
# python train_ensemble.py --cutoff 2025-06-01 --val-cutoff 2025-01-01 --epochs 30 --use-tf-emb --seeds 0 1 2 3 4
#
# walk-forward through today (Nov 2026): 3-month tests, stepping 3 months
# python train_ensemble.py --walk-forward --wf-first-cutoff 2025-06-01 --wf-end 2026-11-01 \
#     --wf-step-months 3 --wf-test-months 3 --wf-val-months 6 --epochs 30 \
#     --use-tf-emb --seeds 0 1 2 --buy-threshold 0.6 --out-dir runs/wf_thr60