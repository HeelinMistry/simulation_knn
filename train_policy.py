"""
train_policy.py  —  train + evaluate the universal trading policy.

Run from the project root (the folder that contains data/):

    python train_policy.py --cutoff 2025-06-01 --val-cutoff 2025-01-01

PIPELINE
────────
 1. load_frames -> make_splits -> make_dataloaders        (data/torch_dataset.py)
 2. train a causal dilated-TCN policy on the soft, masked, reward-weighted target
 3. every epoch: val loss, val expected action-value and val STRATEGY SHARPE
    (stateful backtest) -> early stopping + best checkpoint
 4. final: backtest on val / test_coin / test_tf / test_both against buy-and-hold,
    per series + aggregate, written to CSV/JSON in the run directory

FIXES vs. the Phase-3 sketch
────────────────────────────
 * The sketch never fed `position` to the network, yet action legality and action
   values depend on it. The model now takes it as an input.
 * Soft target was softmax over ALL actions, including illegal ones, and
   log_softmax of masked logits is ~ -3e38 there -> exploding loss. Target is now
   softmax over legal actions only.
 * Symbol / timeframe embeddings are OFF by default. Unseen coins / timeframes would
   hit an untrained embedding row at test time, which defeats zero-shot evaluation
   (features are already hour-denominated and per-asset scaled). Opt in with
   --use-symbol-emb / --use-tf-emb.
 * Receptive field: dilated causal convolutions (1,2,4,8 => 31 candles) instead of
   two k=3 layers (5 candles).

BACKTEST CONVENTIONS (matches the targets: entry="close")
─────────────────────────────────────────────────────────
 * Decision at close of candle t, executed at C_t, earns C_{t+1}/C_t - 1 on the
   next candle. Position state is carried forward (real stateful rollout, not the
   randomly sampled positions used in training).
 * Cost = (fee_bps + slippage_bps)/1e4 per unit of position change, so a round trip
   costs the same 2x(fee+slip) used in the reward.
 * Windows with a data gap are flattened at the gap (cost charged).
 * Buy-and-hold: long every anchor candle, one entry + one exit cost per
   contiguous segment, i.e. it pays costs too.
 * Sharpe = mean/std of per-candle returns * sqrt(candles per year); rf = 0.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.preprocessing import timeframe_hours
from data.torch_dataset import (
    BUY, CLOSE, SELL, SYMBOL_IDS, TF_IDS, TIMEFRAMES,
    action_mask, apply_action_mask, load_frames, make_dataloaders,
    make_splits, n_actions, to_device,
)


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════

class CausalBlock(nn.Module):
    """Residual block of two left-padded dilated convs (output t sees inputs <= t only)."""

    def __init__(self, d: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(d, d, kernel, dilation=dilation)
        self.conv2 = nn.Conv1d(d, d, kernel, dilation=dilation)
        self.norm = nn.LayerNorm(d)          # over channels, per time step (causal)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # [B, C, L]
        h = F.gelu(self.conv1(F.pad(x, (self.pad, 0))))
        h = self.conv2(F.pad(self.drop(h), (self.pad, 0)))
        x = x + h
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class UniversalTradingPolicy(nn.Module):
    def __init__(self, num_features: int, n_actions: int = 3, d_model: int = 128,
                 dilations=(1, 2, 4, 8), kernel: int = 3, dropout: float = 0.1,
                 use_tf_emb: bool = False, use_symbol_emb: bool = False,
                 num_timeframes: int = len(TIMEFRAMES), num_symbols: int = len(SYMBOL_IDS),
                 seen_symbols=(), sym_drop: float = 0.3):
        super().__init__()
        self.use_tf_emb, self.use_symbol_emb = use_tf_emb, use_symbol_emb
        self.proj = nn.Linear(num_features, d_model)
        self.blocks = nn.ModuleList(CausalBlock(d_model, kernel, d, dropout) for d in dilations)
        self.pos_emb = nn.Embedding(3, 16)                       # position -1/0/+1
        ctx = 16
        if use_tf_emb:
            self.tf_emb = nn.Embedding(num_timeframes, 16)
            ctx += 16
        if use_symbol_emb:
            # last row = UNK. Coins not in `seen_symbols` always map to UNK, and during training
            # a fraction `sym_drop` of samples is also mapped to UNK, so UNK is a trained row and
            # unseen coins (XRP/LTC) do not hit a random-init embedding.
            self.sym_emb = nn.Embedding(num_symbols + 1, 16)
            seen = torch.zeros(num_symbols, dtype=torch.bool)
            if len(seen_symbols):
                seen[list(seen_symbols)] = True
            self.register_buffer("seen_sym", seen)
            self.unk, self.sym_drop = num_symbols, float(sym_drop)
            ctx += 16
        self.head = nn.Sequential(
            nn.Linear(2 * d_model + ctx, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, n_actions))

    def forward(self, x, tf_id, position, sym_id=None):
        """x [B,L,F], tf_id [B], position [B] in {-1,0,1}, sym_id [B] -> logits [B,A]."""
        h = self.proj(x).transpose(1, 2)                         # [B, d, L]
        for blk in self.blocks:
            h = blk(h)
        feats = [h[:, :, -1], h.mean(dim=2), self.pos_emb(position + 1)]
        if self.use_tf_emb:
            feats.append(self.tf_emb(tf_id))
        if self.use_symbol_emb:
            sid = sym_id.clamp(min=0)
            unk = torch.full_like(sid, self.unk)
            sid = torch.where(self.seen_sym[sid] & (sym_id >= 0), sid, unk)
            if self.training and self.sym_drop > 0:
                sid = torch.where(torch.rand(sid.shape, device=sid.device) < self.sym_drop, unk, sid)
            feats.append(self.sym_emb(sid))
        return self.head(torch.cat(feats, dim=-1))


# ══════════════════════════════════════════════════════════════════════════════
# Loss
# ══════════════════════════════════════════════════════════════════════════════

def policy_loss(logits: torch.Tensor, batch: dict, temp: float) -> torch.Tensor:
    """Cross-entropy to softmax(action_values / temp) over LEGAL actions only."""
    mask = batch["action_mask"]
    logits = logits.float()
    neg = torch.finfo(logits.dtype).min
    logp = F.log_softmax(logits.masked_fill(~mask, neg), dim=-1)
    tgt = F.softmax((batch["action_values"] / temp).masked_fill(~mask, neg), dim=-1)
    return -(tgt * logp).sum(-1).mean()


def expected_value(logits: torch.Tensor, batch: dict) -> torch.Tensor:
    """Mean reward the (stochastic) policy would collect on these states."""
    p = F.softmax(apply_action_mask(logits.float(), batch["action_mask"]), dim=-1)
    return (p * batch["action_values"]).sum(-1).mean()


def _forward(model, batch, use_sym):
    return model(batch["x"], batch["tf_id"], batch["position"], batch["sym_id"])


# ══════════════════════════════════════════════════════════════════════════════
# Backtest
# ══════════════════════════════════════════════════════════════════════════════

STATES = {False: [0, 1], True: [0, 1, -1]}


@torch.no_grad()
def predict_actions(model, ds, device, allow_short: bool, amp: bool, bs: int = 2048) -> dict:
    """
    Greedy (argmax over legal actions) action for EVERY anchor and EVERY position state,
    batched. The sequential state machine then runs on these tables, so the backtest needs
    no per-step forward pass. Returns {series_index: (anchor_rows, acts[n, n_states])}.
    """
    model.eval()
    states = STATES[allow_short]
    out = {}
    for j, s in enumerate(ds.series):
        rows = ds.t[ds.sid == j]
        if rows.size == 0:
            continue
        X = s.X.to(device)
        offs = torch.arange(-s.L + 1, 1, device=device)
        acts = np.zeros((rows.size, len(states)), dtype=np.int64)
        for a in range(0, rows.size, bs):
            tt = torch.from_numpy(rows[a:a + bs]).to(device)
            x = X[tt[:, None] + offs]
            tf = torch.full((len(tt),), TF_IDS[s.timeframe], dtype=torch.long, device=device)
            sy = torch.full((len(tt),), SYMBOL_IDS.get(s.symbol, 0), dtype=torch.long, device=device)
            for si, p in enumerate(states):
                pos = torch.full((len(tt),), p, dtype=torch.long, device=device)
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=amp and device.type == "cuda"):
                    logits = model(x, tf, pos, sy)
                logits = apply_action_mask(logits.float(), action_mask(pos, allow_short))
                acts[a:a + bs, si] = logits.argmax(-1).cpu().numpy()
        out[j] = (rows, acts)
    return out


def simulate(rows, acts, ts, close, step: int, cost_side: float, allow_short: bool):
    """Stateful rollout. Returns (strategy_rets, buyhold_rets, n_entries, frac_in_market)."""
    sidx = {s: i for i, s in enumerate(STATES[allow_short])}
    n = len(rows)
    strat, bh = np.zeros(n), np.zeros(n)
    pos, bh_pos, entries, in_mkt = 0, 0, 0, 0
    for i in range(n):
        t = int(rows[i])
        a = int(acts[i, sidx[pos]])
        new = pos
        if a == BUY:
            new = 1
        elif a == SELL and allow_short:
            new = -1
        elif a == CLOSE:
            new = 0
        cost = abs(new - pos) * cost_side
        entries += int(new != pos and new != 0)

        nxt_ok = t + 1 < len(close) and (ts[t + 1] - ts[t]) == step
        r = (close[t + 1] / close[t] - 1.0) if nxt_ok else 0.0
        cont = i + 1 < n and int(rows[i + 1]) == t + 1 and nxt_ok       # segment continues

        bh_cost = abs(1 - bh_pos) * cost_side
        in_mkt += int(new != 0)
        s_r, b_r = new * r - cost, r - bh_cost
        if not cont:                                   # gap / end of data: flatten both
            s_r -= abs(new) * cost_side
            b_r -= cost_side
        strat[i], bh[i] = s_r, b_r
        pos, bh_pos = (new, 1) if cont else (0, 0)
    return strat, bh, entries, in_mkt / max(n, 1)


def perf(rets: np.ndarray, ppy: float) -> dict:
    if rets.size < 2:
        return {"sharpe": float("nan"), "total_return": float("nan"), "max_dd": float("nan")}
    sd = rets.std(ddof=1)
    equity = np.cumprod(1.0 + rets)
    return {
        "sharpe": float(rets.mean() / sd * math.sqrt(ppy)) if sd > 1e-12 else float("nan"),
        "total_return": float(equity[-1] - 1.0),
        "max_dd": float((equity / np.maximum.accumulate(equity) - 1.0).min()),
    }


def backtest(model, ds, frames, device, args, label: str = "") -> pd.DataFrame:
    """Per-series strategy vs buy-and-hold table."""
    preds = predict_actions(model, ds, device, args.allow_short, args.amp)
    cost_side = (args.fee_bps + args.slippage_bps) / 1e4
    recs = []
    for j, (rows, acts) in preds.items():
        s = ds.series[j]
        df = frames[(s.symbol, s.timeframe)].reset_index(drop=True)
        close = df["Close"].to_numpy(np.float64)
        step = int(round(timeframe_hours(s.timeframe) * 3600))
        st, bh, entries, frac = simulate(rows, acts, s.ts, close, step, cost_side, args.allow_short)
        ppy = 365.0 * 24.0 / timeframe_hours(s.timeframe)
        ps, pb = perf(st, ppy), perf(bh, ppy)
        recs.append({
            "split": label, "symbol": s.symbol, "timeframe": s.timeframe, "candles": len(rows),
            "strat_sharpe": ps["sharpe"], "bh_sharpe": pb["sharpe"],
            "strat_return": ps["total_return"], "bh_return": pb["total_return"],
            "strat_maxdd": ps["max_dd"], "bh_maxdd": pb["max_dd"],
            "entries": entries, "time_in_market": frac,
        })
    return pd.DataFrame(recs)


def summarise(tab: pd.DataFrame) -> dict:
    if tab.empty:
        return {}
    beat = float((tab["strat_sharpe"] > tab["bh_sharpe"]).mean())
    return {"mean_strat_sharpe": float(tab["strat_sharpe"].mean()),
            "mean_bh_sharpe": float(tab["bh_sharpe"].mean()),
            "median_sharpe_diff": float((tab["strat_sharpe"] - tab["bh_sharpe"]).median()),
            "frac_series_beating_bh": beat,
            "mean_time_in_market": float(tab["time_in_market"].mean())}


# ══════════════════════════════════════════════════════════════════════════════
# Train / eval loops
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, opt, sched, scaler, device, args) -> dict:
    model.train()
    tot, n = 0.0, 0
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=args.amp and device.type == "cuda"):
            logits = _forward(model, batch, args.use_symbol_emb)
        loss = policy_loss(logits, batch, args.temp)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        scaler.step(opt)
        scaler.update()
        sched.step()
        tot += loss.item() * len(batch["x"])
        n += len(batch["x"])
    return {"train_loss": tot / max(n, 1)}


@torch.no_grad()
def eval_loss(model, loader, device, args) -> dict:
    model.eval()
    ls, ev, n = 0.0, 0.0, 0
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=args.amp and device.type == "cuda"):
            logits = _forward(model, batch, args.use_symbol_emb)
        b = len(batch["x"])
        ls += policy_loss(logits, batch, args.temp).item() * b
        ev += expected_value(logits, batch).item() * b
        n += b
    return {"loss": ls / max(n, 1), "exp_value": ev / max(n, 1)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = make_splits.__defaults__  # noqa: F841  (defaults documented in torch_dataset)
    p.add_argument("--train-symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT"])
    p.add_argument("--test-symbols", nargs="+", default=["XRPUSDT", "LTCUSDT"])
    p.add_argument("--train-tfs", nargs="+", default=["15m", "1h", "4h"])
    p.add_argument("--test-tfs", nargs="*", default=[],
                   help="held-out timeframes (default none: 15m is now trained on)")
    p.add_argument("--cutoff", required=True, help="test period starts here, e.g. 2025-06-01")
    p.add_argument("--val-cutoff", default=None, help="validation period [val-cutoff, cutoff)")
    p.add_argument("--lookback-hours", type=float, default=96.0)
    p.add_argument("--horizon-hours", type=float, default=24.0)
    p.add_argument("--fee-bps", type=float, default=10.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--allow-short", action="store_true")
    p.add_argument("--p-in-position", type=float, default=0.5)
    # model
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--use-tf-emb", action="store_true")
    p.add_argument("--use-symbol-emb", action="store_true")
    p.add_argument("--sym-drop", type=float, default=0.3,
                   help="prob. of replacing the symbol id by UNK during training")
    # optimisation
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=5e-2)
    p.add_argument("--warmup-frac", type=float, default=0.15)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--temp", type=float, default=0.5, help="soft-target temperature (lower = sharper)")
    p.add_argument("--select", choices=["val_sharpe", "val_loss"], default="val_sharpe")
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--amp", action="store_true", help="fp16 autocast on CUDA")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir or os.path.join("runs", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"device={device}  out_dir={out_dir}")

    if args.select == "val_sharpe" and args.val_cutoff is None:
        print("  ⚠ no --val-cutoff: falling back to selecting on train loss is not meaningful; "
              "pass --val-cutoff. Using select=val_loss on a missing val set will fail.")
        raise SystemExit("--val-cutoff is required for early stopping / checkpoint selection.")

    # ── data ──────────────────────────────────────────────────────────────────
    splits = make_splits(args.train_symbols, args.test_symbols, args.train_tfs, args.test_tfs,
                         cutoff=args.cutoff, val_cutoff=args.val_cutoff)
    syms = sorted({s for sp in splits.values() for s in sp.symbols})
    tfs = sorted({t for sp in splits.values() for t in sp.timeframes})
    frames = load_frames(syms, tfs, frozen=True, horizon_hours=args.horizon_hours,
                         fee_bps=args.fee_bps, slippage_bps=args.slippage_bps)
    loaders, datasets, sampler = make_dataloaders(
        frames, splits, batch_size=args.batch_size, lookback_hours=args.lookback_hours,
        allow_short=args.allow_short, p_in_position=args.p_in_position,
        num_workers=args.num_workers, seed=args.seed)

    # ── model / optimiser ─────────────────────────────────────────────────────
    model = UniversalTradingPolicy(
        datasets["train"].num_features, n_actions=n_actions(args.allow_short), d_model=args.d_model,
        dropout=args.dropout, use_tf_emb=args.use_tf_emb, use_symbol_emb=args.use_symbol_emb,
        seen_symbols=[SYMBOL_IDS[s] for s in args.train_symbols if s in SYMBOL_IDS],
        sym_drop=args.sym_drop).to(device)
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(loaders["train"]))
    warm = max(1, int(args.warmup_frac * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    # ── train ─────────────────────────────────────────────────────────────────
    ckpt_path = os.path.join(out_dir, "best.pt")
    best, bad, history = -float("inf"), 0, []
    for epoch in range(args.epochs):
        t0 = time.time()
        datasets["train"].set_epoch(epoch)
        sampler.set_epoch(epoch)
        tr = train_one_epoch(model, loaders["train"], opt, sched, scaler, device, args)
        va = eval_loss(model, loaders["val"], device, args)
        vt = backtest(model, datasets["val"], frames, device, args, "val")
        vs = summarise(vt)
        score = vs["mean_strat_sharpe"] if args.select == "val_sharpe" else -va["loss"]
        if not np.isfinite(score):
            score = -float("inf")
        row = {"epoch": epoch, **tr, "val_loss": va["loss"], "val_exp_value": va["exp_value"],
               "val_strat_sharpe": vs["mean_strat_sharpe"], "val_bh_sharpe": vs["mean_bh_sharpe"],
               "val_time_in_market": vs["mean_time_in_market"], "lr": opt.param_groups[0]["lr"],
               "sec": time.time() - t0}
        history.append(row)
        pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)
        flag = ""
        if score > best:
            best, bad, flag = score, 0, "  ★ best"
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch,
                        "num_features": datasets["train"].num_features,
                        "feature_cols": datasets["train"].feature_cols}, ckpt_path)
        else:
            bad += 1
        print(f"ep {epoch:3d} | train {tr['train_loss']:.4f} | val loss {va['loss']:.4f} "
              f"EV {va['exp_value']:+.4f} | val Sharpe strat {vs['mean_strat_sharpe']:+.2f} "
              f"vs B&H {vs['mean_bh_sharpe']:+.2f} | in-mkt {vs['mean_time_in_market']:.0%} "
              f"| {row['sec']:.0f}s{flag}")
        if bad >= args.patience:
            print(f"early stopping (no improvement for {args.patience} epochs)")
            break

    # ── final evaluation with the best checkpoint ─────────────────────────────
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    print("\n" + "=" * 78 + f"\nFINAL EVALUATION (best epoch by {args.select})\n" + "=" * 78)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:+.3f}")
    all_tabs, summary = [], {}
    for name in ("val", "test_coin", "test_tf", "test_both"):
        if name not in datasets:
            continue
        tab = backtest(model, datasets[name], frames, device, args, name)
        all_tabs.append(tab)
        summary[name] = {**summarise(tab), **eval_loss(model, loaders[name], device, args)}
        print(f"\n[{name}]")
        print(tab.drop(columns="split").to_string(index=False))
        print("  summary:", {k: round(v, 3) for k, v in summary[name].items()})
    pd.concat(all_tabs).to_csv(os.path.join(out_dir, "backtest_per_series.csv"), index=False)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {ckpt_path}, history.csv, backtest_per_series.csv, summary.json in {out_dir}")


if __name__ == "__main__":
    main()

#  python train_policy.py --cutoff 2025-06-01 --val-cutoff 2025-01-01 --epochs 30 --use-symbol-emb --use-tf-emb