import numpy as np, pandas as pd, torch
import data.preprocessing as P
import data.torch_dataset as D

rng = np.random.default_rng(0)
STEP = {"15m": 15, "1h": 60, "4h": 240}            # minutes
K = {"15m": 96, "1h": 24, "4h": 6}                 # 24 h horizon
DAYS = 400
FEATS = P.feature_columns(default_only=True)


def fake_frame(tf, seed):
    r = np.random.default_rng(seed)
    n = DAYS * 24 * 60 // STEP[tf] - K[tf]
    ts = pd.date_range("2023-01-01", periods=n, freq=f"{STEP[tf]}min")
    keep = np.ones(n, bool)
    keep[1000:1004] = False                         # inject a data gap
    df = pd.DataFrame(r.uniform(-1, 1, (n, len(FEATS))), columns=FEATS)
    df.insert(0, "Open_time", ts.strftime(P.OPEN_TIME_FMT))
    df["Fwd_ret"] = r.normal(0, 0.02, n)
    df["Target_vol"] = 0.01
    df["Reward_long"] = r.normal(0, 1, n).clip(-5, 5)
    df["Reward_short"] = r.normal(0, 1, n).clip(-5, 5)
    df = df[keep].reset_index(drop=True)
    df.attrs["k"] = K[tf]
    return df


train_syms = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT")
test_syms = ("XRPUSDT", "LTCUSDT")
frames = {(s, tf): fake_frame(tf, hash((s, tf)) % 2**32)
          for i, s in enumerate(train_syms + test_syms) for tf in ("15m", "1h", "4h")}

CUT, VCUT = "2023-11-01", "2023-09-01"
splits = D.make_splits(cutoff=CUT, val_cutoff=VCUT)
loaders, datasets, sampler = D.make_dataloaders(frames, splits, batch_size=128, lookback_hours=96)

# 1. split disjointness: coins / timeframes
tr = datasets["train"]
assert {s.symbol for s in tr.series} == set(train_syms) and {s.timeframe for s in tr.series} == {"1h", "4h"}
assert {s.symbol for s in datasets["test_coin"].series} == set(test_syms)
assert {s.timeframe for s in datasets["test_tf"].series} == {"15m"}
assert {(s.symbol, s.timeframe) for s in datasets["test_both"].series} == {(c, "15m") for c in test_syms}

# 2. no target leakage across the cutoff (embargo) and no test anchors before it
cut_s = D._to_seconds(CUT); vcut_s = D._to_seconds(VCUT)
for i in np.random.default_rng(1).integers(0, len(tr), 2000):
    s = tr.series[tr.sid[i]]; assert s.ts[tr.t[i]] + K[s.timeframe] * STEP[s.timeframe] * 60 <= vcut_s
for name in ("test_coin", "test_tf", "test_both"):
    ds = datasets[name]; assert all(ds.series[a].ts[b] >= cut_s for a, b in zip(ds.sid[::50], ds.t[::50]))

# 3. windows are contiguous: none spans the injected gap
for name, ds in datasets.items():
    for sid, t in zip(ds.sid[::37], ds.t[::37]):
        s = ds.series[sid]; w = s.ts[t - s.L + 1: t + 1]
        assert (np.diff(w) == STEP[s.timeframe] * 60).all()

# 4. batches: homogeneous timeframe, correct shapes, masks obey the rules
tr.set_epoch(0); sampler.set_epoch(0)
seen_tfs, n = set(), 0
for b in loaders["train"]:
    assert b["tf_id"].unique().numel() == 1
    seen_tfs.add(int(b["tf_id"][0]))
    L = D.windows_for(D.TIMEFRAMES[int(b["tf_id"][0])]).bars(96.0)
    assert b["x"].shape == (128, L, len(FEATS)), b["x"].shape
    flat = b["position"] == 0
    m = b["action_mask"]
    assert m[:, D.DO_NOTHING].all()
    assert (m[:, D.BUY] == flat).all() and (m[:, D.CLOSE] == ~flat).all()
    assert (b["action_values"][~m] == 0).all()
    n += 1
    if n >= 40: break
assert seen_tfs == {D.TF_IDS["1h"], D.TF_IDS["4h"]}
print("mask / shape / homogeneity checks passed;", len(loaders["train"]), "train batches per epoch")

# 5. determinism + epoch reshuffle of positions
p0 = tr.positions.copy(); tr.set_epoch(0); assert (p0 == tr.positions).all()
tr.set_epoch(1); assert (p0 != tr.positions).any()

# 6. short mode
ds_s = D.MultiTFWindowDataset(frames, splits["test_coin"], lookback=32, allow_short=True)
b = D.WindowCollate(True)([ds_s[i] for i in range(64)])
assert b["action_mask"].shape[-1] == 4 and set(b["position"].tolist()) <= {-1, 0, 1}
assert (b["action_mask"][:, D.SELL] == (b["position"] == 0)).all()

# 7. device transfer
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tr.set_epoch(0)
b = D.to_device(next(iter(loaders["train"])), dev)
print("device:", dev, "| keys:", sorted(b))
print("ALL OK")