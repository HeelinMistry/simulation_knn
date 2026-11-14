import numpy as np, pandas as pd
import data.preprocessing as P

rng = np.random.default_rng(0)
n15 = 4 * 24 * 400                       # 400 days of 15m candles
idx = pd.date_range("2023-01-01", periods=n15, freq="15min")
ret = rng.standard_normal(n15) * 0.002
close = 0.5 * np.exp(np.cumsum(ret))
open_ = np.r_[close[0], close[:-1]]
hi = np.maximum(open_, close) * (1 + np.abs(rng.standard_normal(n15)) * 0.001)
lo = np.minimum(open_, close) * (1 - np.abs(rng.standard_normal(n15)) * 0.001)
vol = rng.lognormal(10, 0.5, n15)
raw15 = pd.DataFrame({"Open_time": idx.strftime(P.OPEN_TIME_FMT), "Open": open_, "High": hi,
                      "Low": lo, "Close": close, "Volume": vol,
                      "Taker_buy_base_vol": vol * rng.uniform(0.4, 0.6, n15)})
raw15["Quote_asset_volume"] = raw15["Volume"] * raw15["Close"]
raw15["Taker_buy_quote_vol"] = raw15["Taker_buy_base_vol"] * raw15["Close"]

def resample(df, rule):
    d = df.copy(); d.index = pd.to_datetime(d["Open_time"])
    o = d.resample(rule).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last",
                              "Volume": "sum", "Taker_buy_base_vol": "sum",
                              "Quote_asset_volume": "sum", "Taker_buy_quote_vol": "sum"}).dropna()
    o.insert(0, "Open_time", o.index.strftime(P.OPEN_TIME_FMT))
    return o.reset_index(drop=True)

frames = {"15m": raw15, "1h": resample(raw15, "1h"), "4h": resample(raw15, "4h")}
print(P.window_table().to_string(index=False))
for tf, raw in frames.items():
    print(f"\n=== {tf}: {len(raw):,} candles")
    f = P.compute_features(raw, label=tf, timeframe=tf)
    print("feature rows:", len(f), "| cols:", [c for c in f.columns if c.endswith("_Scaled")])
    rep = P.verify_feature_contract(raw.iloc[-6000:] if tf == "15m" else raw, timeframe=tf)
    print("contract ok:", rep["ok"].tolist())
    t = P.compute_targets(raw, timeframe=tf, label=tf)
    print("k =", t.attrs["k"], "| reward stats:", t[["Reward_long", "Reward_short"]].describe().loc[["mean", "std", "min", "max"]].round(3).to_dict())
    print("hash differs across tf:", P.feature_hashes(timeframe=tf)["RSI_Scaled"])
# next_open variant and gap handling
g = frames["4h"].drop(index=range(1000, 1004)).reset_index(drop=True)
t = P.compute_targets(g, "4h", entry="next_open", label="gap")
print("rows NaN around gap:", int(t["Reward_long"].iloc[990:1000].isna().sum()), "(expect 4 = k=6 rows before gap whose t+6 spans it... see below)")
print(t["Reward_long"].iloc[992:1002].isna().tolist())