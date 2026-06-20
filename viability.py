import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/XRPUSDT_master_processed.csv")
df['Open_time'] = pd.to_datetime(df['Open_time'], format="mixed")

# Resample to 4h
df_4h = df.set_index('Open_time').resample('4h')['Close'].ohlc()
df_4h['return'] = df_4h['close'].pct_change().abs()

print(f"15m median abs return: {df['Close'].pct_change().abs().median():.4%}")
print(f"4h  median abs return: {df_4h['return'].median():.4%}")
print(f"Commission as % of 15m move: {0.0003 / df['Close'].pct_change().abs().median():.1%}")
print(f"Commission as % of 4h move:  {0.0003 / df_4h['return'].median():.1%}")