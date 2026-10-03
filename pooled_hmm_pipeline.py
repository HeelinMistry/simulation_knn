"""
pooled_hmm_pipeline.py
─────────────────────────────────────────────────────────────
Trains a single pooled GaussianHMM across multiple cryptocurrency symbols
using sequence lengths, and maps observations back to their generalized states.
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from data.data_manager import update_master_data, SYMBOLS


def run_pooled_hmm():
    feature_cols = [
        "RSI_Scaled", "MACD_Scaled", "BB_Scaled",
        "OBV_Scaled", "ATR_Scaled", "MeanDev_Scaled"
    ]

    all_X = []
    sequence_lengths = []
    coin_metadata = []  # Keeps track of which row belongs to which coin/timestamp

    print(f"\n{'=' * 60}\n  POOLED HMM PIPELINE: Loading {len(SYMBOLS)} symbols\n{'=' * 60}")

    # 1. Independent loading to avoid boundary pollution, aggregate for pooled training
    for symbol in SYMBOLS:
        try:
            df = update_master_data(timeframe="4h", symbol=symbol)
            clean_data = df.dropna(subset=feature_cols).copy()

            if clean_data.empty:
                continue

            X_coin = clean_data[feature_cols].values
            all_X.append(X_coin)
            sequence_lengths.append(len(X_coin))

            # Store metadata for mapping states back to specific assets later
            clean_data["Symbol"] = symbol
            coin_metadata.append(clean_data[["Open_time", "Symbol", "Close"]])

            print(f"  ✓ Loaded {symbol}: {len(X_coin):,} candles")
        except FileNotFoundError:
            print(f"  ⚠ Skipping {symbol}: Master data not found.")

    if not all_X:
        print("❌ Error: No valid data found for any symbol.")
        return

    # Stack all observations into a massive unified matrix
    X_pooled = np.vstack(all_X)
    master_meta_df = pd.concat(coin_metadata, ignore_index=True)

    print(
        f"\n  Fitting Pooled GaussianHMM on {len(X_pooled):,} total observations across {len(sequence_lengths)} assets...")

    # 2. Fit the Pooled HMM using 'lengths'
    n_states = 4
    model = GaussianHMM(
        n_components=4,
        covariance_type="diag",  # Changed from "full" to "diag"
        n_iter=300,
        random_state=42
    )
    model.fit(X_pooled, lengths=sequence_lengths)

    # 3. Decode the hidden states across the entire pooled dataset
    pooled_hidden_states = model.predict(X_pooled)
    master_meta_df["Hidden_State"] = pooled_hidden_states
    master_meta_df[feature_cols] = X_pooled

    # 4. Display Generalized Transition Matrix
    print(f"\n  [Universal Transition Probability Matrix (A)]")
    transmat_df = pd.DataFrame(
        model.transmat_,
        index=[f"State_{i}" for i in range(n_states)],
        columns=[f"State_{j}" for j in range(n_states)]
    )
    print(transmat_df.round(4).to_string())

    # 5. Universal State Characterization
    print(f"\n  [Universal State Characterisation (Mean Feature Values Across All Coins)]")
    state_means = master_meta_df.groupby("Hidden_State")[feature_cols].mean()
    print(state_means.round(3).to_string())

    # Save the generalized cross-asset dataset
    output_path = "data/processed/universal_crypto_pooled_hmm_states.csv"
    master_meta_df.to_csv(output_path, index=False)
    print(f"\n  ✓ Saved universal state-labeled dataset to {output_path}")


if __name__ == "__main__":
    run_pooled_hmm()