"""
main_hmm.py
─────────────────────────────────────────────────────────────
Pipeline runner for Hidden Markov Models using hmmlearn and the
existing data_manager infrastructure.
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from data.data_manager import update_master_data, SYMBOLS

# ── Configuration ─────────────────────────────────────────────────────────────
TARGET_SYMBOL = "BTCUSDT"  # Choose any symbol from your data pool
TARGET_TIMEFRAME = "4h"  # "15m", "1h", or "4h"
N_STATES = 4  # Number of hidden market states to discover
RANDOM_STATE = 42


def run_hmm_pipeline():
    print(f"\n{'=' * 60}\n  HMM PIPELINE: {TARGET_SYMBOL} [{TARGET_TIMEFRAME}]\n{'=' * 60}")

    # 1. Load data using your existing data_manager
    try:
        df = update_master_data(timeframe=TARGET_TIMEFRAME, symbol=TARGET_SYMBOL)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    # 2. Select scaled indicator features for training
    feature_cols = [
        "RSI_Scaled", "MACD_Scaled", "BB_Scaled",
        "OBV_Scaled", "ATR_Scaled", "MeanDev_Scaled"
    ]

    # Drop any remaining rows with NaNs
    model_data = df.dropna(subset=feature_cols).copy()
    X = model_data[feature_cols].values

    print(f"  Training GaussianHMM with {N_STATES} states on {len(X):,} candles...")

    # 3. Initialize and fit the Gaussian HMM
    # Full covariance allows the model to capture complex directional correlations between indicators
    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type="full",
        n_iter=200,
        random_state=RANDOM_STATE
    )
    model.fit(X)

    # 4. Decode the hidden states for the entire historical sequence
    hidden_states = model.predict(X)
    model_data["Hidden_State"] = hidden_states

    # 5. Output the transition probability matrix
    # This matrix answers: "Given the market is in State i, what is the probability it moves to State j?"
    print(f"\n  ✓ HMM Training Complete!")
    print(f"\n  [Transition Probability Matrix (A)]")
    print(f"  Rows = Current State, Columns = Next State")

    transmat_df = pd.DataFrame(
        model.transmat_,
        index=[f"State_{i}" for i in range(N_STATES)],
        columns=[f"State_{j}" for j in range(N_STATES)]
    )
    print(transmat_df.round(4).to_string())

    # 6. Interpret states by checking average indicator values per state
    print(f"\n  [State Characterisation (Mean Feature Values)]")
    state_means = model_data.groupby("Hidden_State")[feature_cols].mean()
    print(state_means.round(3).to_string())

    # Save labeled states back out if needed
    output_filename = f"data/processed/{TARGET_SYMBOL}_{TARGET_TIMEFRAME}_hmm_states.csv"
    model_data.to_csv(output_filename, index=False)
    print(f"\n  ✓ Saved state-labeled dataset to {output_filename}")


if __name__ == "__main__":
    run_hmm_pipeline()