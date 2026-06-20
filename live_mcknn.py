"""
live_mcknn.py
───────────────
Visual backtest + live inference runner for the MC-kNN trading agent.

Mapping vs live.py
─────────────────────
EVERYTHING in this file is structurally identical to live.py except:
  - import agents.sac_agent.SACAgent  -> agents.mc_knn_agent.MCKNNAgent
  - CHECKPOINT_PATH default .pt -> .npz
  - _load_agent(): no .eval()/.requires_grad_(False) calls (nothing to
    freeze — there are no parameters)
  - LiveWindow._draw_panel()'s "Entropy" row label/threshold semantics
    are unchanged in code but now reflect vote-share entropy rather than
    a temperature-controlled softmax entropy (see mc_knn_policy.py note)

Two modes (selected by --mode flag): identical to live.py.
  backtest   Fetch recent historical candles from Binance, run the loaded
             agent through them tick-by-tick in an arcade window.
  live       Fetch the same warm-up history, then poll Binance every 4
             hours for the next closed candle, run one inference step,
             log the action and probabilities, and update the display.
             No orders are placed — observation only.

Usage
─────
  python live_mcknn.py                          # backtest mode (default)
  python live_mcknn.py --mode backtest --symbol XRPUSDT
  python live_mcknn.py --mode live     --symbol BTCUSDT

Dependencies
────────────
  pip install arcade requests pandas numpy
  (no torch dependency for the agent itself — numpy only)
"""

import argparse
import logging
import os
import time
from collections import deque
from datetime import datetime, timezone

import arcade
import numpy as np
import pandas as pd
import requests

from agents.mc_knn_agent     import MCKNNAgent
from agents.unified_executor import UnifiedExecutor
from data.preprocessing      import preprocess_indicators_data

# ─────────────────────────────────────────────────────────────────────────────
# Configuration  (must match main_mcknn.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

BINANCE_API_URL  = "https://api.binance.com/api/"
CANDLE_INTERVAL  = "4h"
CANDLE_SECONDS   = 4 * 60 * 60      # 14 400 s per candle

WARMUP_CANDLES   = 600              # 600 × 4 h ≈ 100 days — satisfies rolling(200)
WARMUP_IDX       = 128              # must match main_mcknn.py WARMUP_IDX

FEATURES         = ["RSI_Scaled", "MACD_Scaled", "BB_Scaled",
                    "OBV_Scaled", "ATR_Scaled", "MeanDev_Scaled"]
NUM_INDICATORS   = len(FEATURES)
PACES     = (1, 6, 42, 90)
STATE_DIM = (NUM_INDICATORS * 2 * len(PACES)) + 2
ACTION_DIM       = 4
ACTION_NAMES     = {0: "LONG", 1: "SHORT", 2: "CLOSE", 3: "HOLD"}
K_NEIGHBORS      = 25

CHECKPOINT_PATH  = "outcomes/mc_knn_agent_best.npz"
LOG_DIR          = "outcomes/live_logs_mcknn"
OUTCOME_DIR      = "outcomes"

os.makedirs(LOG_DIR,     exist_ok=True)
os.makedirs(OUTCOME_DIR, exist_ok=True)

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(LOG_DIR, "live_session.log"), encoding="utf-8"
        ),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Binance helpers — IDENTICAL to live.py
# ─────────────────────────────────────────────────────────────────────────────

def _parse_klines(data: list) -> pd.DataFrame:
    """Convert raw Binance kline list to a clean DataFrame."""
    df = pd.DataFrame(data, columns=[
        "Open_time", "Open", "High", "Low", "Close", "Volume",
        "Close_time", "Quote_volume", "Trades",
        "Taker_buy_base", "Taker_buy_quote", "Ignore",
    ])
    df[["Open", "High", "Low", "Close", "Volume"]] = \
        df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    df["Open_time"] = pd.to_datetime(df["Open_time"], unit="ms")
    return df


def get_candles(symbol: str, limit: int = WARMUP_CANDLES) -> pd.DataFrame | None:
    """
    Fetch the latest `limit` closed candles for symbol+USDT.
    Always drops the last row (currently forming candle).
    Returns None on any network error.
    """
    url = (f"{BINANCE_API_URL}v3/klines"
           f"?symbol={symbol}&interval={CANDLE_INTERVAL}&limit={limit + 1}")
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        df = _parse_klines(r.json())
        return df.iloc[:-1].reset_index(drop=True)   # drop the forming candle
    except Exception as exc:
        log.error(f"Binance fetch error: {exc}")
        return None


def apply_indicators(raw_df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Run preprocess_indicators_data on an in-memory DataFrame.
    Saves to a temp file (required by the existing function signature)
    then reads it back. Returns None if fewer than WARMUP_IDX+1 rows survive.
    """
    tmp = os.path.join(LOG_DIR, "_live_tmp.csv")
    try:
        raw_df.to_csv(tmp, index=False)
        result = preprocess_indicators_data(tmp, tmp)
        if len(result) < WARMUP_IDX + 2:
            log.error(
                f"Only {len(result)} rows after indicators — need ≥{WARMUP_IDX+2}. "
                "Increase WARMUP_CANDLES."
            )
            return None
        return result
    except Exception as exc:
        log.error(f"Indicator calculation failed: {exc}")
        return None
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# ─────────────────────────────────────────────────────────────────────────────
# Arcade window — IDENTICAL to live.py (no agent-specific code in here)
# ─────────────────────────────────────────────────────────────────────────────

# Colours
C_BG        = (15,  15,  26)
C_GRID      = (42,  42,  74)
C_PRICE     = (170, 170, 200)
C_LONG      = (46,  204, 113)
C_SHORT     = (231,  76,  60)
C_CLOSE     = (243, 156,  18)
C_HOLD      = (149, 165, 166)
C_PNL_POS   = (46,  204, 113, 80)
C_PNL_NEG   = (231,  76,  60, 80)
C_WHITE     = (255, 255, 255)
C_YELLOW    = (255, 215,   0)
C_PURPLE    = (127,  90, 240)
C_DIM       = (130, 130, 160)

WIN_W, WIN_H = 1280, 780

# Layout zones
CHART_X0, CHART_X1 = 60,  920
CHART_Y0, CHART_Y1 = 140, 520
PNL_Y0,   PNL_Y1   = 30,  120
PANEL_X              = 940


class LiveWindow(arcade.Window):
    """
    Handles both backtest (replay) and live (real-time poll) rendering.
    Shared draw logic; mode controls how ticks are generated.

    Identical to live.py's LiveWindow except the docstring above the
    Entropy row in _draw_panel notes that "probs" are vote-shares, not
    a calibrated softmax — see mc_knn_policy.py.
    """

    def __init__(
        self,
        df:           pd.DataFrame,
        executor:     UnifiedExecutor,
        symbol:       str,
        mode:         str = "backtest",      # "backtest" | "live"
    ):
        title = f"MC-kNN {'Backtest' if mode == 'backtest' else 'LIVE'}  —  {symbol}  4h"
        super().__init__(WIN_W, WIN_H, title, update_rate=1 / 60)

        self.df       = df
        self.executor = executor
        self.symbol   = symbol
        self.mode     = mode

        # Price range for Y-mapping (cached)
        self._price_min = float(df["Close"].min())
        self._price_max = float(df["Close"].max())

        # Chart geometry
        self._n_display = len(df)   # visible tick range; updated in live mode

        # Per-tick geometry for the price polyline
        self._price_pts: list[tuple[float, float]] = []

        # Signal markers: list of (x, y, action_idx)
        self._signals: deque = deque(maxlen=600)

        # PnL curve: list of (trade_idx, cumulative_pnl)
        self._pnl_points: list[tuple[int, float]] = []
        self._cum_pnl    = 0.0
        self._trade_count = 0

        # State tracking
        self.current_tick = WARMUP_IDX
        self.finished     = False
        self.saved        = False

        # Live-mode: next expected close time (UTC unix seconds)
        self._next_candle_ts: float = 0.0
        self._live_df_raw:    pd.DataFrame | None = None  # grows with new candles

        # Warm up aggregator
        indicators_arr = df[FEATURES].values.astype(np.float32)
        executor.aggregator.tick = 0
        executor.aggregator.warm_up_all(indicators_arr, WARMUP_IDX)
        log.info(f"Aggregator warmed up to index {WARMUP_IDX}")

        if mode == "live":
            # In live mode, the last closed candle gives us the expected
            # next close time.
            last_ts = df["Open_time"].iloc[-1]
            self._next_candle_ts = (
                last_ts.timestamp() + CANDLE_SECONDS
            )
            # Keep a copy of the raw candles so we can append and recalculate
            self._live_df_raw = df.copy()
            log.info(
                f"Live mode — next candle expected at "
                f"{datetime.fromtimestamp(self._next_candle_ts, tz=timezone.utc)}"
            )

    # ── Coordinate helpers ────────────────────────────────────────────────────

    def _px(self, tick: int) -> float:
        """Map tick index → screen X inside the chart zone."""
        n = max(self._n_display, 1)
        return CHART_X0 + (tick / n) * (CHART_X1 - CHART_X0)

    def _py(self, price: float) -> float:
        """Map price → screen Y inside the chart zone."""
        span = self._price_max - self._price_min + 1e-9
        return CHART_Y0 + ((price - self._price_min) / span) * (CHART_Y1 - CHART_Y0)

    def _pnl_y(self, pnl: float) -> float:
        """Map cumulative PnL → Y in the PnL strip at the bottom."""
        span = 0.05   # ±5% display range
        return (PNL_Y0 + PNL_Y1) / 2 + (pnl / span) * ((PNL_Y1 - PNL_Y0) / 2)

    # ── Tick step ─────────────────────────────────────────────────────────────

    def _step_tick(self):
        """Process one tick of data and record all visual outputs."""
        row  = self.df.iloc[self.current_tick]
        ind  = np.array([row[f] for f in FEATURES], dtype=np.float32)
        px   = self._px(self.current_tick)
        py   = self._py(float(row["Close"]))

        # Price line point
        self._price_pts.append((px, py))

        # MC-kNN inference (same call shape as the SAC version's
        # executor.step — see unified_executor.py: agent-agnostic)
        action, probs, realised_pnl, _state = self.executor.step(
            indicators=ind,
            price=float(row["Close"]),
            tick=self.current_tick,
        )

        # Track PnL
        if realised_pnl != 0.0:
            self._cum_pnl   += realised_pnl
            self._trade_count += 1
            self._pnl_points.append((self._trade_count, self._cum_pnl))

        # Record signal
        if action in (0, 1, 2):
            self._signals.append((px, py, action))

        # Log every non-HOLD action
        if action != 3:
            status = self.executor.get_status()
            log.info(
                f"[{row['Open_time']}]  {ACTION_NAMES[action]:<5}  "
                f"price=${row['Close']:.4f}  "
                f"vote_shares={[f'{p:.2%}' for p in probs]}  "
                f"position={status['position']}  pnl={self._cum_pnl:.4%}"
            )

        self.current_tick += 1

    # ── Arcade update ─────────────────────────────────────────────────────────

    def on_update(self, delta_time: float):
        if self.mode == "backtest":
            self._update_backtest()
        else:
            self._update_live()

    def _update_backtest(self):
        """Replay: advance up to 8 ticks per frame for fast playback."""
        if self.current_tick >= len(self.df) - 1:
            if not self.saved:
                self._save_screenshot()
            return
        steps = min(8, len(self.df) - 1 - self.current_tick)
        for _ in range(steps):
            self._step_tick()

    def _update_live(self):
        """
        Live: check wall clock; when the next 4h candle should have closed,
        fetch new data from Binance, recompute indicators, and step one tick.
        """
        now = time.time()
        if now < self._next_candle_ts + 5:   # +5s grace for Binance propagation
            return

        log.info("Fetching new candle from Binance...")
        raw = get_candles(self.symbol, limit=WARMUP_CANDLES)
        if raw is None:
            log.warning("Fetch failed — will retry next cycle.")
            self._next_candle_ts += CANDLE_SECONDS
            return

        new_df = apply_indicators(raw)
        if new_df is None:
            self._next_candle_ts += CANDLE_SECONDS
            return

        # Replace df with freshly processed data; fast-forward current_tick
        old_len = len(self.df)
        self.df = new_df
        new_len = len(self.df)
        delta   = new_len - old_len

        if delta <= 0:
            log.warning("No new candles since last fetch.")
            self._next_candle_ts += CANDLE_SECONDS
            return

        # Update price range for correct Y mapping
        self._price_min = float(self.df["Close"].min())
        self._price_max = float(self.df["Close"].max())
        self._n_display = new_len

        # Rebuild the aggregator from scratch with the new full dataset
        ind_arr = self.df[FEATURES].values.astype(np.float32)
        self.executor.aggregator.tick = 0
        self.executor.aggregator.warm_up_all(ind_arr, WARMUP_IDX)

        # Advance current_tick to the newly added rows
        self.current_tick = new_len - delta
        for _ in range(delta):
            if self.current_tick < new_len - 1:
                self._step_tick()

        self._next_candle_ts += CANDLE_SECONDS * delta
        log.info(f"Stepped {delta} new candle(s).  Next at "
                 f"{datetime.fromtimestamp(self._next_candle_ts, tz=timezone.utc)}")

    # ── Draw ─────────────────────────────────────────────────────────────────

    def on_draw(self):
        self.clear()
        arcade.set_background_color(C_BG)

        self._draw_grid()
        self._draw_price_line()
        self._draw_pnl_strip()
        self._draw_signals()
        self._draw_panel()
        self._draw_prob_bars()
        self._draw_header()

    def _draw_grid(self):
        arcade.draw_lrbt_rectangle_outline(
            CHART_X0, CHART_X1, CHART_Y0, CHART_Y1,
            color=C_GRID, border_width=1
        )
        for i in range(1, 5):
            y = CHART_Y0 + i * (CHART_Y1 - CHART_Y0) / 5
            price_at = self._price_min + i * (self._price_max - self._price_min) / 5
            arcade.draw_line(CHART_X0, y, CHART_X1, y, C_GRID, 1)
            arcade.draw_text(f"{price_at:.4f}", 2, y - 6,
                             C_DIM, 8)
        arcade.draw_lrbt_rectangle_outline(
            CHART_X0, CHART_X1, PNL_Y0, PNL_Y1,
            color=C_GRID, border_width=1
        )
        mid_y = (PNL_Y0 + PNL_Y1) / 2
        arcade.draw_line(CHART_X0, mid_y, CHART_X1, mid_y, C_GRID, 1)
        arcade.draw_text("PnL", 2, mid_y - 6, C_DIM, 8)

    def _draw_price_line(self):
        if len(self._price_pts) < 2:
            return
        pts = self._price_pts
        for i in range(1, len(pts)):
            x0, y0 = pts[i - 1]
            x1, y1 = pts[i]
            arcade.draw_line(x0, y0, x1, y1, C_PRICE, 1)

    def _draw_pnl_strip(self):
        if len(self._pnl_points) < 2:
            return
        n_trades = self._pnl_points[-1][0]
        def tx(idx):
            return CHART_X0 + (idx / max(n_trades, 1)) * (CHART_X1 - CHART_X0)

        mid_y = (PNL_Y0 + PNL_Y1) / 2
        for i in range(1, len(self._pnl_points)):
            t0, p0 = self._pnl_points[i - 1]
            t1, p1 = self._pnl_points[i]
            col = C_LONG if p1 >= 0 else C_SHORT
            arcade.draw_line(tx(t0), self._pnl_y(p0),
                             tx(t1), self._pnl_y(p1), col, 1)

    def _draw_signals(self):
        for (sx, sy, act) in self._signals:
            if act == 0:
                col, sym = C_LONG,  "▲"
            elif act == 1:
                col, sym = C_SHORT, "▼"
            else:
                col, sym = C_CLOSE, "✘"
            arcade.draw_text(sym, sx, sy, col, 11,
                             anchor_x="center", anchor_y="center",
                             bold=True)

    def _draw_panel(self):
        """Right-side status panel."""
        px = PANEL_X + 10
        status = self.executor.get_status()
        pos    = status["position"]
        pnl    = self._cum_pnl

        def row(label, value, y, col=C_WHITE):
            arcade.draw_text(f"{label}", px,      y, C_DIM,  9)
            arcade.draw_text(f"{value}", px + 90, y, col,   10, bold=True)

        row("Mode",      self.mode.upper(),                           WIN_H - 40)
        row("Symbol",    self.symbol,                                 WIN_H - 58)
        row("Tick",      f"{self.current_tick:,} / {len(self.df):,}", WIN_H - 76)
        row("Trades",    self._trade_count,                           WIN_H - 94)

        pos_col = C_LONG if pos == "LONG" else C_SHORT if pos == "SHORT" else C_WHITE
        row("Position",  pos,                                         WIN_H - 118, pos_col)

        entry = status.get("entry")
        row("Entry",     f"${entry:.4f}" if entry else "—",           WIN_H - 136)

        pnl_col = C_LONG if pnl >= 0 else C_SHORT
        row("Cum PnL",   f"{pnl:+.3%}",                               WIN_H - 154, pnl_col)

        # Current price
        safe = min(self.current_tick, len(self.df) - 1)
        row("Price",     f"${self.df.iloc[safe]['Close']:.4f}",       WIN_H - 178)

        # Entropy — computed from vote-shares (probs), not a calibrated
        # softmax. Same threshold/colour logic as live.py for visual
        # consistency between the two agent types.
        probs = self.executor.last_probs
        ent   = float(-np.sum(probs * np.log2(probs + 1e-9)))
        ent_col = C_YELLOW if ent > 1.5 else C_CLOSE if ent > 0.8 else C_LONG
        row("Entropy",   f"{ent:.2f} bits",                           WIN_H - 202, ent_col)

        # Timestamp of current tick
        if "Open_time" in self.df.columns:
            ts = str(self.df.iloc[safe]["Open_time"])[:16]
            row("Time",  ts,                                           WIN_H - 220)

        # Live-mode: countdown to next candle (4 h interval — show hrs:mins)
        if self.mode == "live":
            remaining = max(0, self._next_candle_ts - time.time())
            hrs, rem  = divmod(int(remaining), 3600)
            mins      = rem // 60
            row("Next candle", f"{hrs:02d}h {mins:02d}m",              WIN_H - 244, C_PURPLE)

    def _draw_prob_bars(self):
        """Horizontal vote-share bars on the right panel (same layout as live.py)."""
        probs  = self.executor.last_probs
        labels = ["LONG", "SHORT", "CLOSE", "HOLD"]
        colors = [C_LONG, C_SHORT, C_CLOSE, C_HOLD]
        bar_x0 = PANEL_X + 10
        bar_max = WIN_W - bar_x0 - 10

        for i, (p, label, col) in enumerate(zip(probs, labels, colors)):
            y    = 300 - i * 38
            w    = max(2, p * bar_max)
            arcade.draw_lrbt_rectangle_filled(
                bar_x0, bar_x0 + bar_max, y - 1, y + 20,
                (30, 30, 50)
            )
            arcade.draw_lrbt_rectangle_filled(
                bar_x0, bar_x0 + w, y, y + 18,
                col
            )
            arcade.draw_text(
                f"{label}: {p:.1%}", bar_x0 + 4, y + 4,
                C_WHITE, 9, bold=(i == np.argmax(probs))
            )

    def _draw_header(self):
        safe  = min(self.current_tick, len(self.df) - 1)
        row   = self.df.iloc[safe]
        price = float(row["Close"])
        ts    = str(row["Open_time"])[:16] if "Open_time" in self.df.columns else ""
        mode_str = "◉ LIVE" if self.mode == "live" else "▶ BACKTEST"
        arcade.draw_text(
            f"{mode_str}  {self.symbol}  ${price:.4f}  [{ts}]",
            CHART_X0, WIN_H - 30,
            C_WHITE, 13, bold=True
        )

    # ── Keyboard ─────────────────────────────────────────────────────────────

    def on_key_press(self, key, modifiers):
        if key == arcade.key.ESCAPE:
            self.close()
        elif key == arcade.key.S:
            self._save_screenshot()
        elif key == arcade.key.SPACE and self.mode == "backtest":
            self._paused = not getattr(self, "_paused", False)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_screenshot(self):
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(
            OUTCOME_DIR,
            f"simulation_mcknn_{self.mode}_{self.symbol}_{ts}.png"
        )
        image = arcade.get_image()
        image.save(filename)
        self.saved = True
        log.info(f"Screenshot saved → {filename}")
        print(f"✅  Screenshot saved → {filename}")

    def on_close(self):
        if not self.saved:
            self._save_screenshot()
        super().on_close()


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def _load_agent() -> MCKNNAgent:
    """
    MCKNNAgent equivalent of live.py's _load_agent. No .eval() or
    .requires_grad_(False) calls — there are no nn.Module parameters
    to freeze; the memory bank is read-only by construction once loaded
    (query() never mutates state).
    """
    agent = MCKNNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM, k=K_NEIGHBORS)
    agent.load(CHECKPOINT_PATH)
    return agent


def _fetch_and_prepare(symbol: str) -> pd.DataFrame | None:
    """Fetch WARMUP_CANDLES candles, compute indicators, validate. IDENTICAL to live.py."""
    log.info(f"Fetching {WARMUP_CANDLES} candles for {symbol}...")
    raw = get_candles(symbol, limit=WARMUP_CANDLES)
    if raw is None:
        return None
    log.info(f"  Received {len(raw)} candles  "
             f"({raw['Open_time'].iloc[0]} → {raw['Open_time'].iloc[-1]})")

    log.info("Computing indicators...")
    df = apply_indicators(raw)
    if df is None:
        return None
    log.info(f"  {len(df)} rows after indicator calculation and dropna.")
    return df


def main():
    global CHECKPOINT_PATH
    parser = argparse.ArgumentParser(description="MC-kNN Live / Backtest Runner")
    parser.add_argument("--mode",   default="backtest",
                        choices=["backtest", "live"],
                        help="backtest: replay history | live: poll Binance in real-time")
    parser.add_argument("--symbol", default="XRPUSDT",
                        help="Binance symbol, e.g. XRPUSDT BTCUSDT ETHUSDT")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH,
                        help="Path to .npz memory-bank checkpoint")
    args = parser.parse_args()

    CHECKPOINT_PATH = args.checkpoint
    symbol          = args.symbol.upper()

    log.info(f"{'='*56}")
    log.info(f"  MODE       : {args.mode.upper()}")
    log.info(f"  SYMBOL     : {symbol}")
    log.info(f"  CHECKPOINT : {CHECKPOINT_PATH}")
    log.info(f"{'='*56}")

    # ── Load agent ────────────────────────────────────────────────────────────
    log.info("Loading MC-kNN agent...")
    agent = _load_agent()

    # ── Fetch data ────────────────────────────────────────────────────────────
    df = _fetch_and_prepare(symbol)
    if df is None:
        log.error("Could not fetch/process data. Exiting.")
        return

    # ── Build executor ────────────────────────────────────────────────────────
    executor = UnifiedExecutor(
        name=f"{symbol}_{args.mode}",
        agent=agent,
        paces=PACES,
        deterministic=True,
        num_indicators=NUM_INDICATORS,
    )

    # ── Launch arcade ─────────────────────────────────────────────────────────
    log.info("Launching window...")
    window = LiveWindow(
        df=df,
        executor=executor,
        symbol=symbol,
        mode=args.mode,
    )

    log.info(f"  Data range: {df['Open_time'].iloc[0]} → {df['Open_time'].iloc[-1]}")
    log.info(f"  Rows: {len(df):,}   |   Starting from tick {WARMUP_IDX}")
    log.info("  Keys:  [ESC] quit   [S] screenshot   [SPACE] pause/resume (backtest)")

    arcade.run()


if __name__ == "__main__":
    main()