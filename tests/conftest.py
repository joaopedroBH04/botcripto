# ============================================================
# conftest.py — shared fixtures for BotCripto test suite
# ============================================================
#
# Central helper: make_df(**kwargs)
#   Builds a synthetic 60-row OHLCV DataFrame pre-loaded with
#   every indicator column that analysis.py reads.  Individual
#   tests override only the columns / values they care about.
# ============================================================

import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Stub out Streamlit BEFORE any project module is imported.
# data_fetcher.py decorates functions with @st.cache_data; we replace that
# with a no-op pass-through so the functions are importable in tests without
# a running Streamlit server.
# ---------------------------------------------------------------------------
_st_mock = MagicMock()
# Support bare @st.cache_data (no parentheses) AND @st.cache_data(ttl=…)
_st_mock.cache_data = lambda *a, **kw: (
    a[0] if a and callable(a[0]) else (lambda f: f)
)
sys.modules["streamlit"] = _st_mock

# ---------------------------------------------------------------------------
# Stub out yfinance — tests that exercise yfinance paths patch it themselves.
# This lets tests run in environments where yfinance is not installed.
# ---------------------------------------------------------------------------
sys.modules["yfinance"] = MagicMock()

# ---------------------------------------------------------------------------
# Standard imports
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Core factory
# ---------------------------------------------------------------------------

def make_df(
    n: int = 60,
    # Price
    close: float = 100.0,
    close_prev: float = None,          # row[-2] close; defaults to same as close
    # RSI
    rsi: float = 50.0,
    # Stochastic RSI — current row
    stoch_k: float = 50.0,
    stoch_d: float = 50.0,
    # Stochastic RSI — previous row (for crossover detection)
    stoch_k_prev: float = None,
    stoch_d_prev: float = None,
    # MACD
    macd_val: float = 0.0,
    macd_signal: float = 0.0,
    macd_hist: float = 0.0,
    macd_hist_prev: float = 0.0,       # row[-2] histogram value
    # Moving averages
    sma_20: float = None,
    sma_50: float = None,
    sma_20_prev: float = None,         # 10 rows back (slope calculation)
    sma_50_prev: float = None,
    ema_9: float = None,
    ema_21: float = None,
    ema_9_prev3: float = None,         # row[-3] ema9 (crossover detection)
    ema_21_prev3: float = None,
    # ADX
    adx: float = 25.0,
    adx_pos: float = 25.0,
    adx_neg: float = 20.0,
    # Bollinger Bands
    bb_upper: float = 110.0,
    bb_middle: float = 100.0,
    bb_lower: float = 90.0,
    bb_upper_5: float = None,          # row[-5] for expansion detection
    bb_lower_5: float = None,
    bb_middle_5: float = None,
    # ATR
    atr: float = 2.0,
    # Volume / OBV
    volume: float = 1_000.0,
    volume_sma_20: float = 1_000.0,
    obv: float = 100_000.0,
    obv_sma_20: float = 90_000.0,
) -> pd.DataFrame:
    """
    Return a DataFrame of *n* daily rows with all indicator columns populated.

    The last row carries the values supplied via keyword arguments.
    Earlier rows receive sensible defaults so that helper functions that
    inspect *window* positions (e.g. iloc[-2], iloc[-3], slope over 10 bars)
    behave predictably.
    """
    dates = pd.date_range(end="2025-01-01", periods=n, freq="D")
    df = pd.DataFrame(index=dates)

    # --- OHLCV ---------------------------------------------------------------
    df["Close"] = close
    df["Open"] = close
    df["High"] = close * 1.02
    df["Low"] = close * 0.98
    df["Volume"] = volume

    # Override row[-2] close if supplied (used by volume scoring)
    _cp = close_prev if close_prev is not None else close
    df.iloc[-2, df.columns.get_loc("Close")] = _cp

    # --- RSI -----------------------------------------------------------------
    df["rsi_14"] = rsi

    # --- Stochastic RSI ------------------------------------------------------
    _sk_prev = stoch_k_prev if stoch_k_prev is not None else stoch_k
    _sd_prev = stoch_d_prev if stoch_d_prev is not None else stoch_d
    df["stoch_rsi_k"] = _sk_prev
    df["stoch_rsi_d"] = _sd_prev
    df.iloc[-1, df.columns.get_loc("stoch_rsi_k")] = stoch_k
    df.iloc[-1, df.columns.get_loc("stoch_rsi_d")] = stoch_d

    # --- MACD ----------------------------------------------------------------
    df["macd"] = macd_val
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist_prev
    df.iloc[-1, df.columns.get_loc("macd_hist")] = macd_hist

    # --- SMAs ----------------------------------------------------------------
    _s20 = sma_20 if sma_20 is not None else close
    _s50 = sma_50 if sma_50 is not None else close
    df["sma_20"] = _s20
    df["sma_50"] = _s50
    # Fill earlier rows with "previous" values to create a measurable slope
    if sma_20_prev is not None:
        df.iloc[:-1, df.columns.get_loc("sma_20")] = sma_20_prev
    if sma_50_prev is not None:
        df.iloc[:-1, df.columns.get_loc("sma_50")] = sma_50_prev

    # --- EMAs ----------------------------------------------------------------
    _e9 = ema_9 if ema_9 is not None else close
    _e21 = ema_21 if ema_21 is not None else close
    df["ema_9"] = _e9
    df["ema_21"] = _e21
    # Override row[-3] for EMA crossover detection
    if ema_9_prev3 is not None:
        df.iloc[-3, df.columns.get_loc("ema_9")] = ema_9_prev3
    if ema_21_prev3 is not None:
        df.iloc[-3, df.columns.get_loc("ema_21")] = ema_21_prev3

    # --- ADX -----------------------------------------------------------------
    df["adx"] = adx
    df["adx_pos"] = adx_pos
    df["adx_neg"] = adx_neg

    # --- Bollinger Bands -----------------------------------------------------
    df["bb_upper"] = bb_upper
    df["bb_middle"] = bb_middle
    df["bb_lower"] = bb_lower
    # 5 rows back used for band-expansion detection
    _bu5 = bb_upper_5 if bb_upper_5 is not None else bb_upper
    _bl5 = bb_lower_5 if bb_lower_5 is not None else bb_lower
    _bm5 = bb_middle_5 if bb_middle_5 is not None else bb_middle
    df.iloc[-5, df.columns.get_loc("bb_upper")] = _bu5
    df.iloc[-5, df.columns.get_loc("bb_lower")] = _bl5
    df.iloc[-5, df.columns.get_loc("bb_middle")] = _bm5

    # --- Volume / OBV --------------------------------------------------------
    df["volume_sma_20"] = volume_sma_20
    df["obv"] = obv
    df["obv_sma_20"] = obv_sma_20

    # --- ATR -----------------------------------------------------------------
    df["atr"] = atr

    return df


# ---------------------------------------------------------------------------
# Named scenario fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def neutral_df():
    """Balanced/neutral market — no strong signal in any direction."""
    return make_df()


@pytest.fixture
def oversold_df():
    """Deeply oversold market (RSI=18, Stoch=15) — maximum buy score expected."""
    return make_df(
        rsi=18.0,
        stoch_k=15.0, stoch_d=20.0,
        stoch_k_prev=10.0, stoch_d_prev=18.0,   # bullish crossover
        macd_val=0.5, macd_signal=0.2,
        macd_hist=0.3, macd_hist_prev=0.1,
        adx=30.0, adx_pos=35.0, adx_neg=15.0,
        bb_lower=95.0, bb_middle=100.0, bb_upper=105.0,
        close=95.5,
    )


@pytest.fixture
def overbought_df():
    """Overbought market (RSI=82, Stoch=88) — low buy score expected."""
    return make_df(
        rsi=82.0,
        stoch_k=88.0, stoch_d=85.0,
        macd_val=-0.5, macd_signal=0.2,
        macd_hist=-0.3, macd_hist_prev=-0.1,
        adx=35.0, adx_pos=15.0, adx_neg=40.0,
        bb_lower=90.0, bb_middle=100.0, bb_upper=105.0,
        close=104.5,
    )


@pytest.fixture
def empty_df():
    return pd.DataFrame()


@pytest.fixture
def short_df():
    """Too short for indicator calculation (< 30 rows)."""
    return make_df(n=20)
