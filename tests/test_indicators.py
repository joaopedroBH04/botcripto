# ============================================================
# test_indicators.py — unit tests for the indicator and
# trend/divergence detection functions in analysis.py
#
# Functions covered:
#   _find_swing_lows, _find_swing_highs
#   compute_fibonacci
#   _calc_slope
#   detect_rsi_divergence, detect_macd_divergence
#   detect_trend
#   classify_dip
#   compute_indicators (smoke test)
# ============================================================

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis import (
    _find_swing_lows,
    _find_swing_highs,
    compute_fibonacci,
    _calc_slope,
    detect_rsi_divergence,
    detect_macd_divergence,
    detect_trend,
    classify_dip,
    compute_indicators,
)
from tests.conftest import make_df


# ===========================================================================
# _find_swing_lows / _find_swing_highs
# ===========================================================================

class TestFindSwings:
    def _series(self, values):
        return pd.Series(values, dtype=float)

    def test_single_valley_detected(self):
        # obvious V shape
        vals = [10, 9, 8, 7, 6, 7, 8, 9, 10, 11, 12]
        lows = _find_swing_lows(self._series(vals), order=2)
        assert len(lows) == 1
        _, val = lows[0]
        assert val == 6.0

    def test_single_peak_detected(self):
        vals = [5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3]
        highs = _find_swing_highs(self._series(vals), order=2)
        assert len(highs) == 1
        _, val = highs[0]
        assert val == 9.0

    def test_multiple_valleys_detected(self):
        # two separate valleys
        vals = [10, 5, 10, 5, 10, 5, 10]
        lows = _find_swing_lows(self._series(vals), order=1)
        assert len(lows) >= 2

    def test_flat_series_treats_every_point_as_swing(self):
        # The implementation uses <= / >= (not strict < / >), so every interior
        # point in a flat series satisfies both conditions simultaneously.
        # This test documents that known behaviour.
        vals = [50.0] * 20
        lows  = _find_swing_lows(self._series(vals), order=2)
        highs = _find_swing_highs(self._series(vals), order=2)
        # With order=2 and 20 values, interior range is [2, 17] → 16 results
        assert len(lows)  == 16
        assert len(highs) == 16

    def test_too_short_for_order_returns_empty(self):
        # with order=5 we need at least 11 points
        vals = [1, 2, 3, 4, 5, 4, 3, 2, 1]
        lows = _find_swing_lows(self._series(vals), order=5)
        assert lows == []

    def test_series_with_nans_does_not_raise(self):
        vals = [np.nan, 5.0, 3.0, 5.0, np.nan, 4.0, 2.0, 4.0, np.nan]
        # should not raise; may return fewer results
        lows = _find_swing_lows(self._series(vals), order=1)
        assert isinstance(lows, list)


# ===========================================================================
# compute_fibonacci
# ===========================================================================

class TestComputeFibonacci:
    def test_levels_keys_present(self):
        df = make_df(n=100)
        result = compute_fibonacci(df)
        assert "levels" in result
        assert "swing_high" in result
        assert "swing_low" in result

    def test_level_0_is_swing_high(self):
        df = make_df(n=100)
        result = compute_fibonacci(df)
        levels = result["levels"]
        assert abs(levels["0%"] - result["swing_high"]) < 1e-4

    def test_level_100_is_swing_low(self):
        df = make_df(n=100)
        result = compute_fibonacci(df)
        levels = result["levels"]
        assert abs(levels["100%"] - result["swing_low"]) < 1e-4

    def test_levels_are_ordered_high_to_low(self):
        """Each successive Fibonacci level should be lower than the previous."""
        df = make_df(n=100)
        result = compute_fibonacci(df)
        levels = result["levels"]
        ordered_keys = ["0%", "23.6%", "38.2%", "50%", "61.8%", "78.6%", "100%"]
        prices = [levels[k] for k in ordered_keys]
        assert prices == sorted(prices, reverse=True)

    def test_zero_range_returns_empty_levels(self):
        # High == Low → diff == 0
        df = make_df(n=100, close=100.0)
        df["High"] = 100.0
        df["Low"] = 100.0
        result = compute_fibonacci(df)
        assert result["levels"] == {}

    def test_lookback_longer_than_df_uses_full_df(self):
        df = make_df(n=30)
        # Should not raise even with lookback=90
        result = compute_fibonacci(df, lookback=90)
        assert "levels" in result

    def test_618_level_formula(self):
        """61.8% level = swing_high - diff * 0.618."""
        df = make_df(n=100)
        df["High"] = 200.0
        df["Low"] = 100.0
        result = compute_fibonacci(df, lookback=100)
        expected = 200.0 - (100.0 * 0.618)
        assert abs(result["levels"]["61.8%"] - expected) < 0.01


# ===========================================================================
# _calc_slope
# ===========================================================================

class TestCalcSlope:
    def test_rising_series_has_positive_slope(self):
        s = pd.Series(np.linspace(10, 20, 20))
        slope = _calc_slope(s, window=10)
        assert slope > 0

    def test_falling_series_has_negative_slope(self):
        s = pd.Series(np.linspace(20, 10, 20))
        slope = _calc_slope(s, window=10)
        assert slope < 0

    def test_flat_series_has_near_zero_slope(self):
        # np.polyfit on a constant series returns floating-point noise (~1e-17),
        # not exact zero; use an absolute tolerance.
        s = pd.Series([100.0] * 20)
        slope = _calc_slope(s, window=10)
        assert abs(slope) < 1e-10

    def test_none_series_returns_zero(self):
        assert _calc_slope(None) == 0.0

    def test_too_short_series_returns_zero(self):
        s = pd.Series([1.0, 2.0, 3.0])
        assert _calc_slope(s, window=10) == 0.0

    def test_series_with_all_nans_returns_zero(self):
        s = pd.Series([np.nan] * 20)
        assert _calc_slope(s, window=10) == 0.0


# ===========================================================================
# detect_rsi_divergence
# ===========================================================================

class TestDetectRSIDivergence:
    def _make_divergence_df(self, bullish=True):
        """
        Build a DataFrame whose Close and rsi_14 series contain exactly
        two swing lows (bullish) or two swing highs (bearish) with
        the divergence pattern.
        """
        n = 80
        close_vals = np.ones(n) * 100.0
        rsi_vals = np.ones(n) * 50.0

        if bullish:
            # Price: two lows where second < first (lower low)
            # RSI:   two lows where second > first (higher low) — bullish div
            close_vals[20] = 85.0   # first low
            close_vals[45] = 80.0   # second lower low
            rsi_vals[20] = 28.0    # first rsi low
            rsi_vals[45] = 33.0    # second higher rsi low
        else:
            # Price: two highs where second > first (higher high)
            # RSI:   two highs where second < first (lower high) — bearish div
            close_vals[20] = 120.0
            close_vals[45] = 125.0
            rsi_vals[20] = 72.0
            rsi_vals[45] = 67.0

        dates = pd.date_range(end="2025-01-01", periods=n, freq="D")
        df = make_df(n=n)
        df["Close"] = close_vals
        df["rsi_14"] = rsi_vals
        return df

    def test_missing_rsi_column_returns_none_type(self):
        df = make_df()
        df.drop(columns=["rsi_14"], inplace=True)
        result = detect_rsi_divergence(df)
        assert result["type"] == "none"

    def test_too_short_df_returns_none_type(self):
        df = make_df(n=30)
        result = detect_rsi_divergence(df, lookback=60)
        assert result["type"] == "none"

    def test_result_has_required_keys(self):
        df = make_df()
        result = detect_rsi_divergence(df)
        assert "type" in result
        assert "strength" in result
        assert "description" in result

    def test_type_is_valid_value(self):
        df = make_df()
        result = detect_rsi_divergence(df)
        assert result["type"] in ("none", "bullish", "bearish")

    def test_strength_is_non_negative(self):
        df = make_df()
        result = detect_rsi_divergence(df)
        assert result["strength"] >= 0


# ===========================================================================
# detect_macd_divergence
# ===========================================================================

class TestDetectMACDDivergence:
    def test_missing_macd_hist_returns_none_type(self):
        df = make_df()
        df.drop(columns=["macd_hist"], inplace=True)
        result = detect_macd_divergence(df)
        assert result["type"] == "none"

    def test_too_short_df_returns_none_type(self):
        df = make_df(n=30)
        result = detect_macd_divergence(df, lookback=60)
        assert result["type"] == "none"

    def test_result_has_required_keys(self):
        df = make_df()
        result = detect_macd_divergence(df)
        assert "type" in result
        assert "strength" in result
        assert "description" in result

    def test_type_is_valid_value(self):
        df = make_df()
        result = detect_macd_divergence(df)
        assert result["type"] in ("none", "bullish", "bearish")


# ===========================================================================
# detect_trend
# ===========================================================================

class TestDetectTrend:
    def test_empty_df_returns_indefinido(self):
        trend, strength = detect_trend(pd.DataFrame())
        assert trend == "indefinido"
        assert strength == 0

    def test_short_df_returns_indefinido(self):
        df = make_df(n=30)
        trend, strength = detect_trend(df)
        assert trend == "indefinido"
        assert strength == 0

    def test_sideways_market_returns_lateral(self):
        # adx < 20 → lateral
        df = make_df(adx=15.0, adx_pos=18.0, adx_neg=16.0)
        trend, strength = detect_trend(df)
        assert trend == "lateral"

    def test_strong_uptrend_returns_alta(self):
        # di+ > di-, close > sma_20, rising slopes
        df = make_df(
            close=110.0,
            sma_20=105.0, sma_20_prev=100.0,
            sma_50=100.0, sma_50_prev=98.0,
            adx=35.0, adx_pos=40.0, adx_neg=20.0,
        )
        trend, strength = detect_trend(df)
        assert trend == "alta"

    def test_strong_downtrend_returns_baixa(self):
        # di- > di+, close < sma_20, falling slopes
        df = make_df(
            close=88.0,
            sma_20=95.0, sma_20_prev=100.0,
            sma_50=100.0, sma_50_prev=102.0,
            adx=35.0, adx_pos=15.0, adx_neg=40.0,
        )
        trend, strength = detect_trend(df)
        assert trend == "baixa"

    def test_strength_is_between_0_and_100(self):
        df = make_df()
        _, strength = detect_trend(df)
        assert 0 <= strength <= 100

    def test_trend_returns_valid_string(self):
        valid = {"alta", "reversao_alta", "alta_fraca", "baixa",
                 "reversao_baixa", "baixa_fraca", "lateral", "indefinido"}
        df = make_df()
        trend, _ = detect_trend(df)
        assert trend in valid

    def test_missing_sma_columns_returns_indefinido(self):
        df = make_df()
        df.drop(columns=["sma_20", "sma_50"], inplace=True)
        trend, _ = detect_trend(df)
        assert trend == "indefinido"


# ===========================================================================
# classify_dip
# ===========================================================================

class TestClassifyDip:
    def test_empty_df_returns_indefinido(self):
        result = classify_dip(pd.DataFrame())
        assert result["type"] == "indefinido"

    def test_stable_price_returns_estavel(self):
        # drawdown < atr_pct * 1.5  → "estavel"
        # With close=100, atr=2 → atr_pct=2.0%, atr_pct*1.5=3%
        # Keep all 30 rows at 100 (drawdown ≈ 0%)
        df = make_df(n=60, close=100.0, atr=2.0)
        result = classify_dip(df)
        assert result["type"] == "estavel"

    def test_large_drawdown_returns_queda_forte(self):
        # drawdown >= 20%
        # Set recent_high high and current price low
        df = make_df(n=60, close=75.0, atr=2.0)
        # Set rows[-30:] to have max at 100 (but last row at 75)
        df.iloc[-30:-1, df.columns.get_loc("Close")] = 100.0
        result = classify_dip(df)
        assert result["type"] == "queda_forte"

    def test_drawdown_is_numeric(self):
        df = make_df(n=60, close=95.0, atr=2.0)
        result = classify_dip(df)
        assert isinstance(result["drawdown"], float)
        assert result["drawdown"] >= 0

    def test_result_has_required_keys(self):
        df = make_df(n=60)
        result = classify_dip(df)
        assert "type" in result
        assert "drawdown" in result
        assert "explanation" in result

    def test_type_is_valid_value(self):
        valid = {"estavel", "ruido", "alerta", "correcao",
                 "queda_moderada", "queda_forte", "indefinido"}
        df = make_df(n=60)
        result = classify_dip(df)
        assert result["type"] in valid


# ===========================================================================
# compute_indicators (smoke test — delegates to `ta` library)
# ===========================================================================

class TestComputeIndicators:
    def test_short_df_returned_unchanged(self):
        df = make_df(n=20)
        # Remove indicator columns so we can verify nothing was added
        base_cols = {"Close", "Open", "High", "Low", "Volume"}
        result = compute_indicators(df[list(base_cols)])
        # Function should return the df as-is (no ta columns added)
        assert set(result.columns) == base_cols

    def test_sufficient_data_adds_rsi_column(self):
        # Build a minimal real OHLCV frame — no pre-filled indicator columns
        n = 60
        dates = pd.date_range(end="2025-01-01", periods=n, freq="D")
        prices = np.linspace(90, 110, n)
        df = pd.DataFrame({
            "Close": prices,
            "Open": prices * 0.99,
            "High": prices * 1.01,
            "Low": prices * 0.98,
            "Volume": np.random.randint(500, 2000, n).astype(float),
        }, index=dates)
        result = compute_indicators(df)
        assert "rsi_14" in result.columns

    def test_sufficient_data_adds_macd_column(self):
        n = 60
        dates = pd.date_range(end="2025-01-01", periods=n, freq="D")
        prices = np.linspace(90, 110, n)
        df = pd.DataFrame({
            "Close": prices,
            "Open": prices * 0.99,
            "High": prices * 1.01,
            "Low": prices * 0.98,
            "Volume": np.ones(n) * 1000,
        }, index=dates)
        result = compute_indicators(df)
        assert "macd" in result.columns
        assert "macd_signal" in result.columns

    def test_fibonacci_stored_in_attrs(self):
        n = 60
        dates = pd.date_range(end="2025-01-01", periods=n, freq="D")
        prices = np.linspace(90, 110, n)
        df = pd.DataFrame({
            "Close": prices,
            "Open": prices,
            "High": prices * 1.01,
            "Low": prices * 0.99,
            "Volume": np.ones(n) * 1000,
        }, index=dates)
        result = compute_indicators(df)
        assert "fibonacci" in result.attrs
        assert "levels" in result.attrs["fibonacci"]
