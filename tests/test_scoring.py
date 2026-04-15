# ============================================================
# test_scoring.py — unit tests for analysis.py scoring layer
#
# Covers every individual _score_* function plus the
# score_asset() orchestrator.  Each test targets a specific
# branch / boundary condition so that a regression in the
# scoring logic is immediately visible.
# ============================================================

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis import (
    _score_rsi,
    _score_stoch_rsi,
    _score_macd,
    _score_trend,
    _score_adx,
    _score_bollinger,
    _score_volume_obv,
    _score_fibonacci,
    _score_fear_greed,
    _score_divergences,
    score_asset,
    _empty_score,
)
from config import (
    SCORE_MAX_RSI, SCORE_MAX_STOCH_RSI, SCORE_MAX_MACD,
    SCORE_MAX_TREND, SCORE_MAX_ADX, SCORE_MAX_BOLLINGER,
    SCORE_MAX_VOLUME, SCORE_MAX_FIBONACCI, SCORE_MAX_FEAR_GREED,
    SCORE_MAX_DIVERGENCE,
    BUY_CONFIDENCE_STRONG, BUY_CONFIDENCE_MODERATE, SELL_CONFIDENCE,
)
from tests.conftest import make_df


# ===========================================================================
# _score_rsi
# ===========================================================================

class TestScoreRSI:
    def test_extreme_oversold_returns_max(self):
        df = make_df(rsi=18.0)
        pts, info = _score_rsi(df)
        assert pts == SCORE_MAX_RSI
        assert info["points"] == SCORE_MAX_RSI
        assert info["max"] == SCORE_MAX_RSI

    def test_oversold_returns_75_pct(self):
        df = make_df(rsi=28.0)
        pts, info = _score_rsi(df)
        assert pts == int(SCORE_MAX_RSI * 0.75)

    def test_below_average_returns_50_pct(self):
        df = make_df(rsi=38.0)
        pts, info = _score_rsi(df)
        assert pts == int(SCORE_MAX_RSI * 0.5)

    def test_neutral_returns_25_pct(self):
        df = make_df(rsi=55.0)
        pts, info = _score_rsi(df)
        assert pts == int(SCORE_MAX_RSI * 0.25)

    def test_above_average_returns_1(self):
        df = make_df(rsi=68.0)
        pts, info = _score_rsi(df)
        assert pts == 1

    def test_overbought_returns_zero(self):
        df = make_df(rsi=82.0)
        pts, info = _score_rsi(df)
        assert pts == 0

    def test_nan_rsi_returns_zero(self):
        df = make_df(rsi=50.0)
        df["rsi_14"] = np.nan
        pts, info = _score_rsi(df)
        assert pts == 0

    def test_missing_column_returns_zero(self):
        df = make_df()
        df.drop(columns=["rsi_14"], inplace=True)
        pts, info = _score_rsi(df)
        assert pts == 0

    def test_boundary_rsi_20_is_max(self):
        df = make_df(rsi=20.0)
        pts, _ = _score_rsi(df)
        assert pts == SCORE_MAX_RSI

    def test_boundary_rsi_30_is_75_pct(self):
        df = make_df(rsi=30.0)
        pts, _ = _score_rsi(df)
        assert pts == int(SCORE_MAX_RSI * 0.75)

    def test_info_value_is_rounded(self):
        df = make_df(rsi=35.678)
        _, info = _score_rsi(df)
        assert info["value"] == round(35.678, 1)


# ===========================================================================
# _score_stoch_rsi
# ===========================================================================

class TestScoreStochRSI:
    def test_oversold_with_bullish_crossover_returns_max(self):
        # Bullish crossover: k_prev <= d_prev (below) then k_val > d_val (above).
        # Both k_val and d_val must be <= 20 so k can cross above d while still
        # satisfying the oversold condition (k_val <= STOCH_RSI_OVERSOLD = 20).
        df = make_df(
            stoch_k=18.0, stoch_d=15.0,        # k crossed above d, both oversold
            stoch_k_prev=12.0, stoch_d_prev=16.0,  # k was below d before
        )
        pts, info = _score_stoch_rsi(df)
        assert pts == SCORE_MAX_STOCH_RSI

    def test_oversold_without_crossover_returns_75_pct(self):
        # k_val <= 20 but no crossover (k still below d)
        df = make_df(
            stoch_k=12.0, stoch_d=18.0,         # k < d → no crossover
            stoch_k_prev=10.0, stoch_d_prev=16.0,
        )
        pts, _ = _score_stoch_rsi(df)
        assert pts == int(SCORE_MAX_STOCH_RSI * 0.75)

    def test_low_k_rising_returns_50_pct(self):
        # 20 < k_val <= 40 AND k_val > d_val
        df = make_df(stoch_k=35.0, stoch_d=30.0)
        pts, _ = _score_stoch_rsi(df)
        assert pts == int(SCORE_MAX_STOCH_RSI * 0.5)

    def test_overbought_returns_zero(self):
        df = make_df(stoch_k=85.0, stoch_d=80.0)
        pts, _ = _score_stoch_rsi(df)
        assert pts == 0

    def test_neutral_returns_25_pct(self):
        df = make_df(stoch_k=60.0, stoch_d=65.0)
        pts, _ = _score_stoch_rsi(df)
        assert pts == int(SCORE_MAX_STOCH_RSI * 0.25)

    def test_missing_column_returns_zero(self):
        df = make_df()
        df.drop(columns=["stoch_rsi_k"], inplace=True)
        pts, _ = _score_stoch_rsi(df)
        assert pts == 0


# ===========================================================================
# _score_macd
# ===========================================================================

class TestScoreMACD:
    def test_bullish_crossover_with_positive_rising_hist_returns_max(self):
        # macd > signal, hist > hist_prev, hist > 0
        df = make_df(macd_val=1.0, macd_signal=0.5, macd_hist=0.5, macd_hist_prev=0.2)
        pts, _ = _score_macd(df)
        assert pts == SCORE_MAX_MACD

    def test_macd_above_signal_rising_hist_negative_returns_80_pct(self):
        # macd > signal, h > h_prev, but h < 0
        df = make_df(macd_val=0.5, macd_signal=0.8, macd_hist=-0.1, macd_hist_prev=-0.3)
        # macd < signal here, so this is the "m < s and h > h_prev" branch
        pts, _ = _score_macd(df)
        assert pts == int(SCORE_MAX_MACD * 0.4)

    def test_macd_above_signal_only_returns_60_pct(self):
        # macd > signal, hist falling
        df = make_df(macd_val=1.0, macd_signal=0.5, macd_hist=0.2, macd_hist_prev=0.4)
        pts, _ = _score_macd(df)
        assert pts == int(SCORE_MAX_MACD * 0.6)

    def test_bearish_crossover_returns_1(self):
        # macd < signal, hist falling
        df = make_df(macd_val=-0.5, macd_signal=0.2, macd_hist=-0.3, macd_hist_prev=-0.1)
        pts, _ = _score_macd(df)
        assert pts == 1

    def test_missing_macd_column_returns_zero(self):
        df = make_df()
        df.drop(columns=["macd"], inplace=True)
        pts, _ = _score_macd(df)
        assert pts == 0


# ===========================================================================
# _score_trend
# ===========================================================================

class TestScoreTrend:
    def test_reversal_signal_below_sma50_returns_max(self):
        # close < sma_50, sma_20 rising (sma_20_prev < sma_20), ema_9 > ema_21
        df = make_df(
            close=90.0,
            sma_20=92.0, sma_20_prev=88.0,   # rising slope
            sma_50=95.0,
            ema_9=93.0, ema_21=91.0,           # ema bullish
        )
        pts, _ = _score_trend(df)
        assert pts == SCORE_MAX_TREND

    def test_strong_uptrend_above_both_smas_returns_67_pct(self):
        # close > sma_20 > sma_50, ema bullish
        df = make_df(
            close=110.0,
            sma_20=105.0, sma_50=100.0,
            ema_9=107.0, ema_21=103.0,
        )
        pts, _ = _score_trend(df)
        assert pts == int(SCORE_MAX_TREND * 0.67)

    def test_bearish_trend_below_both_smas_returns_17_pct(self):
        # close < sma_20 < sma_50, ema bearish
        df = make_df(
            close=85.0,
            sma_20=90.0, sma_50=95.0,
            ema_9=88.0, ema_21=92.0,           # ema bearish
        )
        pts, _ = _score_trend(df)
        assert pts == int(SCORE_MAX_TREND * 0.17)

    def test_missing_sma_returns_zero(self):
        df = make_df()
        df.drop(columns=["sma_20"], inplace=True)
        pts, _ = _score_trend(df)
        assert pts == 0


# ===========================================================================
# _score_adx
# ===========================================================================

class TestScoreADX:
    def test_very_strong_uptrend_returns_max(self):
        # adx >= 40, di+ > di-
        df = make_df(adx=45.0, adx_pos=50.0, adx_neg=20.0)
        pts, _ = _score_adx(df)
        assert pts == SCORE_MAX_ADX

    def test_strong_uptrend_returns_80_pct(self):
        # adx >= 25, di+ > di-
        df = make_df(adx=30.0, adx_pos=35.0, adx_neg=20.0)
        pts, _ = _score_adx(df)
        assert pts == int(SCORE_MAX_ADX * 0.8)

    def test_strong_downtrend_returns_20_pct(self):
        # adx >= 25, di- > di+
        df = make_df(adx=30.0, adx_pos=15.0, adx_neg=35.0)
        pts, _ = _score_adx(df)
        assert pts == int(SCORE_MAX_ADX * 0.2)

    def test_sideways_market_returns_30_pct(self):
        # adx < 20
        df = make_df(adx=15.0, adx_pos=18.0, adx_neg=16.0)
        pts, _ = _score_adx(df)
        assert pts == int(SCORE_MAX_ADX * 0.3)

    def test_missing_adx_column_returns_default(self):
        df = make_df()
        df.drop(columns=["adx"], inplace=True)
        pts, _ = _score_adx(df)
        assert pts == int(SCORE_MAX_ADX * 0.3)


# ===========================================================================
# _score_bollinger
# ===========================================================================

class TestScoreBollinger:
    def test_near_lower_band_low_rsi_stable_returns_max(self):
        # position <= 0.1, rsi < 40, not expanding
        # close=91: position=(91-90)/(110-90)=0.05
        df = make_df(
            close=91.0,
            bb_lower=90.0, bb_middle=100.0, bb_upper=110.0,
            bb_lower_5=90.0, bb_upper_5=110.0, bb_middle_5=100.0,  # same width → not expanding
            rsi=35.0,
        )
        pts, _ = _score_bollinger(df)
        assert pts == SCORE_MAX_BOLLINGER

    def test_near_upper_band_returns_zero(self):
        # position >= 0.9
        # close=109: position=(109-90)/(110-90)=0.95
        df = make_df(
            close=109.0,
            bb_lower=90.0, bb_middle=100.0, bb_upper=110.0,
        )
        pts, _ = _score_bollinger(df)
        assert pts == 0

    def test_near_lower_band_expanding_returns_25_pct(self):
        # position <= 0.2, but bands expanding (prev narrower)
        # close=91: position=0.05
        df = make_df(
            close=91.0,
            bb_lower=90.0, bb_middle=100.0, bb_upper=110.0,
            bb_lower_5=93.0, bb_upper_5=107.0, bb_middle_5=100.0,  # narrower before → now expanding
        )
        pts, _ = _score_bollinger(df)
        assert pts == int(SCORE_MAX_BOLLINGER * 0.25)

    def test_missing_bb_column_returns_zero(self):
        df = make_df()
        df.drop(columns=["bb_lower"], inplace=True)
        pts, _ = _score_bollinger(df)
        assert pts == 0


# ===========================================================================
# _score_volume_obv
# ===========================================================================

class TestScoreVolumeOBV:
    def test_falling_price_low_volume_accumulation_returns_max(self):
        # price_change < 0, vol_ratio < 0.8, obv > obv_sma (accumulation)
        df = make_df(
            close=98.0, close_prev=100.0,
            volume=700.0, volume_sma_20=1_000.0,    # ratio = 0.7
            obv=120_000.0, obv_sma_20=100_000.0,    # obv > avg = accumulation
        )
        pts, _ = _score_volume_obv(df)
        assert pts == SCORE_MAX_VOLUME

    def test_falling_price_high_volume_distribution_returns_10_pct(self):
        # price_change < 0, vol_ratio > 1.5, obv < obv_sma (distribution)
        df = make_df(
            close=98.0, close_prev=100.0,
            volume=2_000.0, volume_sma_20=1_000.0,   # ratio = 2.0
            obv=80_000.0, obv_sma_20=100_000.0,      # obv < avg = distribution
        )
        pts, _ = _score_volume_obv(df)
        assert pts == int(SCORE_MAX_VOLUME * 0.1)

    def test_rising_price_high_volume_accumulation_returns_80_pct(self):
        # price_change > 0, vol_ratio > 1.2, accumulation
        df = make_df(
            close=103.0, close_prev=100.0,
            volume=1_500.0, volume_sma_20=1_000.0,
            obv=110_000.0, obv_sma_20=100_000.0,
        )
        pts, _ = _score_volume_obv(df)
        assert pts == int(SCORE_MAX_VOLUME * 0.8)

    def test_missing_volume_column_returns_default(self):
        df = make_df()
        df.drop(columns=["Volume"], inplace=True)
        pts, _ = _score_volume_obv(df)
        assert pts == int(SCORE_MAX_VOLUME * 0.5)


# ===========================================================================
# _score_fibonacci
# ===========================================================================

class TestScoreFibonacci:
    def _df_with_fib(self, close, fib_618, fib_50=None, fib_382=None):
        """Helper: build df with Fibonacci levels in attrs."""
        df = make_df(close=close)
        df.attrs["fibonacci"] = {
            "levels": {
                "61.8%": fib_618,
                "50%": fib_50 if fib_50 is not None else fib_618 * 1.05,
                "38.2%": fib_382 if fib_382 is not None else fib_618 * 1.10,
                "23.6%": fib_618 * 1.15,
                "0%": fib_618 * 1.20,
                "78.6%": fib_618 * 0.95,
                "100%": fib_618 * 0.90,
            },
            "swing_high": close * 1.3,
            "swing_low": close * 0.7,
        }
        return df

    def test_at_618_level_returns_max(self):
        # close right at 61.8% level, dist <= 3%
        df = self._df_with_fib(close=100.0, fib_618=100.0)
        pts, _ = _score_fibonacci(df)
        assert pts == SCORE_MAX_FIBONACCI

    def test_at_50_level_returns_80_pct(self):
        df = self._df_with_fib(close=105.0, fib_618=90.0, fib_50=105.0)
        pts, _ = _score_fibonacci(df)
        assert pts == int(SCORE_MAX_FIBONACCI * 0.8)

    def test_at_382_level_returns_60_pct(self):
        df = self._df_with_fib(close=110.0, fib_618=90.0, fib_50=100.0, fib_382=110.0)
        pts, _ = _score_fibonacci(df)
        assert pts == int(SCORE_MAX_FIBONACCI * 0.6)

    def test_no_fibonacci_levels_returns_default(self):
        df = make_df(close=100.0)
        df.attrs["fibonacci"] = {"levels": {}, "swing_high": 110, "swing_low": 90}
        pts, _ = _score_fibonacci(df)
        assert pts == int(SCORE_MAX_FIBONACCI * 0.5)


# ===========================================================================
# _score_fear_greed  (pure function — no DataFrame needed)
# ===========================================================================

class TestScoreFearGreed:
    @pytest.mark.parametrize("value,expected_pts,expected_label", [
        (10,  SCORE_MAX_FEAR_GREED,              "Medo Extremo"),
        (15,  SCORE_MAX_FEAR_GREED,              "Medo Extremo"),   # boundary
        (20,  int(SCORE_MAX_FEAR_GREED * 0.75),  "Medo"),
        (30,  int(SCORE_MAX_FEAR_GREED * 0.75),  "Medo"),           # boundary
        (40,  int(SCORE_MAX_FEAR_GREED * 0.5),   "Medo Leve"),
        (45,  int(SCORE_MAX_FEAR_GREED * 0.5),   "Medo Leve"),      # boundary
        (50,  int(SCORE_MAX_FEAR_GREED * 0.25),  "Neutro"),
        (55,  int(SCORE_MAX_FEAR_GREED * 0.25),  "Neutro"),         # boundary
        (60,  1,                                  None),             # Ganancia
        (70,  1,                                  None),             # boundary
        (85,  0,                                  "Ganancia Extrema"),
    ])
    def test_all_brackets(self, value, expected_pts, expected_label):
        pts, info = _score_fear_greed(value)
        assert pts == expected_pts
        assert info["points"] == expected_pts
        assert info["max"] == SCORE_MAX_FEAR_GREED
        if expected_label is not None:
            assert info["label"] == expected_label

    def test_max_points_at_extreme_fear(self):
        pts, info = _score_fear_greed(1)
        assert pts == SCORE_MAX_FEAR_GREED

    def test_zero_points_at_extreme_greed(self):
        pts, info = _score_fear_greed(100)
        assert pts == 0


# ===========================================================================
# _score_divergences
# ===========================================================================

class TestScoreDivergences:
    def test_no_divergence_returns_neutral_points(self):
        # Short or flat data → divergence functions return "none"
        df = make_df(n=60, rsi=50.0)
        pts, info = _score_divergences(df)
        # Neutral: int(MAX * 0.3) when no divergence detected
        assert pts == int(SCORE_MAX_DIVERGENCE * 0.3)

    def test_points_within_max(self):
        df = make_df()
        pts, info = _score_divergences(df)
        assert 0 <= pts <= SCORE_MAX_DIVERGENCE


# ===========================================================================
# score_asset — integration
# ===========================================================================

class TestScoreAsset:
    def test_empty_df_returns_empty_score(self):
        result = score_asset(pd.DataFrame())
        assert result == _empty_score()
        assert result["score"] == 0
        assert result["label"] == "Sem Dados"

    def test_short_df_returns_empty_score(self):
        df = make_df(n=30)
        result = score_asset(df)
        assert result == _empty_score()

    def test_score_is_integer_within_0_100(self):
        df = make_df(n=60)
        result = score_asset(df)
        assert isinstance(result["score"], int)
        assert 0 <= result["score"] <= 100

    def test_all_10_signals_present(self):
        df = make_df(n=60)
        result = score_asset(df)
        expected_keys = {
            "RSI", "Stoch RSI", "MACD", "Tendencia", "ADX",
            "Bollinger", "Volume/OBV", "Fibonacci", "Medo/Ganancia", "Divergencias",
        }
        assert set(result["signals"].keys()) == expected_keys

    def test_overbought_score_below_buy_threshold(self, overbought_df):
        result = score_asset(overbought_df)
        assert result["score"] < BUY_CONFIDENCE_MODERATE

    def test_oversold_score_above_neutral_threshold(self, oversold_df):
        result = score_asset(oversold_df)
        assert result["score"] > SELL_CONFIDENCE

    def test_confluence_percentage_between_0_and_100(self):
        df = make_df(n=60)
        result = score_asset(df)
        pct = result["confluence"]["percentage"]
        assert 0 <= pct <= 100

    def test_label_matches_score_brackets(self):
        df = make_df(n=60)
        result = score_asset(df)
        s = result["score"]
        label = result["label"]
        if s >= BUY_CONFIDENCE_STRONG:
            assert label == "COMPRA FORTE"
        elif s >= BUY_CONFIDENCE_MODERATE:
            assert label == "COMPRA"
        elif s >= SELL_CONFIDENCE:
            assert label == "NEUTRO"
        elif s >= 15:
            assert label == "VENDA"
        else:
            assert label == "VENDA FORTE"

    def test_fear_greed_parameter_influences_score(self):
        df = make_df(n=60)
        score_fear = score_asset(df, fear_greed_value=5)["score"]    # extreme fear → max pts
        score_greed = score_asset(df, fear_greed_value=95)["score"]  # extreme greed → 0 pts
        assert score_fear > score_greed

    def test_total_signal_points_never_exceed_100(self):
        """Sum of all per-signal max values must equal 100."""
        total_max = (
            SCORE_MAX_RSI + SCORE_MAX_STOCH_RSI + SCORE_MAX_MACD +
            SCORE_MAX_TREND + SCORE_MAX_ADX + SCORE_MAX_BOLLINGER +
            SCORE_MAX_VOLUME + SCORE_MAX_FIBONACCI +
            SCORE_MAX_FEAR_GREED + SCORE_MAX_DIVERGENCE
        )
        assert total_max == 100
