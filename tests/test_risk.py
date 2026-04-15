# ============================================================
# test_risk.py — unit tests for risk management functions
#
# Functions covered:
#   compute_risk_metrics
#   compute_dca_plan
#   compute_correlation_matrix
#   generate_recommendation
# ============================================================

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis import (
    compute_risk_metrics,
    compute_dca_plan,
    compute_correlation_matrix,
    generate_recommendation,
)
from config import ATR_STOP_MULTIPLIER, DEFAULT_PORTFOLIO_VALUE, DEFAULT_RISK_PER_TRADE
from tests.conftest import make_df


# ===========================================================================
# compute_risk_metrics
# ===========================================================================

class TestComputeRiskMetrics:
    def _df_with_fib(self, close=100.0, atr=2.0):
        """Build a df that has ATR and Fibonacci attrs populated."""
        df = make_df(close=close, atr=atr)
        df.attrs["fibonacci"] = {
            "levels": {
                "0%":    close * 1.20,
                "23.6%": close * 1.15,
                "38.2%": close * 1.10,
                "50%":   close * 1.05,
                "61.8%": close * 1.00,
                "78.6%": close * 0.95,
                "100%":  close * 0.80,
            },
            "swing_high": close * 1.25,
            "swing_low": close * 0.75,
        }
        return df

    def test_returns_dict_with_expected_keys(self):
        df = self._df_with_fib()
        result = compute_risk_metrics(df)
        expected_keys = {
            "preco_atual", "atr", "atr_percentual",
            "stop_loss", "risco_por_unidade",
            "tamanho_posicao", "valor_posicao", "risco_maximo",
            "take_profit_1", "take_profit_2", "take_profit_3",
            "risco_retorno",
        }
        assert set(result.keys()) == expected_keys

    def test_stop_loss_formula(self):
        close = 100.0
        atr = 2.0
        df = self._df_with_fib(close=close, atr=atr)
        result = compute_risk_metrics(df)
        expected_stop = round(close - (atr * ATR_STOP_MULTIPLIER), 2)
        assert result["stop_loss"] == expected_stop

    def test_position_value_equals_size_times_price(self):
        df = self._df_with_fib()
        result = compute_risk_metrics(df)
        assert abs(result["valor_posicao"] - result["tamanho_posicao"] * result["preco_atual"]) < 0.01

    def test_risk_amount_formula(self):
        close = 100.0
        atr = 2.0
        portfolio = 10_000.0
        risk_pct = 2.0
        df = self._df_with_fib(close=close, atr=atr)
        result = compute_risk_metrics(df, portfolio_value=portfolio, risk_pct=risk_pct)
        expected_risk = round(portfolio * (risk_pct / 100), 2)
        assert result["risco_maximo"] == expected_risk

    def test_empty_df_returns_empty_dict(self):
        assert compute_risk_metrics(pd.DataFrame()) == {}

    def test_missing_atr_column_returns_empty_dict(self):
        df = make_df()
        df.drop(columns=["atr"], inplace=True)
        assert compute_risk_metrics(df) == {}

    def test_zero_atr_returns_empty_dict(self):
        df = self._df_with_fib(atr=0.0)
        assert compute_risk_metrics(df) == {}

    def test_nan_atr_returns_empty_dict(self):
        df = self._df_with_fib()
        df["atr"] = np.nan
        assert compute_risk_metrics(df) == {}

    def test_take_profit_1_above_current_price(self):
        df = self._df_with_fib()
        result = compute_risk_metrics(df)
        assert result["take_profit_1"] > result["preco_atual"]

    def test_risk_return_ratio_is_positive(self):
        df = self._df_with_fib()
        result = compute_risk_metrics(df)
        assert result["risco_retorno"] > 0

    def test_custom_portfolio_and_risk_pct(self):
        df = self._df_with_fib(close=200.0, atr=4.0)
        result = compute_risk_metrics(df, portfolio_value=50_000.0, risk_pct=1.0)
        assert result["risco_maximo"] == 500.0   # 1% of 50k


# ===========================================================================
# compute_dca_plan
# ===========================================================================

class TestComputeDCAPlan:
    def _df_with_fib(self, close=100.0):
        """
        Build a df whose Fibonacci levels are all BELOW the current price,
        as expected in a downtrend DCA scenario:
          38.2% < close, 50% < 38.2%, 61.8% < 50%
        """
        df = make_df(close=close)
        df.attrs["fibonacci"] = {
            "levels": {
                "0%":    close * 1.20,
                "23.6%": close * 1.05,
                "38.2%": close * 0.97,    # entry level 1
                "50%":   close * 0.93,    # entry level 2
                "61.8%": close * 0.88,    # entry level 3
                "78.6%": close * 0.82,
                "100%":  close * 0.75,
            },
            "swing_high": close * 1.25,
            "swing_low": close * 0.75,
        }
        return df

    def test_empty_df_returns_empty_list(self):
        assert compute_dca_plan(pd.DataFrame(), total_amount=1000) == []

    def test_returns_correct_number_of_tranches(self):
        df = self._df_with_fib()
        plan = compute_dca_plan(df, total_amount=1000, num_tranches=4)
        assert len(plan) == 4

    def test_tranche_numbers_are_sequential(self):
        df = self._df_with_fib()
        plan = compute_dca_plan(df, total_amount=1000, num_tranches=4)
        assert [p["tranche"] for p in plan] == [1, 2, 3, 4]

    def test_weights_sum_to_total_amount(self):
        total = 1000.0
        df = self._df_with_fib()
        plan = compute_dca_plan(df, total_amount=total, num_tranches=4)
        assert abs(sum(p["valor"] for p in plan) - total) < 0.10

    def test_each_tranche_has_required_keys(self):
        df = self._df_with_fib()
        plan = compute_dca_plan(df, total_amount=1000, num_tranches=4)
        for tranche in plan:
            assert "tranche" in tranche
            assert "preco" in tranche
            assert "valor" in tranche
            assert "percentual" in tranche
            assert "nivel" in tranche

    def test_fallback_without_fibonacci(self):
        df = make_df(close=100.0)
        df.attrs["fibonacci"] = {"levels": {}, "swing_high": 110, "swing_low": 90}
        plan = compute_dca_plan(df, total_amount=1000, num_tranches=4)
        assert len(plan) == 4
        # Fallback uses -3% steps: tranche 1 at current price, tranche 4 at -9%
        assert plan[0]["preco"] == round(100.0, 2)
        assert plan[3]["preco"] == round(100.0 * (1 - 3 * 0.03), 2)

    def test_fallback_equal_distribution(self):
        df = make_df(close=100.0)
        df.attrs["fibonacci"] = {"levels": {}, "swing_high": 110, "swing_low": 90}
        plan = compute_dca_plan(df, total_amount=1000, num_tranches=4)
        for tranche in plan:
            assert tranche["valor"] == round(1000 / 4, 2)

    def test_fib_plan_later_tranches_are_cheaper(self):
        """DCA plan should buy at progressively lower Fibonacci levels."""
        df = self._df_with_fib(close=100.0)
        plan = compute_dca_plan(df, total_amount=1000, num_tranches=4)
        prices = [p["preco"] for p in plan]
        assert prices[0] >= prices[1] >= prices[2] >= prices[3]

    def test_fib_plan_more_money_in_later_tranches(self):
        """Fibonacci plan allocates more capital to lower (cheaper) tranches."""
        df = self._df_with_fib(close=100.0)
        plan = compute_dca_plan(df, total_amount=1000, num_tranches=4)
        # weights [0.15, 0.25, 0.30, 0.30]: tranche 1 < tranche 4
        assert plan[0]["valor"] < plan[3]["valor"]


# ===========================================================================
# compute_correlation_matrix
# ===========================================================================

class TestComputeCorrelationMatrix:
    def _make_price_data(self, n=60):
        dates = pd.date_range(end="2025-01-01", periods=n, freq="D")
        return {
            "BTC":  pd.Series(np.random.randn(n).cumsum() + 100, index=dates),
            "ETH":  pd.Series(np.random.randn(n).cumsum() + 50, index=dates),
            "AAPL": pd.Series(np.random.randn(n).cumsum() + 200, index=dates),
        }

    def test_returns_dataframe(self):
        price_data = self._make_price_data()
        result = compute_correlation_matrix(price_data)
        assert isinstance(result, pd.DataFrame)

    def test_matrix_is_square(self):
        price_data = self._make_price_data()
        result = compute_correlation_matrix(price_data)
        assert result.shape[0] == result.shape[1]

    def test_diagonal_is_one(self):
        price_data = self._make_price_data()
        result = compute_correlation_matrix(price_data)
        for col in result.columns:
            assert abs(result.loc[col, col] - 1.0) < 1e-9

    def test_matrix_is_symmetric(self):
        price_data = self._make_price_data()
        result = compute_correlation_matrix(price_data)
        for a in result.columns:
            for b in result.columns:
                assert abs(result.loc[a, b] - result.loc[b, a]) < 1e-9

    def test_correlations_within_minus1_plus1(self):
        price_data = self._make_price_data()
        result = compute_correlation_matrix(price_data)
        assert (result.values >= -1.0 - 1e-9).all()
        assert (result.values <= 1.0 + 1e-9).all()

    def test_single_asset_returns_empty_df(self):
        dates = pd.date_range(end="2025-01-01", periods=60, freq="D")
        price_data = {"BTC": pd.Series(range(60), index=dates, dtype=float)}
        result = compute_correlation_matrix(price_data)
        assert result.empty

    def test_empty_dict_returns_empty_df(self):
        result = compute_correlation_matrix({})
        assert result.empty

    def test_perfectly_correlated_assets(self):
        """Two identical series → correlation = 1.0."""
        dates = pd.date_range(end="2025-01-01", periods=60, freq="D")
        prices = pd.Series(np.linspace(100, 200, 60), index=dates)
        price_data = {"A": prices, "B": prices * 2}
        result = compute_correlation_matrix(price_data)
        assert abs(result.loc["A", "B"] - 1.0) < 1e-6

    def test_series_shorter_than_20_rows_excluded(self):
        """Assets with fewer than 20 data points should be excluded."""
        dates_long = pd.date_range(end="2025-01-01", periods=60, freq="D")
        dates_short = pd.date_range(end="2025-01-01", periods=10, freq="D")
        price_data = {
            "BTC": pd.Series(np.random.randn(60).cumsum(), index=dates_long),
            "SHORT": pd.Series(np.random.randn(10).cumsum(), index=dates_short),
        }
        result = compute_correlation_matrix(price_data)
        # "SHORT" should be excluded; only 1 asset remains → empty
        assert result.empty or "SHORT" not in result.columns


# ===========================================================================
# generate_recommendation
# ===========================================================================

class TestGenerateRecommendation:
    def _score_result(self, score, label, trend="alta", agree=7):
        return {
            "score": score,
            "label": label,
            "trend": trend,
            "trend_strength": 60,
            "confluence": {"agree_buy": agree, "total": 10, "percentage": agree * 10},
            "divergences": {
                "rsi": {"type": "none", "strength": 0, "description": ""},
                "macd": {"type": "none", "strength": 0, "description": ""},
            },
        }

    def _dip_info(self, dip_type="ruido"):
        return {
            "type": dip_type,
            "drawdown": 5.0,
            "explanation": "Test explanation.",
        }

    def test_returns_string(self):
        result = generate_recommendation(
            self._score_result(75, "COMPRA FORTE"),
            self._dip_info(),
            "BTC",
        )
        assert isinstance(result, str)

    def test_asset_name_in_output(self):
        result = generate_recommendation(
            self._score_result(75, "COMPRA FORTE"),
            self._dip_info(),
            "BITCOIN",
        )
        assert "BITCOIN" in result

    def test_score_in_output(self):
        result = generate_recommendation(
            self._score_result(75, "COMPRA FORTE"),
            self._dip_info(),
            "BTC",
        )
        assert "75" in result

    def test_strong_buy_recommendation_present(self):
        result = generate_recommendation(
            self._score_result(80, "COMPRA FORTE"),
            self._dip_info("ruido"),
            "ETH",
        )
        assert "RECOMENDACAO" in result

    def test_sell_recommendation_present_for_low_score(self):
        result = generate_recommendation(
            self._score_result(20, "VENDA FORTE", trend="baixa"),
            self._dip_info("queda_forte"),
            "MGLU3",
        )
        assert "NAO COMPRE" in result or "Sinais negativos" in result

    def test_bullish_divergence_mentioned_when_present(self):
        score = self._score_result(80, "COMPRA FORTE")
        score["divergences"]["rsi"]["type"] = "bullish"
        result = generate_recommendation(score, self._dip_info(), "SOL")
        assert "DIVERGENCIA BULLISH" in result

    def test_bearish_divergence_mentioned_when_present(self):
        score = self._score_result(20, "VENDA", trend="baixa")
        score["divergences"]["macd"]["type"] = "bearish"
        result = generate_recommendation(score, self._dip_info("queda_forte"), "SOL")
        assert "DIVERGENCIA BEARISH" in result
