# ============================================================
# tests/test_backtesting.py — Testes da logica de simulacao
# ============================================================
#
# Testa _simulate_strategy() isoladamente, sem Streamlit.
# ============================================================

import sys
from unittest.mock import MagicMock

# Stub Streamlit (ja feito em conftest, mas importado aqui por garantia)
if "streamlit" not in sys.modules:
    _st = MagicMock()
    _st.cache_data = lambda *a, **kw: (a[0] if a and callable(a[0]) else (lambda f: f))
    sys.modules["streamlit"] = _st

import pytest
import numpy as np
import pandas as pd
from datetime import date, timedelta


# ── Importa a funcao diretamente do modulo ──────────────────────────────
# app.py nao e importavel facilmente em testes (set_page_config no nivel
# de modulo), entao copiamos a logica em um helper de teste equivalente.

def _simulate(score_hist, price_hist, buy_thresh=72, sell_thresh=30, capital=10000.0):
    """Wrapper que importa _simulate_strategy de app.py com guards."""
    import importlib, types
    # Garante que o modulo app seja importavel sem travar o Streamlit
    st_mock = sys.modules.get("streamlit")
    if st_mock:
        st_mock.set_page_config = MagicMock()
        st_mock.markdown = MagicMock()
        st_mock.sidebar = MagicMock()
        st_mock.sidebar.markdown = MagicMock()
        st_mock.sidebar.radio = MagicMock(return_value="▸  01   Visao Geral")

    # Importa somente a funcao, sem executar o corpo do modulo top-level
    import ast, types as t
    src = open("/home/user/botcripto/app.py").read()
    tree = ast.parse(src)
    # Extrai apenas a funcao _simulate_strategy
    func_src = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_simulate_strategy":
            func_src = ast.get_source_segment(src, node)
            break
    assert func_src, "_simulate_strategy not found in app.py"
    ns = {"pd": pd, "go": MagicMock()}
    exec(func_src, ns)
    return ns["_simulate_strategy"](score_hist, price_hist, buy_thresh, sell_thresh, capital)


def _make_score_df(dates, scores):
    return pd.DataFrame({"date": pd.to_datetime(dates), "score": scores})


def _make_price_df(dates, closes):
    idx = pd.to_datetime(dates)
    return pd.DataFrame({"Close": closes}, index=idx)


class TestSimulateStrategy:
    def _run(self, dates, scores, closes, buy=72, sell=30, capital=10000.0):
        sh = _make_score_df(dates, scores)
        ph = _make_price_df(dates, closes)
        return _simulate(sh, ph, buy, sell, capital)

    def test_no_signal_no_trades(self):
        dates = pd.date_range("2025-01-01", periods=10)
        scores = [50] * 10
        closes = [100.0] * 10
        trades, final, eq = self._run(dates, scores, closes)
        assert len(trades) == 0
        assert final == 10000.0

    def test_single_buy_sell_cycle(self):
        # Score starts low, crosses above 72, then crosses below 30
        dates = pd.date_range("2025-01-01", periods=5)
        scores = [50, 50, 75, 75, 20]   # Cross buy at index 2, sell at index 4
        closes = [100, 100, 100, 120, 120]
        trades, final, eq = self._run(dates, scores, closes, buy=72, sell=30)
        assert len(trades) == 1
        # Bought at 100, sold at 120 → +20% on 10000 = 12000
        assert abs(final - 12000.0) < 1.0

    def test_winning_trade_recorded_as_lucro(self):
        dates = pd.date_range("2025-01-01", periods=4)
        scores = [50, 75, 75, 20]
        closes = [100, 100, 150, 150]
        trades, _, _ = self._run(dates, scores, closes)
        assert trades[0]["Resultado"] == "Lucro"

    def test_losing_trade_recorded_as_prejuizo(self):
        dates = pd.date_range("2025-01-01", periods=4)
        scores = [50, 75, 75, 20]
        closes = [100, 100, 80, 80]
        trades, _, _ = self._run(dates, scores, closes)
        assert trades[0]["Resultado"] == "Prejuizo"

    def test_equity_curve_length_matches_data(self):
        dates = pd.date_range("2025-01-01", periods=10)
        scores = [50] * 10
        closes = [100.0] * 10
        _, _, eq = self._run(dates, scores, closes)
        assert len(eq) == 10

    def test_multiple_cycles(self):
        # Two buy-sell cycles
        dates = pd.date_range("2025-01-01", periods=10)
        scores = [50, 75, 75, 20, 50, 75, 75, 20, 50, 50]
        closes = [100] * 10
        trades, _, _ = self._run(dates, scores, closes)
        assert len(trades) == 2

    def test_empty_score_history_returns_empty(self):
        sh = pd.DataFrame({"date": [], "score": []})
        ph = _make_price_df(pd.date_range("2025-01-01", periods=3), [100, 100, 100])
        trades, capital, eq = _simulate(sh, ph, capital=10000.0)
        assert trades == []
        assert capital == 10000.0
        assert eq.empty

    def test_empty_price_history_returns_empty(self):
        sh = _make_score_df(pd.date_range("2025-01-01", periods=3), [50, 75, 20])
        ph = pd.DataFrame({"Close": []})
        ph.index = pd.DatetimeIndex([])
        trades, capital, eq = _simulate(sh, ph, capital=10000.0)
        assert trades == []
        assert capital == 10000.0

    def test_initial_capital_preserved_when_no_trades(self):
        dates = pd.date_range("2025-01-01", periods=5)
        scores = [40, 45, 50, 55, 60]   # Never crosses 72
        closes = [100.0] * 5
        trades, final, _ = self._run(dates, scores, closes)
        assert len(trades) == 0
        assert final == 10000.0

    def test_open_position_at_end_uses_last_price(self):
        # Buy and never sell (score stays high)
        dates = pd.date_range("2025-01-01", periods=5)
        scores = [50, 75, 80, 85, 90]   # Cross buy, never cross sell
        closes = [100, 100, 110, 120, 150]
        _, final, _ = self._run(dates, scores, closes, sell=20)
        # Bought at 100, last price 150 → final = 10000 * 1.5
        assert abs(final - 15000.0) < 1.0
