# ============================================================
# tests/test_database.py — Testes do modulo database.py
# ============================================================
#
# Usa banco SQLite em memoria (:memory:) para isolar os testes
# sem criar arquivos em disco. A fixture `db` reinicia o estado
# do modulo a cada teste, garantindo isolamento completo.
# ============================================================

import os
import sys
import json
import pytest
import pandas as pd
from datetime import date, timedelta
from unittest.mock import patch, MagicMock


# ============================================================
# Patch de ambiente: aponta para banco em memoria antes de
# importar database.py, para evitar criar arquivo em disco.
# ============================================================

os.environ.setdefault("BOTCRIPTO_DB_URL", "sqlite:///:memory:")


@pytest.fixture(autouse=True)
def db(monkeypatch):
    """
    Reinicia o engine e as tabelas a cada teste.

    Garante que cada teste comece com um banco vazio e que
    nao haja vazamento de estado entre testes.
    """
    import database as db_module

    # Reinicia flags de inicializacao
    monkeypatch.setattr(db_module, "_initialized", False)

    # Recria engine em memoria
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    db_module.Base.metadata.create_all(bind=engine)

    monkeypatch.setattr(db_module, "engine", engine)
    monkeypatch.setattr(db_module, "SessionLocal", SessionLocal)
    monkeypatch.setattr(db_module, "_initialized", True)

    yield db_module


# ============================================================
# Helpers
# ============================================================

def _make_ohlcv(n: int = 5, close: float = 100.0) -> pd.DataFrame:
    """DataFrame OHLCV minimo para testes de save_price_history."""
    dates = pd.date_range(end="2025-01-10", periods=n, freq="D")
    return pd.DataFrame({
        "Open": close, "High": close * 1.01,
        "Low": close * 0.99, "Close": close, "Volume": 1000.0,
    }, index=dates)


def _make_score_item(
    asset_id: str = "bitcoin",
    score: int = 60,
    rsi_div: str = "none",
    macd_div: str = "none",
) -> dict:
    """Item de score no formato produzido por analyze_all_assets()."""
    return {
        "id": asset_id,
        "Ativo": asset_id.title(),
        "type": "crypto",
        "Score": score,
        "Sinal": "COMPRA",
        "_score_result": {
            "score": score,
            "label": "COMPRA",
            "trend": "alta",
            "trend_strength": 5,
            "confluence": {"agree_buy": 6, "total": 10},
            "divergences": {
                "rsi": {"type": rsi_div, "description": ""},
                "macd": {"type": macd_div, "description": ""},
            },
            "signals": {},
        },
        "_dip_info": {},
        "_df": None,
    }


# ============================================================
# init_db
# ============================================================

class TestInitDb:
    def test_creates_tables(self, db):
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
        assert "price_history" in tables
        assert "score_snapshots" in tables
        assert "alert_log" in tables
        assert "portfolio" in tables

    def test_idempotent(self, db):
        """Chamar init_db() multiplas vezes nao levanta excecao."""
        db.init_db()
        db.init_db()


# ============================================================
# save_price_history
# ============================================================

class TestSavePriceHistory:
    def test_inserts_rows(self, db):
        df = _make_ohlcv(5)
        inserted = db.save_price_history("bitcoin", "crypto", df)
        assert inserted == 5

    def test_skips_duplicate_dates(self, db):
        df = _make_ohlcv(5)
        db.save_price_history("bitcoin", "crypto", df)
        inserted2 = db.save_price_history("bitcoin", "crypto", df)
        assert inserted2 == 0

    def test_returns_zero_for_empty_df(self, db):
        assert db.save_price_history("bitcoin", "crypto", pd.DataFrame()) == 0

    def test_returns_zero_when_close_missing(self, db):
        df = pd.DataFrame({"Open": [100], "Volume": [1000]},
                          index=pd.date_range("2025-01-01", periods=1))
        assert db.save_price_history("bitcoin", "crypto", df) == 0

    def test_returns_zero_for_none(self, db):
        assert db.save_price_history("bitcoin", "crypto", None) == 0

    def test_different_assets_stored_separately(self, db):
        df = _make_ohlcv(3)
        db.save_price_history("bitcoin", "crypto", df)
        db.save_price_history("ethereum", "crypto", df)
        btc = db.load_price_history("bitcoin")
        eth = db.load_price_history("ethereum")
        assert len(btc) == 3
        assert len(eth) == 3


# ============================================================
# load_price_history
# ============================================================

class TestLoadPriceHistory:
    def test_returns_empty_df_when_no_data(self, db):
        result = db.load_price_history("nonexistent")
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_returns_correct_columns(self, db):
        df = _make_ohlcv(3)
        db.save_price_history("bitcoin", "crypto", df)
        result = db.load_price_history("bitcoin")
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col in result.columns

    def test_sorted_ascending(self, db):
        df = _make_ohlcv(5)
        db.save_price_history("bitcoin", "crypto", df)
        result = db.load_price_history("bitcoin")
        assert list(result.index) == sorted(result.index)


# ============================================================
# save_scores_snapshot
# ============================================================

class TestSaveScoresSnapshot:
    def test_inserts_new_scores(self, db):
        scores = [_make_score_item("bitcoin", 80), _make_score_item("ethereum", 55)]
        inserted = db.save_scores_snapshot(scores)
        assert inserted == 2

    def test_skips_duplicate_assets_same_day(self, db):
        scores = [_make_score_item("bitcoin", 80)]
        db.save_scores_snapshot(scores)
        inserted2 = db.save_scores_snapshot(scores)
        assert inserted2 == 0

    def test_allows_same_asset_different_day(self, db):
        scores = [_make_score_item("bitcoin", 80)]
        yesterday = date.today() - timedelta(days=1)
        db.save_scores_snapshot(scores, snapshot_date=yesterday)
        inserted2 = db.save_scores_snapshot(scores, snapshot_date=date.today())
        assert inserted2 == 1

    def test_returns_zero_for_empty_list(self, db):
        assert db.save_scores_snapshot([]) == 0

    def test_persists_rsi_divergence(self, db):
        scores = [_make_score_item("bitcoin", 80, rsi_div="bullish")]
        db.save_scores_snapshot(scores)
        hist = db.load_score_history("bitcoin")
        assert hist.iloc[0]["rsi_divergence"] == "bullish"

    def test_skips_item_without_id(self, db):
        scores = [{"Score": 70, "_score_result": {"score": 70}}]
        inserted = db.save_scores_snapshot(scores)
        assert inserted == 0


# ============================================================
# load_score_history
# ============================================================

class TestLoadScoreHistory:
    def test_returns_empty_when_no_data(self, db):
        result = db.load_score_history("nonexistent")
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_returns_correct_values(self, db):
        scores = [_make_score_item("bitcoin", 75)]
        db.save_scores_snapshot(scores)
        hist = db.load_score_history("bitcoin")
        assert not hist.empty
        assert hist.iloc[0]["score"] == 75


# ============================================================
# get_score_trend
# ============================================================

class TestGetScoreTrend:
    def test_returns_none_when_no_data(self, db):
        result = db.get_score_trend("nonexistent")
        assert result["latest_score"] is None
        assert result["delta"] is None
        assert result["direction"] is None

    def test_returns_none_delta_with_single_snapshot(self, db):
        db.save_scores_snapshot([_make_score_item("bitcoin", 60)])
        result = db.get_score_trend("bitcoin")
        assert result["latest_score"] == 60
        assert result["delta"] is None

    def test_direction_up(self, db):
        yesterday = date.today() - timedelta(days=1)
        db.save_scores_snapshot([_make_score_item("bitcoin", 50)], snapshot_date=yesterday)
        db.save_scores_snapshot([_make_score_item("bitcoin", 65)], snapshot_date=date.today())
        result = db.get_score_trend("bitcoin")
        assert result["delta"] == 15
        assert result["direction"] == "up"

    def test_direction_down(self, db):
        yesterday = date.today() - timedelta(days=1)
        db.save_scores_snapshot([_make_score_item("bitcoin", 70)], snapshot_date=yesterday)
        db.save_scores_snapshot([_make_score_item("bitcoin", 55)], snapshot_date=date.today())
        result = db.get_score_trend("bitcoin")
        assert result["delta"] == -15
        assert result["direction"] == "down"

    def test_direction_flat(self, db):
        yesterday = date.today() - timedelta(days=1)
        db.save_scores_snapshot([_make_score_item("bitcoin", 60)], snapshot_date=yesterday)
        db.save_scores_snapshot([_make_score_item("bitcoin", 60)], snapshot_date=date.today())
        result = db.get_score_trend("bitcoin")
        assert result["delta"] == 0
        assert result["direction"] == "flat"


# ============================================================
# record_alert
# ============================================================

class TestRecordAlert:
    def test_first_alert_returns_true(self, db):
        assert db.record_alert("bitcoin", "strong_buy") is True

    def test_duplicate_same_day_returns_false(self, db):
        db.record_alert("bitcoin", "strong_buy")
        assert db.record_alert("bitcoin", "strong_buy") is False

    def test_different_types_both_allowed(self, db):
        assert db.record_alert("bitcoin", "strong_buy") is True
        assert db.record_alert("bitcoin", "bullish_divergence_rsi") is True

    def test_different_assets_both_allowed(self, db):
        assert db.record_alert("bitcoin", "strong_buy") is True
        assert db.record_alert("ethereum", "strong_buy") is True

    def test_same_asset_type_different_day_allowed(self, db):
        yesterday = date.today() - timedelta(days=1)
        db.record_alert("bitcoin", "strong_buy", snapshot_date=yesterday)
        assert db.record_alert("bitcoin", "strong_buy", snapshot_date=date.today()) is True


# ============================================================
# load_alert_history
# ============================================================

class TestLoadAlertHistory:
    def test_returns_empty_when_no_alerts(self, db):
        result = db.load_alert_history()
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_returns_recent_alerts(self, db):
        db.record_alert("bitcoin", "strong_buy")
        result = db.load_alert_history(days=1)
        assert not result.empty
        assert result.iloc[0]["Ativo"] == "bitcoin"

    def test_excludes_old_alerts(self, db):
        old_date = date.today() - timedelta(days=60)
        db.record_alert("bitcoin", "strong_buy", snapshot_date=old_date)
        result = db.load_alert_history(days=30)
        assert result.empty


# ============================================================
# Portfolio
# ============================================================

class TestPortfolio:
    def test_save_and_load(self, db):
        db.save_portfolio_entry("bitcoin", "crypto", 0.5, 50000.0)
        result = db.load_portfolio()
        assert "bitcoin" in result
        assert result["bitcoin"]["quantity"] == 0.5
        assert result["bitcoin"]["buy_price"] == 50000.0
        assert result["bitcoin"]["type"] == "crypto"

    def test_upsert_updates_existing(self, db):
        db.save_portfolio_entry("bitcoin", "crypto", 0.5, 50000.0)
        db.save_portfolio_entry("bitcoin", "crypto", 1.0, 45000.0)
        result = db.load_portfolio()
        assert result["bitcoin"]["quantity"] == 1.0
        assert result["bitcoin"]["buy_price"] == 45000.0

    def test_save_multiple_assets(self, db):
        db.save_portfolio_entry("bitcoin", "crypto", 0.5, 50000.0)
        db.save_portfolio_entry("ethereum", "crypto", 2.0, 3000.0)
        db.save_portfolio_entry("PETR4.SA", "stock", 100, 38.5)
        result = db.load_portfolio()
        assert len(result) == 3

    def test_delete_entry(self, db):
        db.save_portfolio_entry("bitcoin", "crypto", 0.5, 50000.0)
        db.delete_portfolio_entry("bitcoin")
        result = db.load_portfolio()
        assert "bitcoin" not in result

    def test_delete_nonexistent_returns_true(self, db):
        """Deletar algo inexistente nao deve levantar erro."""
        assert db.delete_portfolio_entry("nonexistent") is True

    def test_clear_portfolio(self, db):
        db.save_portfolio_entry("bitcoin", "crypto", 0.5, 50000.0)
        db.save_portfolio_entry("ethereum", "crypto", 2.0, 3000.0)
        db.clear_portfolio()
        assert db.load_portfolio() == {}

    def test_load_empty_portfolio(self, db):
        result = db.load_portfolio()
        assert result == {}
