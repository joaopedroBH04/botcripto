# ============================================================
# tests/test_config.py — Testes de config.py
# ============================================================

import pytest
from config import (
    score_label,
    BUY_CONFIDENCE_STRONG,
    BUY_CONFIDENCE_MODERATE,
    SELL_CONFIDENCE,
)


class TestScoreLabel:
    def test_strong_buy_at_threshold(self):
        label, css = score_label(BUY_CONFIDENCE_STRONG)
        assert label == "COMPRA FORTE"
        assert css == "buy"

    def test_strong_buy_above_threshold(self):
        label, css = score_label(100)
        assert label == "COMPRA FORTE"
        assert css == "buy"

    def test_watch_at_moderate_threshold(self):
        label, css = score_label(BUY_CONFIDENCE_MODERATE)
        assert label == "OBSERVACAO"
        assert css == "watch"

    def test_watch_just_below_strong(self):
        label, css = score_label(BUY_CONFIDENCE_STRONG - 1)
        assert label == "OBSERVACAO"
        assert css == "watch"

    def test_neutral_at_sell_threshold(self):
        label, css = score_label(SELL_CONFIDENCE)
        assert label == "NEUTRO"
        assert css == "neutral"

    def test_neutral_just_below_moderate(self):
        label, css = score_label(BUY_CONFIDENCE_MODERATE - 1)
        assert label == "NEUTRO"
        assert css == "neutral"

    def test_sell_below_threshold(self):
        label, css = score_label(SELL_CONFIDENCE - 1)
        assert label == "VENDA"
        assert css == "sell"

    def test_sell_at_zero(self):
        label, css = score_label(0)
        assert label == "VENDA"
        assert css == "sell"

    def test_returns_tuple(self):
        result = score_label(50)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_all_classes_distinct(self):
        classes = {score_label(s)[1] for s in [0, SELL_CONFIDENCE, BUY_CONFIDENCE_MODERATE, BUY_CONFIDENCE_STRONG]}
        assert len(classes) == 4
