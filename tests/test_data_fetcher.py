# ============================================================
# test_data_fetcher.py — unit tests for data_fetcher.py
#
# All external network calls (requests.get, yfinance) are
# mocked so the tests run fully offline and deterministically.
#
# Functions covered:
#   RateLimiter.wait
#   _coingecko_get  (retry logic: 429, Timeout, ConnectionError)
#   fetch_crypto_history  (response parsing + OHLC construction)
#   fetch_fear_greed      (response parsing + empty fallback)
#   get_fear_greed_current
#   fetch_news            (sentiment scoring)
# ============================================================

import sys
import os
import json
import time
import threading
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, call
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# conftest.py already patched streamlit before this import
from data_fetcher import (
    RateLimiter,
    _coingecko_get,
    fetch_crypto_history,
    fetch_fear_greed,
    get_fear_greed_current,
    fetch_news,
    POSITIVE_WORDS,
    NEGATIVE_WORDS,
)


# ===========================================================================
# RateLimiter
# ===========================================================================

class TestRateLimiter:
    def test_does_not_sleep_below_limit(self):
        limiter = RateLimiter(max_calls=5, period=60.0)
        with patch("time.sleep") as mock_sleep:
            for _ in range(4):
                limiter.wait()
            mock_sleep.assert_not_called()

    def test_sleeps_when_limit_reached(self):
        limiter = RateLimiter(max_calls=3, period=60.0)
        # Pre-fill the call log with 3 recent timestamps
        with limiter.lock:
            now = time.time()
            limiter.calls = [now, now, now]

        with patch("time.sleep") as mock_sleep:
            with patch("time.time", return_value=now + 1.0):
                limiter.wait()
            mock_sleep.assert_called_once()

    def test_old_calls_are_pruned(self):
        limiter = RateLimiter(max_calls=3, period=10.0)
        with limiter.lock:
            # Two calls that are outside the window (> 10s ago)
            past = time.time() - 20.0
            limiter.calls = [past, past]

        with patch("time.sleep") as mock_sleep:
            limiter.wait()
        mock_sleep.assert_not_called()

    def test_thread_safety_no_exception(self):
        """Multiple threads calling wait() concurrently should not raise."""
        limiter = RateLimiter(max_calls=100, period=60.0)
        errors = []

        def worker():
            try:
                limiter.wait()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


# ===========================================================================
# _coingecko_get — retry logic
# ===========================================================================

class TestCoinGeckoGet:
    def _ok_response(self, body=None):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = body or {}
        resp.raise_for_status = MagicMock()
        return resp

    def _rate_limited_response(self):
        resp = MagicMock()
        resp.status_code = 429
        resp.raise_for_status = MagicMock()
        return resp

    def test_success_on_first_attempt(self):
        with patch("data_fetcher._cg_limiter") as mock_lim, \
             patch("requests.get", return_value=self._ok_response({"k": "v"})) as mock_get, \
             patch("time.sleep"):
            mock_lim.wait = MagicMock()
            resp = _coingecko_get("http://fake-url")
        assert resp.status_code == 200
        assert mock_get.call_count == 1

    def test_retries_on_429_then_succeeds(self):
        responses = [self._rate_limited_response(), self._ok_response()]
        with patch("data_fetcher._cg_limiter") as mock_lim, \
             patch("requests.get", side_effect=responses), \
             patch("time.sleep"):
            mock_lim.wait = MagicMock()
            resp = _coingecko_get("http://fake-url")
        assert resp.status_code == 200

    def test_raises_after_4_consecutive_429s(self):
        responses = [self._rate_limited_response()] * 4
        with patch("data_fetcher._cg_limiter") as mock_lim, \
             patch("requests.get", side_effect=responses), \
             patch("time.sleep"):
            mock_lim.wait = MagicMock()
            with pytest.raises(requests.exceptions.HTTPError):
                _coingecko_get("http://fake-url")

    def test_retries_on_timeout_then_succeeds(self):
        responses = [
            requests.exceptions.Timeout,
            self._ok_response(),
        ]
        with patch("data_fetcher._cg_limiter") as mock_lim, \
             patch("requests.get", side_effect=responses), \
             patch("time.sleep"):
            mock_lim.wait = MagicMock()
            resp = _coingecko_get("http://fake-url")
        assert resp.status_code == 200

    def test_raises_timeout_after_all_retries(self):
        with patch("data_fetcher._cg_limiter") as mock_lim, \
             patch("requests.get", side_effect=requests.exceptions.Timeout), \
             patch("time.sleep"):
            mock_lim.wait = MagicMock()
            with pytest.raises(requests.exceptions.Timeout):
                _coingecko_get("http://fake-url")

    def test_retries_on_connection_error_then_succeeds(self):
        responses = [
            requests.exceptions.ConnectionError,
            self._ok_response(),
        ]
        with patch("data_fetcher._cg_limiter") as mock_lim, \
             patch("requests.get", side_effect=responses), \
             patch("time.sleep"):
            mock_lim.wait = MagicMock()
            resp = _coingecko_get("http://fake-url")
        assert resp.status_code == 200


# ===========================================================================
# fetch_crypto_history — response parsing and OHLC construction
# ===========================================================================

class TestFetchCryptoHistory:
    def _make_response(self, n=30):
        """Build a fake CoinGecko /market_chart response."""
        base_ts = 1_700_000_000_000  # ms
        prices  = [[base_ts + i * 86_400_000, 100.0 + i] for i in range(n)]
        volumes = [[base_ts + i * 86_400_000, 1_000.0 + i * 10] for i in range(n)]
        resp = MagicMock()
        resp.json.return_value = {"prices": prices, "total_volumes": volumes}
        resp.raise_for_status = MagicMock()
        return resp

    def test_returns_dataframe_with_correct_columns(self):
        with patch("data_fetcher._coingecko_get", return_value=self._make_response()):
            df = fetch_crypto_history("bitcoin", days=30)
        assert isinstance(df, pd.DataFrame)
        for col in ("Close", "Open", "High", "Low", "Volume"):
            assert col in df.columns

    def test_no_duplicate_dates(self):
        with patch("data_fetcher._coingecko_get", return_value=self._make_response()):
            df = fetch_crypto_history("bitcoin", days=30)
        assert df.index.duplicated().sum() == 0

    def test_open_is_previous_close(self):
        with patch("data_fetcher._coingecko_get", return_value=self._make_response(30)):
            df = fetch_crypto_history("bitcoin", days=30)
        # Open[i] == Close[i-1] for i > 0
        assert df["Open"].iloc[1] == df["Close"].iloc[0]

    def test_high_gte_close_and_open(self):
        with patch("data_fetcher._coingecko_get", return_value=self._make_response(30)):
            df = fetch_crypto_history("bitcoin", days=30)
        assert (df["High"] >= df["Close"]).all()
        assert (df["High"] >= df["Open"]).all()

    def test_low_lte_close_and_open(self):
        with patch("data_fetcher._coingecko_get", return_value=self._make_response(30)):
            df = fetch_crypto_history("bitcoin", days=30)
        assert (df["Low"] <= df["Close"]).all()
        assert (df["Low"] <= df["Open"]).all()

    def test_empty_prices_returns_empty_df(self):
        resp = MagicMock()
        resp.json.return_value = {"prices": [], "total_volumes": []}
        with patch("data_fetcher._coingecko_get", return_value=resp):
            df = fetch_crypto_history("bitcoin", days=30)
        assert df.empty

    def test_api_error_returns_empty_df(self):
        with patch("data_fetcher._coingecko_get", side_effect=Exception("API error")):
            df = fetch_crypto_history("bitcoin", days=30)
        assert df.empty

    def test_index_is_datetime(self):
        with patch("data_fetcher._coingecko_get", return_value=self._make_response(30)):
            df = fetch_crypto_history("bitcoin", days=30)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_volume_column_populated(self):
        with patch("data_fetcher._coingecko_get", return_value=self._make_response(30)):
            df = fetch_crypto_history("bitcoin", days=30)
        assert df["Volume"].sum() > 0


# ===========================================================================
# fetch_fear_greed — response parsing
# ===========================================================================

class TestFetchFearGreed:
    def _make_response(self, n=5):
        now = int(time.time())
        data = [
            {
                "timestamp": str(now - i * 86_400),
                "value": str(30 + i * 5),
                "value_classification": "Fear",
            }
            for i in range(n)
        ]
        resp = MagicMock()
        resp.json.return_value = {"data": data}
        resp.raise_for_status = MagicMock()
        return resp

    def test_returns_dataframe(self):
        with patch("requests.get", return_value=self._make_response()):
            df = fetch_fear_greed()
        assert isinstance(df, pd.DataFrame)

    def test_returns_correct_number_of_rows(self):
        with patch("requests.get", return_value=self._make_response(5)):
            df = fetch_fear_greed()
        assert len(df) == 5

    def test_has_expected_columns(self):
        with patch("requests.get", return_value=self._make_response()):
            df = fetch_fear_greed()
        assert "value" in df.columns
        assert "classification" in df.columns
        assert "date" in df.columns

    def test_value_column_is_integer(self):
        with patch("requests.get", return_value=self._make_response()):
            df = fetch_fear_greed()
        assert df["value"].dtype in (int, "int64")

    def test_api_error_returns_empty_df(self):
        with patch("requests.get", side_effect=Exception("network error")):
            df = fetch_fear_greed()
        assert df.empty


# ===========================================================================
# get_fear_greed_current
# ===========================================================================

class TestGetFearGreedCurrent:
    def test_returns_tuple_of_int_and_str(self):
        fake_df = pd.DataFrame([
            {"date": "2025-01-01", "value": 25, "classification": "Fear"}
        ])
        with patch("data_fetcher.fetch_fear_greed", return_value=fake_df):
            value, label = get_fear_greed_current()
        assert isinstance(value, (int, np.integer))
        assert isinstance(label, str)

    def test_returns_first_row_values(self):
        fake_df = pd.DataFrame([
            {"date": "2025-01-02", "value": 42, "classification": "Fear"},
            {"date": "2025-01-01", "value": 35, "classification": "Fear"},
        ])
        with patch("data_fetcher.fetch_fear_greed", return_value=fake_df):
            value, label = get_fear_greed_current()
        assert value == 42
        assert label == "Fear"

    def test_empty_df_returns_neutral_fallback(self):
        with patch("data_fetcher.fetch_fear_greed", return_value=pd.DataFrame()):
            value, label = get_fear_greed_current()
        assert value == 50
        assert label == "Neutral"


# ===========================================================================
# fetch_news — sentiment scoring
# ===========================================================================

class TestFetchNewsSentiment:
    """
    Sentiment scoring is embedded inside fetch_news.  We test the logic
    by providing feed entries whose titles contain known positive/negative
    words and asserting the resulting sentiment value and label.
    """

    def _mock_feed(self, title: str):
        entry = MagicMock()
        entry.get = lambda key, default="": {
            "title": title, "summary": "", "link": "http://test.com",
            "published": "Mon, 01 Jan 2025 00:00:00 +0000",
        }.get(key, default)
        feed = MagicMock()
        feed.entries = [entry]
        return feed

    def test_positive_title_gets_sentiment_above_0_6(self):
        title = "Bitcoin surge rally bull gain"
        with patch("feedparser.parse", return_value=self._mock_feed(title)):
            articles = fetch_news()
        assert len(articles) > 0
        assert articles[0]["sentiment"] > 0.6
        assert articles[0]["sentiment_label"] == "Positivo"

    def test_negative_title_gets_sentiment_below_0_4(self):
        title = "Bitcoin crash dump bear loss hack scam"
        with patch("feedparser.parse", return_value=self._mock_feed(title)):
            articles = fetch_news()
        assert len(articles) > 0
        assert articles[0]["sentiment"] < 0.4
        assert articles[0]["sentiment_label"] == "Negativo"

    def test_neutral_title_gets_sentiment_around_0_5(self):
        # Equal number of positive and negative words → sentiment = 0.5
        title = "rally crash"   # 1 positive, 1 negative → 0.5
        with patch("feedparser.parse", return_value=self._mock_feed(title)):
            articles = fetch_news()
        assert len(articles) > 0
        assert articles[0]["sentiment"] == 0.5
        assert articles[0]["sentiment_label"] == "Neutro"

    def test_no_sentiment_words_gets_neutral(self):
        # Carefully chosen sentence — verified to contain no substring from
        # either POSITIVE_WORDS or NEGATIVE_WORDS (which use substring matching).
        title = "Astronomers discover three new planets in distant galaxy"
        with patch("feedparser.parse", return_value=self._mock_feed(title)):
            articles = fetch_news()
        # 0 positive + 0 negative → total=0 → sentiment = 0.5
        assert articles[0]["sentiment"] == 0.5

    def test_feed_error_is_swallowed(self):
        with patch("feedparser.parse", side_effect=Exception("feed error")):
            articles = fetch_news()
        assert articles == []

    def test_article_has_required_keys(self):
        title = "crypto rally surge"
        with patch("feedparser.parse", return_value=self._mock_feed(title)):
            articles = fetch_news()
        for key in ("title", "url", "sentiment", "sentiment_label", "date", "source"):
            assert key in articles[0]

    def test_positive_words_set_is_non_empty(self):
        assert len(POSITIVE_WORDS) > 0

    def test_negative_words_set_is_non_empty(self):
        assert len(NEGATIVE_WORDS) > 0

    def test_positive_and_negative_sets_are_disjoint(self):
        overlap = POSITIVE_WORDS & NEGATIVE_WORDS
        assert overlap == set(), f"Overlapping words: {overlap}"
