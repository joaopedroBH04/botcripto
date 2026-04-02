# ============================================================
# BotCripto - Modulo de coleta de dados
# CoinGecko, Yahoo Finance, Fear & Greed, RSS News
# Com rate limiting inteligente e retry automatico
# ============================================================

import time
import threading
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import feedparser
import streamlit as st
from datetime import datetime, timedelta
from config import (
    COINGECKO_BASE, FEAR_GREED_URL, CACHE_TTL, LOOKBACK_DAYS
)


# -------------------------------------------------------
# Rate Limiter global para CoinGecko
# (max 8 requests/minuto para ficar seguro no plano free)
# -------------------------------------------------------
class RateLimiter:
    """Controla a taxa de requisicoes para nao estourar o limite da API."""
    def __init__(self, max_calls: int = 8, period: float = 60.0):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.time()
            # Remove chamadas antigas (fora da janela)
            self.calls = [t for t in self.calls if now - t < self.period]
            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (now - self.calls[0]) + 0.5
                if sleep_time > 0:
                    time.sleep(sleep_time)
            self.calls.append(time.time())


# Conservador: max 5 chamadas por minuto (CoinGecko free permite ~10-30,
# mas na pratica bloqueia com menos dependendo do IP/regiao)
_cg_limiter = RateLimiter(max_calls=5, period=60.0)


def _coingecko_get(url: str, params: dict = None, timeout: int = 20) -> requests.Response:
    """
    Faz GET ao CoinGecko com rate limiting e retry automatico.
    Tenta ate 4 vezes com espera generosa em caso de 429.
    """
    _cg_limiter.wait()

    max_retries = 4
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)

            if resp.status_code == 429:
                # Rate limited - espera progressiva: 20s, 40s, 60s, 80s
                wait_time = (attempt + 1) * 20
                time.sleep(wait_time)
                _cg_limiter.wait()
                continue

            resp.raise_for_status()
            return resp

        except requests.exceptions.HTTPError as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 20
                time.sleep(wait_time)
                continue
            raise
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            raise
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                time.sleep(10)
                continue
            raise

    raise requests.exceptions.HTTPError("CoinGecko: Limite de requisicoes excedido apos 4 tentativas")


# -------------------------------------------------------
# Criptomoedas (CoinGecko)
# -------------------------------------------------------

@st.cache_data(ttl=CACHE_TTL)
def fetch_crypto_current(coin_ids: list[str]) -> pd.DataFrame:
    """Busca dados atuais de multiplas criptos de uma vez (1 requisicao)."""
    try:
        ids_str = ",".join(coin_ids)
        url = f"{COINGECKO_BASE}/coins/markets"
        params = {
            "vs_currency": "usd",
            "ids": ids_str,
            "order": "market_cap_desc",
            "per_page": len(coin_ids),
            "page": 1,
            "sparkline": False,
            "price_change_percentage": "1h,24h,7d,30d",
        }
        resp = _coingecko_get(url, params=params)
        data = resp.json()

        rows = []
        for coin in data:
            rows.append({
                "id": coin["id"],
                "symbol": coin["symbol"].upper(),
                "name": coin["name"],
                "price": coin["current_price"],
                "market_cap": coin["market_cap"],
                "volume_24h": coin["total_volume"],
                "change_1h": coin.get("price_change_percentage_1h_in_currency", 0) or 0,
                "change_24h": coin.get("price_change_percentage_24h_in_currency", 0) or 0,
                "change_7d": coin.get("price_change_percentage_7d_in_currency", 0) or 0,
                "change_30d": coin.get("price_change_percentage_30d_in_currency", 0) or 0,
                "ath": coin.get("ath", 0) or 0,
                "ath_change": coin.get("ath_change_percentage", 0) or 0,
                "type": "crypto",
            })
        return pd.DataFrame(rows)
    except Exception as e:
        st.warning(f"⚠️ Erro ao buscar dados de criptomoedas: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=CACHE_TTL * 3)  # Cache 30 min
def fetch_crypto_history(coin_id: str, days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Busca historico diario de precos via /market_chart (1 chamada por ativo).
    Constroi OHLCV deterministico a partir dos dados de Close + Volume.
    Evita o endpoint /ohlc que gera timestamps duplicados ao agregar de 4h para 1d.
    """
    try:
        url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
        params = {"vs_currency": "usd", "days": days, "interval": "daily"}
        resp = _coingecko_get(url, params=params)
        data = resp.json()

        prices  = data.get("prices", [])
        volumes = data.get("total_volumes", [])

        if not prices:
            return pd.DataFrame()

        df = pd.DataFrame(prices, columns=["timestamp", "Close"])
        df["Volume"] = [v[1] for v in volumes] if volumes and len(volumes) == len(prices) else 0.0
        df["Date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.normalize()
        df.drop(columns=["timestamp"], inplace=True)
        df.set_index("Date", inplace=True)

        # Remover eventuais datas duplicadas (mantem a ultima entrada do dia)
        df = df[~df.index.duplicated(keep="last")]
        df.sort_index(inplace=True)

        # Construir OHLC deterministico a partir do Close diario
        df["Open"] = df["Close"].shift(1).fillna(df["Close"])
        daily_range = df["Close"].pct_change().abs().fillna(0.005).clip(0.002, 0.04)
        df["High"] = df[["Open", "Close"]].max(axis=1) * (1 + daily_range * 0.5)
        df["Low"]  = df[["Open", "Close"]].min(axis=1) * (1 - daily_range * 0.5)

        return df

    except Exception as e:
        st.warning(f"⚠️ Erro ao buscar historico de {coin_id}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=CACHE_TTL * 3)  # Cache 30 min — dados diarios mudam pouco
def fetch_all_crypto_histories(coin_ids: list[str], days: int = LOOKBACK_DAYS) -> dict:
    """
    Busca historico de todas as criptos com rate limiting controlado.
    1 chamada por ativo (vs. 2 antes) + pausa de 2s entre ativos.
    """
    results = {}
    progress_bar = st.progress(0, text="Inicializando...")
    total = len(coin_ids)

    for i, coin_id in enumerate(coin_ids):
        pct = int((i / total) * 100)
        progress_bar.progress(pct, text=f"📊 {coin_id.title()} ({i+1}/{total})...")
        df = fetch_crypto_history(coin_id, days)
        if not df.empty:
            results[coin_id] = df
        if i < total - 1:
            time.sleep(2)  # 2s entre ativos (1 chamada cada, plano free suporta bem)

    progress_bar.progress(100, text="Concluido!")
    time.sleep(0.3)
    progress_bar.empty()
    return results


# -------------------------------------------------------
# Acoes e ETFs (Yahoo Finance)
# -------------------------------------------------------

@st.cache_data(ttl=CACHE_TTL)
def fetch_stock_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Busca historico OHLCV de uma acao/ETF."""
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index.name = "Date"
        # Garantir que o indice e DatetimeIndex sem timezone
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception as e:
        st.warning(f"⚠️ Erro ao buscar historico de {ticker}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=CACHE_TTL * 4)
def fetch_usd_brl() -> float:
    """Busca cotacao atual do dolar (USD/BRL) para conversao de precos."""
    try:
        df = yf.download("BRL=X", period="2d", progress=False, auto_adjust=True)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return float(df["Close"].iloc[-1])
        return 5.0  # Fallback
    except Exception:
        return 5.0


@st.cache_data(ttl=CACHE_TTL)
def fetch_stock_current(tickers: list[str]) -> pd.DataFrame:
    """Busca dados atuais de multiplas acoes."""
    rows = []
    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker)
            info = tk.info
            hist = tk.history(period="5d")
            if hist.empty:
                continue

            current_price = hist["Close"].iloc[-1]
            prev_price = hist["Close"].iloc[-2] if len(hist) > 1 else current_price
            change_24h = ((current_price - prev_price) / prev_price) * 100

            rows.append({
                "id": ticker,
                "symbol": ticker.replace(".SA", ""),
                "name": info.get("shortName", ticker),
                "price": current_price,
                "market_cap": info.get("marketCap", 0) or 0,
                "volume_24h": info.get("volume", 0) or 0,
                "change_1h": 0,
                "change_24h": change_24h,
                "change_7d": 0,
                "change_30d": 0,
                "ath": info.get("fiftyTwoWeekHigh", 0) or 0,
                "ath_change": 0,
                "type": "stock",
            })
        except Exception:
            continue
        time.sleep(0.3)  # Evitar rate limiting do Yahoo

    return pd.DataFrame(rows)


# -------------------------------------------------------
# Fear & Greed Index
# -------------------------------------------------------

@st.cache_data(ttl=CACHE_TTL)
def fetch_fear_greed() -> pd.DataFrame:
    """Busca o indice Fear & Greed dos ultimos 30 dias."""
    try:
        resp = requests.get(FEAR_GREED_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data", [])

        rows = []
        for item in data:
            rows.append({
                "date": datetime.fromtimestamp(int(item["timestamp"])),
                "value": int(item["value"]),
                "classification": item["value_classification"],
            })
        return pd.DataFrame(rows)
    except Exception as e:
        st.warning(f"⚠️ Erro ao buscar Fear & Greed Index: {e}")
        return pd.DataFrame()


def get_fear_greed_current() -> tuple[int, str]:
    """Retorna valor atual e classificacao do Fear & Greed."""
    df = fetch_fear_greed()
    if df.empty:
        return 50, "Neutral"
    return df.iloc[0]["value"], df.iloc[0]["classification"]


# -------------------------------------------------------
# BTC Dominance
# -------------------------------------------------------

@st.cache_data(ttl=CACHE_TTL * 2)
def fetch_btc_dominance() -> float:
    """Busca dominancia do Bitcoin no mercado."""
    try:
        url = f"{COINGECKO_BASE}/global"
        resp = _coingecko_get(url)
        data = resp.json().get("data", {})
        return data.get("market_cap_percentage", {}).get("btc", 0)
    except Exception:
        return 0.0


# -------------------------------------------------------
# Noticias (RSS feeds - sem API key)
# -------------------------------------------------------

POSITIVE_WORDS = {
    "surge", "rally", "bull", "gain", "high", "up", "growth", "profit",
    "recovery", "breakout", "adoption", "launch", "partnership", "upgrade",
    "approval", "bullish", "soar", "jump", "boost", "positive",
    "alta", "subiu", "ganho", "lucro", "positivo", "recorde",
}

NEGATIVE_WORDS = {
    "crash", "dump", "bear", "loss", "low", "down", "decline", "sell",
    "risk", "hack", "scam", "ban", "regulation", "fear", "warning",
    "bearish", "plunge", "drop", "collapse", "negative", "fraud",
    "queda", "caiu", "perda", "risco", "fraude", "negativo",
}

RSS_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=BTC-USD&region=US&lang=en-US",
]


@st.cache_data(ttl=600)  # Cache 10 min para noticias
def fetch_news() -> list[dict]:
    """Busca noticias de feeds RSS e calcula sentimento basico."""
    articles = []
    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:10]:
                title = entry.get("title", "")
                summary = entry.get("summary", "")
                text = f"{title} {summary}".lower()

                pos = sum(1 for w in POSITIVE_WORDS if w in text)
                neg = sum(1 for w in NEGATIVE_WORDS if w in text)
                total = pos + neg
                if total == 0:
                    sentiment = 0.5
                else:
                    sentiment = pos / total

                published = entry.get("published", "")
                articles.append({
                    "title": title,
                    "url": entry.get("link", ""),
                    "sentiment": sentiment,
                    "sentiment_label": "Positivo" if sentiment > 0.6 else ("Negativo" if sentiment < 0.4 else "Neutro"),
                    "date": published,
                    "source": feed_url.split("/")[2],
                })
        except Exception:
            continue

    return articles[:30]
