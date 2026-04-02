# ============================================================
# BotCripto v2 - Motor de Analise Tecnica e Scoring
# Indicadores: RSI, Stoch RSI, MACD, EMA, SMA, ADX, ATR,
#   Bollinger, OBV, Fibonacci, Divergencias, Confluencia
# ============================================================

import pandas as pd
import numpy as np
import ta
from config import (
    RSI_OVERSOLD, RSI_OVERBOUGHT,
    STOCH_RSI_OVERSOLD, STOCH_RSI_OVERBOUGHT,
    ADX_WEAK, ADX_STRONG, ADX_VERY_STRONG,
    SMA_PERIODS, EMA_PERIODS, ATR_PERIOD, ATR_STOP_MULTIPLIER,
    FIBONACCI_LEVELS, FIBONACCI_LABELS,
    DEFAULT_RISK_PER_TRADE, DEFAULT_PORTFOLIO_VALUE,
    SCORE_MAX_RSI, SCORE_MAX_STOCH_RSI, SCORE_MAX_MACD,
    SCORE_MAX_TREND, SCORE_MAX_ADX, SCORE_MAX_BOLLINGER,
    SCORE_MAX_VOLUME, SCORE_MAX_FIBONACCI, SCORE_MAX_FEAR_GREED,
    SCORE_MAX_DIVERGENCE,
    BUY_CONFIDENCE_STRONG, BUY_CONFIDENCE_MODERATE, SELL_CONFIDENCE,
)


# =============================================================
# INDICADORES TECNICOS
# =============================================================

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula TODOS os indicadores tecnicos sobre um DataFrame OHLCV."""
    if df.empty or len(df) < 30:
        return df

    df = df.copy()
    close = df["Close"].astype(float)
    high = df.get("High", close).astype(float)
    low = df.get("Low", close).astype(float)
    volume = df.get("Volume", pd.Series(0, index=df.index)).astype(float)

    # --- RSI (14) ---
    df["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    # --- Stochastic RSI ---
    stoch_rsi = ta.momentum.StochRSIIndicator(close, window=14, smooth1=3, smooth2=3)
    df["stoch_rsi_k"] = stoch_rsi.stochrsi_k() * 100  # 0-100
    df["stoch_rsi_d"] = stoch_rsi.stochrsi_d() * 100

    # --- SMAs ---
    for p in SMA_PERIODS:
        if len(df) >= p:
            df[f"sma_{p}"] = ta.trend.SMAIndicator(close, window=p).sma_indicator()

    # --- EMAs ---
    for p in EMA_PERIODS:
        df[f"ema_{p}"] = ta.trend.EMAIndicator(close, window=p).ema_indicator()

    # --- MACD ---
    macd = ta.trend.MACD(close)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # --- ADX (Average Directional Index) ---
    adx_ind = ta.trend.ADXIndicator(high, low, close, window=14)
    df["adx"] = adx_ind.adx()
    df["adx_pos"] = adx_ind.adx_pos()  # DI+
    df["adx_neg"] = adx_ind.adx_neg()  # DI-

    # --- Bollinger Bands ---
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()

    # --- ATR (Average True Range) ---
    df["atr"] = ta.volatility.AverageTrueRange(high, low, close, window=ATR_PERIOD).average_true_range()

    # --- OBV (On Balance Volume) ---
    if volume.sum() > 0:
        df["obv"] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        df["obv_sma_20"] = ta.trend.SMAIndicator(df["obv"].astype(float), window=20).sma_indicator()
        df["volume_sma_20"] = ta.trend.SMAIndicator(volume, window=20).sma_indicator()
    else:
        df["obv"] = 0
        df["obv_sma_20"] = 0
        df["volume_sma_20"] = 0

    # --- Fibonacci ---
    fib = compute_fibonacci(df)
    df.attrs["fibonacci"] = fib

    return df


# =============================================================
# FIBONACCI RETRACEMENT
# =============================================================

def compute_fibonacci(df: pd.DataFrame, lookback: int = 90) -> dict:
    """
    Calcula niveis de Fibonacci baseado no swing high/low recente.
    Retorna dict com niveis de preco para cada nivel Fibonacci.
    """
    if len(df) < lookback:
        lookback = len(df)

    recent = df.iloc[-lookback:]
    swing_high = recent["High"].max() if "High" in recent.columns else recent["Close"].max()
    swing_low = recent["Low"].min() if "Low" in recent.columns else recent["Close"].min()
    diff = swing_high - swing_low

    if diff == 0:
        return {"levels": {}, "swing_high": swing_high, "swing_low": swing_low}

    levels = {}
    for fib_level, label in zip(FIBONACCI_LEVELS, FIBONACCI_LABELS):
        # Retracoes sao calculadas do topo para baixo
        price = swing_high - (diff * fib_level)
        levels[label] = round(price, 4)

    return {
        "levels": levels,
        "swing_high": swing_high,
        "swing_low": swing_low,
    }


# =============================================================
# DETECCAO DE DIVERGENCIAS
# =============================================================

def _find_swing_lows(series: pd.Series, order: int = 5) -> list[tuple[int, float]]:
    """Encontra minimos locais (swing lows) em uma serie."""
    lows = []
    values = series.dropna().values
    indices = series.dropna().index

    for i in range(order, len(values) - order):
        if all(values[i] <= values[i - j] for j in range(1, order + 1)) and \
           all(values[i] <= values[i + j] for j in range(1, order + 1)):
            lows.append((i, values[i]))

    return lows


def _find_swing_highs(series: pd.Series, order: int = 5) -> list[tuple[int, float]]:
    """Encontra maximos locais (swing highs) em uma serie."""
    highs = []
    values = series.dropna().values

    for i in range(order, len(values) - order):
        if all(values[i] >= values[i - j] for j in range(1, order + 1)) and \
           all(values[i] >= values[i + j] for j in range(1, order + 1)):
            highs.append((i, values[i]))

    return highs


def detect_rsi_divergence(df: pd.DataFrame, lookback: int = 60) -> dict:
    """
    Detecta divergencia entre preco e RSI.
    - Bullish: preco faz novo low, RSI faz low mais alto (reversao para cima)
    - Bearish: preco faz novo high, RSI faz high mais baixo (reversao para baixo)
    """
    result = {"type": "none", "strength": 0, "description": "Sem divergencia detectada"}

    if "rsi_14" not in df.columns or len(df) < lookback:
        return result

    recent_close = df["Close"].iloc[-lookback:]
    recent_rsi = df["rsi_14"].iloc[-lookback:]

    # Buscar swing lows para divergencia bullish
    price_lows = _find_swing_lows(recent_close, order=5)
    rsi_lows = _find_swing_lows(recent_rsi, order=5)

    if len(price_lows) >= 2 and len(rsi_lows) >= 2:
        # Pegar os 2 lows mais recentes
        p1, p1_val = price_lows[-2]
        p2, p2_val = price_lows[-1]
        r1, r1_val = rsi_lows[-2]
        r2, r2_val = rsi_lows[-1]

        # Divergencia bullish: preco caiu mais, RSI subiu
        if p2_val < p1_val and r2_val > r1_val and abs(p2 - p1) >= 5:
            strength = min(int(abs(r2_val - r1_val) * 2), 100)
            return {
                "type": "bullish",
                "strength": strength,
                "description": f"Divergencia Bullish: preco fez novo fundo mas RSI subiu. Sinal forte de reversao para ALTA.",
            }

    # Buscar swing highs para divergencia bearish
    price_highs = _find_swing_highs(recent_close, order=5)
    rsi_highs = _find_swing_highs(recent_rsi, order=5)

    if len(price_highs) >= 2 and len(rsi_highs) >= 2:
        p1, p1_val = price_highs[-2]
        p2, p2_val = price_highs[-1]
        r1, r1_val = rsi_highs[-2]
        r2, r2_val = rsi_highs[-1]

        # Divergencia bearish: preco subiu mais, RSI caiu
        if p2_val > p1_val and r2_val < r1_val and abs(p2 - p1) >= 5:
            strength = min(int(abs(r1_val - r2_val) * 2), 100)
            return {
                "type": "bearish",
                "strength": strength,
                "description": f"Divergencia Bearish: preco fez novo topo mas RSI caiu. Sinal de possivel QUEDA.",
            }

    return result


def detect_macd_divergence(df: pd.DataFrame, lookback: int = 60) -> dict:
    """Detecta divergencia entre preco e MACD histograma."""
    result = {"type": "none", "strength": 0, "description": "Sem divergencia MACD detectada"}

    if "macd_hist" not in df.columns or len(df) < lookback:
        return result

    recent_close = df["Close"].iloc[-lookback:]
    recent_macd = df["macd_hist"].iloc[-lookback:]

    # Divergencia bullish
    price_lows = _find_swing_lows(recent_close, order=5)
    macd_lows = _find_swing_lows(recent_macd, order=3)

    if len(price_lows) >= 2 and len(macd_lows) >= 2:
        p1, p1_val = price_lows[-2]
        p2, p2_val = price_lows[-1]
        m1, m1_val = macd_lows[-2]
        m2, m2_val = macd_lows[-1]

        if p2_val < p1_val and m2_val > m1_val and abs(p2 - p1) >= 5:
            strength = min(int(abs(m2_val - m1_val) * 100), 100)
            return {
                "type": "bullish",
                "strength": strength,
                "description": "Divergencia Bullish no MACD: momentum melhorando enquanto preco cai. Reversao provavel.",
            }

    # Divergencia bearish
    price_highs = _find_swing_highs(recent_close, order=5)
    macd_highs = _find_swing_highs(recent_macd, order=3)

    if len(price_highs) >= 2 and len(macd_highs) >= 2:
        p1, p1_val = price_highs[-2]
        p2, p2_val = price_highs[-1]
        m1, m1_val = macd_highs[-2]
        m2, m2_val = macd_highs[-1]

        if p2_val > p1_val and m2_val < m1_val and abs(p2 - p1) >= 5:
            strength = min(int(abs(m1_val - m2_val) * 100), 100)
            return {
                "type": "bearish",
                "strength": strength,
                "description": "Divergencia Bearish no MACD: momentum enfraquecendo enquanto preco sobe. Cuidado.",
            }

    return result


# =============================================================
# TENDENCIA E CLASSIFICACAO
# =============================================================

def _calc_slope(series: pd.Series, window: int = 10) -> float:
    """Calcula a inclinacao de uma serie nos ultimos N periodos."""
    if series is None or len(series.dropna()) < window:
        return 0.0
    recent = series.dropna().iloc[-window:]
    x = np.arange(len(recent))
    try:
        slope = np.polyfit(x, recent.values, 1)[0]
        return slope / recent.mean() if recent.mean() != 0 else 0.0
    except Exception:
        return 0.0


def detect_trend(df: pd.DataFrame) -> tuple[str, int]:
    """
    Detecta tendencia usando SMA + EMA + ADX.
    Retorna: (tendencia, forca_0_a_100)
    """
    if df.empty or len(df) < 50:
        return "indefinido", 0

    close = df["Close"]
    sma_20 = df.get("sma_20")
    sma_50 = df.get("sma_50")
    ema_9 = df.get("ema_9")
    ema_21 = df.get("ema_21")
    adx = df.get("adx")
    adx_pos = df.get("adx_pos")
    adx_neg = df.get("adx_neg")

    if sma_20 is None or sma_50 is None:
        return "indefinido", 0

    current_price = close.iloc[-1]
    sma20_val = sma_20.iloc[-1]
    sma50_val = sma_50.iloc[-1]

    # ADX para forca
    adx_val = adx.iloc[-1] if adx is not None and not pd.isna(adx.iloc[-1]) else 20
    di_plus = adx_pos.iloc[-1] if adx_pos is not None and not pd.isna(adx_pos.iloc[-1]) else 0
    di_minus = adx_neg.iloc[-1] if adx_neg is not None and not pd.isna(adx_neg.iloc[-1]) else 0

    # EMA crossover
    ema_bullish = False
    ema_bearish = False
    if ema_9 is not None and ema_21 is not None:
        e9 = ema_9.iloc[-1]
        e21 = ema_21.iloc[-1]
        if not pd.isna(e9) and not pd.isna(e21):
            ema_bullish = e9 > e21
            ema_bearish = e9 < e21

    sma20_slope = _calc_slope(sma_20, 10)
    sma50_slope = _calc_slope(sma_50, 10)

    # Classificacao combinada
    if adx_val < ADX_WEAK:
        trend = "lateral"
        strength = max(10, int(adx_val * 1.5))
    elif di_plus > di_minus and current_price > sma20_val:
        if sma20_slope > 0 and sma50_slope > 0:
            trend = "alta"
        elif ema_bullish:
            trend = "reversao_alta"
        else:
            trend = "alta_fraca"
        strength = min(int(adx_val * 1.5), 100)
    elif di_minus > di_plus and current_price < sma20_val:
        if sma20_slope < 0 and sma50_slope < 0:
            trend = "baixa"
        elif ema_bearish:
            trend = "reversao_baixa"
        else:
            trend = "baixa_fraca"
        strength = min(int(adx_val * 1.5), 100)
    elif ema_bullish and sma20_slope > 0:
        trend = "reversao_alta"
        strength = min(int(adx_val), 70)
    elif ema_bearish and sma20_slope < 0:
        trend = "reversao_baixa"
        strength = min(int(adx_val), 70)
    else:
        trend = "lateral"
        strength = 30

    return trend, strength


def classify_dip(df: pd.DataFrame) -> dict:
    """
    Classifica se uma queda e ruido, correcao ou tendencia de baixa.
    Usa ATR para calibrar o que e "normal" vs "anormal".
    """
    if df.empty or len(df) < 30:
        return {"type": "indefinido", "drawdown": 0, "explanation": "Dados insuficientes"}

    close = df["Close"]
    recent_high = close.iloc[-30:].max()
    current = close.iloc[-1]
    drawdown = ((recent_high - current) / recent_high) * 100

    # Usar ATR para contexto de volatilidade
    atr = df.get("atr")
    atr_val = atr.iloc[-1] if atr is not None and not pd.isna(atr.iloc[-1]) else 0
    atr_pct = (atr_val / current * 100) if current > 0 else 2

    sma_50 = df.get("sma_50")
    sma50_slope = _calc_slope(sma_50, 10) if sma_50 is not None else 0

    adx = df.get("adx")
    adx_val = adx.iloc[-1] if adx is not None and not pd.isna(adx.iloc[-1]) else 20

    # Drawdown relativo a volatilidade normal
    if drawdown < atr_pct * 1.5:
        return {
            "type": "estavel",
            "drawdown": drawdown,
            "explanation": f"Preco estavel. Queda de {drawdown:.1f}% esta dentro da volatilidade normal (ATR: {atr_pct:.1f}%).",
        }
    elif drawdown < atr_pct * 4:
        if sma50_slope > 0:
            return {
                "type": "ruido",
                "drawdown": drawdown,
                "explanation": f"Queda de {drawdown:.1f}% - flutuacao normal. Tendencia de medio prazo ainda positiva (SMA50 subindo).",
            }
        else:
            return {
                "type": "alerta",
                "drawdown": drawdown,
                "explanation": f"Queda de {drawdown:.1f}% com media de 50 dias enfraquecendo. ADX em {adx_val:.0f}. Fique atento.",
            }
    elif drawdown < 20:
        if sma50_slope > 0:
            return {
                "type": "correcao",
                "drawdown": drawdown,
                "explanation": f"Correcao de {drawdown:.1f}% em tendencia de alta. Pode ser oportunidade se confirmada por indicadores.",
            }
        else:
            return {
                "type": "queda_moderada",
                "drawdown": drawdown,
                "explanation": f"Queda de {drawdown:.1f}% com medias caindo e ADX em {adx_val:.0f}. Espere sinais de reversao.",
            }
    else:
        return {
            "type": "queda_forte",
            "drawdown": drawdown,
            "explanation": f"Queda de {drawdown:.1f}% da maxima. So considere compra em suportes Fibonacci com divergencia bullish confirmada.",
        }


# =============================================================
# SISTEMA DE SCORING v2
# =============================================================

def score_asset(df: pd.DataFrame, fear_greed_value: int = 50) -> dict:
    """
    Calcula score de confianca de compra (0-100) com 10 indicadores.
    """
    if df.empty or len(df) < 50:
        return _empty_score()

    result = {
        "score": 0,
        "label": "Sem Dados",
        "signals": {},
        "trend": "indefinido",
        "trend_strength": 0,
        "confluence": {"agree_buy": 0, "total": 0, "percentage": 0},
        "divergences": {},
    }

    total_points = 0
    signal_scores = []  # Para calcular confluencia

    # 1. RSI (0-12 pontos)
    pts, info = _score_rsi(df)
    total_points += pts
    result["signals"]["RSI"] = info
    signal_scores.append((pts, info["max"]))

    # 2. Stochastic RSI (0-8 pontos)
    pts, info = _score_stoch_rsi(df)
    total_points += pts
    result["signals"]["Stoch RSI"] = info
    signal_scores.append((pts, info["max"]))

    # 3. MACD (0-10 pontos)
    pts, info = _score_macd(df)
    total_points += pts
    result["signals"]["MACD"] = info
    signal_scores.append((pts, info["max"]))

    # 4. Tendencia SMA + EMA (0-12 pontos)
    pts, info = _score_trend(df)
    total_points += pts
    result["signals"]["Tendencia"] = info
    signal_scores.append((pts, info["max"]))

    # 5. ADX - Forca da Tendencia (0-10 pontos)
    pts, info = _score_adx(df)
    total_points += pts
    result["signals"]["ADX"] = info
    signal_scores.append((pts, info["max"]))

    # 6. Bollinger Bands (0-8 pontos)
    pts, info = _score_bollinger(df)
    total_points += pts
    result["signals"]["Bollinger"] = info
    signal_scores.append((pts, info["max"]))

    # 7. Volume + OBV (0-10 pontos)
    pts, info = _score_volume_obv(df)
    total_points += pts
    result["signals"]["Volume/OBV"] = info
    signal_scores.append((pts, info["max"]))

    # 8. Fibonacci (0-10 pontos)
    pts, info = _score_fibonacci(df)
    total_points += pts
    result["signals"]["Fibonacci"] = info
    signal_scores.append((pts, info["max"]))

    # 9. Fear & Greed (0-8 pontos)
    pts, info = _score_fear_greed(fear_greed_value)
    total_points += pts
    result["signals"]["Medo/Ganancia"] = info
    signal_scores.append((pts, info["max"]))

    # 10. Divergencias RSI + MACD (0-12 pontos)
    pts, info = _score_divergences(df)
    total_points += pts
    result["signals"]["Divergencias"] = info
    result["divergences"] = info.get("details", {})
    signal_scores.append((pts, info["max"]))

    result["score"] = int(total_points)

    # Classificar
    if total_points >= BUY_CONFIDENCE_STRONG:
        result["label"] = "COMPRA FORTE"
    elif total_points >= BUY_CONFIDENCE_MODERATE:
        result["label"] = "COMPRA"
    elif total_points >= SELL_CONFIDENCE:
        result["label"] = "NEUTRO"
    elif total_points >= 15:
        result["label"] = "VENDA"
    else:
        result["label"] = "VENDA FORTE"

    # Tendencia
    trend, strength = detect_trend(df)
    result["trend"] = trend
    result["trend_strength"] = strength

    # Confluencia: quantos indicadores concordam com compra
    agree = sum(1 for pts, max_pts in signal_scores if max_pts > 0 and pts / max_pts >= 0.6)
    total_indicators = len(signal_scores)
    result["confluence"] = {
        "agree_buy": agree,
        "total": total_indicators,
        "percentage": int(agree / total_indicators * 100) if total_indicators > 0 else 0,
    }

    return result


def _empty_score():
    return {
        "score": 0,
        "label": "Sem Dados",
        "signals": {},
        "trend": "indefinido",
        "trend_strength": 0,
        "confluence": {"agree_buy": 0, "total": 0, "percentage": 0},
        "divergences": {},
    }


# ---- Funcoes de scoring individuais ----

def _score_rsi(df: pd.DataFrame) -> tuple[float, dict]:
    MAX = SCORE_MAX_RSI
    rsi = df.get("rsi_14")
    if rsi is None or pd.isna(rsi.iloc[-1]):
        return 0, {"value": 0, "signal": "sem dados", "points": 0, "max": MAX}

    val = rsi.iloc[-1]
    if val <= 20:
        pts = MAX
        signal = "Extremamente sobrevendido - forte sinal de compra"
    elif val <= RSI_OVERSOLD:
        pts = int(MAX * 0.75)
        signal = "Sobrevendido - possivel oportunidade de compra"
    elif val <= 40:
        pts = int(MAX * 0.5)
        signal = "Abaixo da media - moderadamente favoravel"
    elif val <= 60:
        pts = int(MAX * 0.25)
        signal = "Neutro - sem sinal claro"
    elif val <= RSI_OVERBOUGHT:
        pts = 1
        signal = "Acima da media - cuidado"
    else:
        pts = 0
        signal = "Sobrecomprado - risco de queda"

    return pts, {"value": round(val, 1), "signal": signal, "points": pts, "max": MAX}


def _score_stoch_rsi(df: pd.DataFrame) -> tuple[float, dict]:
    MAX = SCORE_MAX_STOCH_RSI
    k = df.get("stoch_rsi_k")
    d = df.get("stoch_rsi_d")

    if k is None or pd.isna(k.iloc[-1]):
        return 0, {"value": 0, "signal": "sem dados", "points": 0, "max": MAX}

    k_val = k.iloc[-1]
    d_val = d.iloc[-1] if d is not None and not pd.isna(d.iloc[-1]) else k_val

    # Cruzamento bullish do Stoch RSI
    k_prev = k.iloc[-2] if len(k) > 1 and not pd.isna(k.iloc[-2]) else k_val
    d_prev = d.iloc[-2] if d is not None and len(d) > 1 and not pd.isna(d.iloc[-2]) else d_val
    bullish_cross = k_prev <= d_prev and k_val > d_val

    if k_val <= STOCH_RSI_OVERSOLD and bullish_cross:
        pts = MAX
        signal = "Stoch RSI sobrevendido COM cruzamento bullish - sinal de compra muito forte"
    elif k_val <= STOCH_RSI_OVERSOLD:
        pts = int(MAX * 0.75)
        signal = "Stoch RSI sobrevendido - aguarde cruzamento para confirmar"
    elif k_val <= 40 and k_val > d_val:
        pts = int(MAX * 0.5)
        signal = "Stoch RSI baixo com momentum subindo"
    elif k_val >= STOCH_RSI_OVERBOUGHT:
        pts = 0
        signal = "Stoch RSI sobrecomprado - nao e hora de comprar"
    else:
        pts = int(MAX * 0.25)
        signal = "Stoch RSI neutro"

    return pts, {"value": round(k_val, 1), "signal": signal, "points": pts, "max": MAX}


def _score_macd(df: pd.DataFrame) -> tuple[float, dict]:
    MAX = SCORE_MAX_MACD
    macd = df.get("macd")
    macd_sig = df.get("macd_signal")
    macd_hist = df.get("macd_hist")

    if macd is None or macd_sig is None or pd.isna(macd.iloc[-1]) or pd.isna(macd_sig.iloc[-1]):
        return 0, {"value": 0, "signal": "sem dados", "points": 0, "max": MAX}

    m = macd.iloc[-1]
    s = macd_sig.iloc[-1]
    h = macd_hist.iloc[-1] if macd_hist is not None and not pd.isna(macd_hist.iloc[-1]) else 0
    h_prev = macd_hist.iloc[-2] if macd_hist is not None and len(macd_hist) > 1 and not pd.isna(macd_hist.iloc[-2]) else 0

    if m > s and h > h_prev and h > 0:
        pts = MAX
        signal = "Cruzamento bullish com momentum crescente - forte sinal de compra"
    elif m > s and h > h_prev:
        pts = int(MAX * 0.8)
        signal = "MACD acima do sinal com histograma melhorando"
    elif m > s:
        pts = int(MAX * 0.6)
        signal = "MACD acima do sinal - tendencia positiva"
    elif m < s and h > h_prev:
        pts = int(MAX * 0.4)
        signal = "Abaixo do sinal mas momentum melhorando - possivel reversao"
    elif m < s and h < h_prev:
        pts = 1
        signal = "Cruzamento bearish com momentum caindo - sinal negativo"
    else:
        pts = int(MAX * 0.3)
        signal = "MACD neutro"

    return pts, {"value": round(h, 4), "signal": signal, "points": pts, "max": MAX}


def _score_trend(df: pd.DataFrame) -> tuple[float, dict]:
    """Pontua tendencia combinando SMA + EMA."""
    MAX = SCORE_MAX_TREND
    close = df["Close"].iloc[-1]
    sma_20 = df.get("sma_20")
    sma_50 = df.get("sma_50")
    ema_9 = df.get("ema_9")
    ema_21 = df.get("ema_21")

    if sma_20 is None or sma_50 is None:
        return 0, {"value": 0, "signal": "sem dados", "points": 0, "max": MAX}

    s20 = sma_20.iloc[-1]
    s50 = sma_50.iloc[-1]

    if pd.isna(s20) or pd.isna(s50):
        return 0, {"value": 0, "signal": "sem dados", "points": 0, "max": MAX}

    # EMA crossover
    ema_bull = False
    ema_bear = False
    ema_info = ""
    if ema_9 is not None and ema_21 is not None:
        e9 = ema_9.iloc[-1]
        e21 = ema_21.iloc[-1]
        if not pd.isna(e9) and not pd.isna(e21):
            ema_bull = e9 > e21
            ema_bear = e9 < e21

            # Detectar cruzamento recente
            e9_prev = ema_9.iloc[-3] if len(ema_9) > 2 else e9
            e21_prev = ema_21.iloc[-3] if len(ema_21) > 2 else e21
            if not pd.isna(e9_prev) and not pd.isna(e21_prev):
                if e9_prev <= e21_prev and e9 > e21:
                    ema_info = " EMA9 cruzou acima da EMA21 recentemente!"

    sma20_slope = _calc_slope(sma_20, 10)

    # Scoring combinado
    if close < s50 and sma20_slope > 0 and ema_bull:
        pts = MAX
        signal = f"Preco abaixo da SMA50 mas com reversao confirmada por EMA.{ema_info}"
    elif close < s50 and sma20_slope > 0:
        pts = int(MAX * 0.83)
        signal = "Preco abaixo da media mas com sinais de reversao"
    elif close > s20 > s50 and ema_bull:
        pts = int(MAX * 0.67)
        signal = f"Tendencia de alta confirmada por SMAs e EMAs.{ema_info}"
    elif close > s20 and ema_bull:
        pts = int(MAX * 0.58)
        signal = "Preco acima da SMA20 com EMA bullish"
    elif close < s20 and close < s50 and ema_bear:
        pts = int(MAX * 0.17)
        signal = "Preco abaixo das medias com EMA bearish - tendencia negativa"
    elif close < s20 and close < s50 and sma20_slope < -0.001:
        pts = int(MAX * 0.17)
        signal = "Preco abaixo das medias e caindo - espere estabilizacao"
    else:
        pts = int(MAX * 0.42)
        signal = "Posicao mista em relacao as medias"

    return pts, {"value": round(close, 2), "signal": signal, "points": pts, "max": MAX}


def _score_adx(df: pd.DataFrame) -> tuple[float, dict]:
    """Pontua forca da tendencia usando ADX + DI+/DI-."""
    MAX = SCORE_MAX_ADX
    adx = df.get("adx")
    di_plus = df.get("adx_pos")
    di_minus = df.get("adx_neg")

    if adx is None or pd.isna(adx.iloc[-1]):
        return int(MAX * 0.3), {"value": 0, "signal": "sem dados ADX", "points": int(MAX * 0.3), "max": MAX}

    adx_val = adx.iloc[-1]
    dip = di_plus.iloc[-1] if di_plus is not None and not pd.isna(di_plus.iloc[-1]) else 0
    dim = di_minus.iloc[-1] if di_minus is not None and not pd.isna(di_minus.iloc[-1]) else 0

    if adx_val >= ADX_VERY_STRONG and dip > dim:
        pts = MAX
        signal = f"Tendencia de ALTA muito forte (ADX: {adx_val:.0f}, DI+: {dip:.0f} > DI-: {dim:.0f})"
    elif adx_val >= ADX_STRONG and dip > dim:
        pts = int(MAX * 0.8)
        signal = f"Tendencia de alta confirmada (ADX: {adx_val:.0f})"
    elif adx_val >= ADX_STRONG and dim > dip:
        pts = int(MAX * 0.2)
        signal = f"Tendencia de BAIXA confirmada (ADX: {adx_val:.0f}, DI-: {dim:.0f} > DI+: {dip:.0f}) - desfavoravel"
    elif adx_val >= ADX_VERY_STRONG and dim > dip:
        pts = 0
        signal = f"Tendencia de baixa muito forte - NAO comprar"
    elif adx_val < ADX_WEAK:
        pts = int(MAX * 0.3)
        signal = f"Mercado lateral/sem tendencia (ADX: {adx_val:.0f}) - sinais menos confiaveis"
    else:
        pts = int(MAX * 0.5)
        signal = f"ADX em {adx_val:.0f} - tendencia moderada"

    return pts, {"value": round(adx_val, 1), "signal": signal, "points": pts, "max": MAX}


def _score_bollinger(df: pd.DataFrame) -> tuple[float, dict]:
    MAX = SCORE_MAX_BOLLINGER
    close = df["Close"].iloc[-1]
    bb_lower = df.get("bb_lower")
    bb_upper = df.get("bb_upper")
    bb_middle = df.get("bb_middle")

    if bb_lower is None or bb_upper is None or pd.isna(bb_lower.iloc[-1]) or pd.isna(bb_upper.iloc[-1]):
        return 0, {"value": 0, "signal": "sem dados", "points": 0, "max": MAX}

    lower = bb_lower.iloc[-1]
    upper = bb_upper.iloc[-1]
    middle = bb_middle.iloc[-1]

    band_width = (upper - lower) / middle if middle != 0 else 0
    position = (close - lower) / (upper - lower) if (upper - lower) != 0 else 0.5

    # Verificar se bandas estao expandindo
    prev_width = 0
    if len(df) > 5:
        prev_upper = bb_upper.iloc[-5]
        prev_lower = bb_lower.iloc[-5]
        prev_mid = bb_middle.iloc[-5]
        if not pd.isna(prev_upper) and prev_mid != 0:
            prev_width = (prev_upper - prev_lower) / prev_mid
    expanding = band_width > prev_width

    rsi = df.get("rsi_14")
    rsi_val = rsi.iloc[-1] if rsi is not None and not pd.isna(rsi.iloc[-1]) else 50

    if position <= 0.1 and rsi_val < 40 and not expanding:
        pts = MAX
        signal = "Preco na banda inferior + RSI baixo + volatilidade estavel - forte oportunidade"
    elif position <= 0.2 and not expanding:
        pts = int(MAX * 0.75)
        signal = "Proximo da banda inferior - possivel bounce"
    elif position <= 0.2 and expanding:
        pts = int(MAX * 0.25)
        signal = "Perto da banda inferior mas volatilidade crescendo - pode cair mais"
    elif position <= 0.5:
        pts = int(MAX * 0.5)
        signal = "Preco entre banda inferior e media"
    elif position >= 0.9:
        pts = 0
        signal = "Preco na banda superior - sobrecomprado"
    else:
        pts = int(MAX * 0.25)
        signal = "Preco acima da media Bollinger"

    return pts, {"value": round(position * 100, 1), "signal": signal, "points": pts, "max": MAX}


def _score_volume_obv(df: pd.DataFrame) -> tuple[float, dict]:
    """Pontua Volume + OBV combinados."""
    MAX = SCORE_MAX_VOLUME
    volume = df.get("Volume")
    vol_sma = df.get("volume_sma_20")
    obv = df.get("obv")
    obv_sma = df.get("obv_sma_20")

    if volume is None or vol_sma is None:
        return int(MAX * 0.5), {"value": 0, "signal": "sem dados de volume", "points": int(MAX * 0.5), "max": MAX}

    current_vol = volume.iloc[-1]
    avg_vol = vol_sma.iloc[-1]

    if pd.isna(current_vol) or pd.isna(avg_vol) or avg_vol == 0:
        return int(MAX * 0.5), {"value": 0, "signal": "sem dados de volume", "points": int(MAX * 0.5), "max": MAX}

    vol_ratio = current_vol / avg_vol
    price_change = 0
    if len(df) > 1:
        prev_close = df["Close"].iloc[-2]
        if prev_close != 0:
            price_change = (df["Close"].iloc[-1] - prev_close) / prev_close

    # Analise OBV
    obv_trend = "neutro"
    if obv is not None and obv_sma is not None:
        obv_val = obv.iloc[-1]
        obv_avg = obv_sma.iloc[-1]
        if not pd.isna(obv_val) and not pd.isna(obv_avg):
            if obv_val > obv_avg:
                obv_trend = "acumulacao"
            else:
                obv_trend = "distribuicao"

    # Scoring combinado
    if price_change < 0 and vol_ratio < 0.8 and obv_trend == "acumulacao":
        pts = MAX
        signal = "Queda com volume baixo + OBV subindo = smart money acumulando. Forte sinal de compra."
    elif price_change < 0 and vol_ratio < 0.8:
        pts = int(MAX * 0.7)
        signal = "Queda com volume baixo - vendedores perdendo forca"
    elif price_change < 0 and vol_ratio > 1.5 and obv_trend == "distribuicao":
        pts = int(MAX * 0.1)
        signal = "Queda com volume ALTO + OBV caindo = distribuicao. Pressao vendedora forte."
    elif price_change < 0 and vol_ratio > 1.5:
        pts = int(MAX * 0.2)
        signal = "Queda com volume alto - pressao vendedora"
    elif price_change > 0 and vol_ratio > 1.2 and obv_trend == "acumulacao":
        pts = int(MAX * 0.8)
        signal = "Alta com volume forte + OBV confirmando - momentum comprador real"
    elif price_change > 0 and vol_ratio > 1.2:
        pts = int(MAX * 0.6)
        signal = "Alta com volume forte - interesse comprador"
    elif price_change > 0 and obv_trend == "distribuicao":
        pts = int(MAX * 0.3)
        signal = "Alta mas OBV caindo - smart money pode estar vendendo. Cuidado."
    else:
        pts = int(MAX * 0.5)
        signal = f"Volume normal ({vol_ratio:.1f}x media)"

    return pts, {"value": round(vol_ratio, 2), "signal": signal, "points": pts, "max": MAX}


def _score_fibonacci(df: pd.DataFrame) -> tuple[float, dict]:
    """Pontua proximidade a niveis chave de Fibonacci."""
    MAX = SCORE_MAX_FIBONACCI
    fib = df.attrs.get("fibonacci", {})
    levels = fib.get("levels", {})

    if not levels:
        return int(MAX * 0.5), {"value": 0, "signal": "sem dados Fibonacci", "points": int(MAX * 0.5), "max": MAX}

    close = df["Close"].iloc[-1]
    swing_high = fib.get("swing_high", close)
    swing_low = fib.get("swing_low", close)

    if swing_high == swing_low:
        return int(MAX * 0.5), {"value": 0, "signal": "range muito estreito", "points": int(MAX * 0.5), "max": MAX}

    # Encontrar nivel Fibonacci mais proximo
    closest_level = None
    closest_dist = float("inf")
    closest_label = ""

    for label, price in levels.items():
        dist = abs(close - price) / close * 100
        if dist < closest_dist:
            closest_dist = dist
            closest_level = price
            closest_label = label

    # Verificar se estamos em niveis chave de SUPORTE (61.8% e 50% sao os mais fortes)
    fib_618 = levels.get("61.8%", 0)
    fib_50 = levels.get("50%", 0)
    fib_382 = levels.get("38.2%", 0)

    dist_618 = abs(close - fib_618) / close * 100 if fib_618 > 0 else 999
    dist_50 = abs(close - fib_50) / close * 100 if fib_50 > 0 else 999
    dist_382 = abs(close - fib_382) / close * 100 if fib_382 > 0 else 999

    if dist_618 <= 3 and close >= fib_618:
        pts = MAX
        signal = f"Preco no nivel 61.8% Fibonacci (${fib_618:.2f}) - nivel de suporte MAIS forte. Zona de compra ideal."
    elif dist_50 <= 3 and close >= fib_50:
        pts = int(MAX * 0.8)
        signal = f"Preco no nivel 50% Fibonacci (${fib_50:.2f}) - suporte forte"
    elif dist_382 <= 3 and close >= fib_382:
        pts = int(MAX * 0.6)
        signal = f"Preco no nivel 38.2% Fibonacci (${fib_382:.2f}) - suporte moderado"
    elif close < fib_618:
        pts = int(MAX * 0.3)
        signal = f"Preco abaixo do suporte 61.8% - busque proximo nivel ou aguarde"
    elif close > fib_382:
        pts = int(MAX * 0.4)
        signal = f"Preco acima dos niveis de retracemento. Mais perto de {closest_label} (${closest_level:.2f})"
    else:
        pts = int(MAX * 0.5)
        signal = f"Preco entre niveis Fibonacci. Mais proximo de {closest_label}"

    return pts, {"value": round(close, 2), "signal": signal, "points": pts, "max": MAX,
                 "levels": levels}


def _score_fear_greed(value: int) -> tuple[float, dict]:
    """Pontua Fear & Greed Index (logica contrarian)."""
    MAX = SCORE_MAX_FEAR_GREED
    if value <= 15:
        pts = MAX
        signal = "Medo Extremo - historicamente melhor momento para comprar"
        label = "Medo Extremo"
    elif value <= 30:
        pts = int(MAX * 0.75)
        signal = "Medo - mercado pessimista, oportunidades surgem"
        label = "Medo"
    elif value <= 45:
        pts = int(MAX * 0.5)
        signal = "Medo leve - mercado cauteloso"
        label = "Medo Leve"
    elif value <= 55:
        pts = int(MAX * 0.25)
        signal = "Neutro - sem sentimento dominante"
        label = "Neutro"
    elif value <= 70:
        pts = 1
        signal = "Ganancia - cuidado com compras neste momento"
        label = "Ganancia"
    else:
        pts = 0
        signal = "Ganancia Extrema - alto risco de correcao iminente"
        label = "Ganancia Extrema"

    return pts, {"value": value, "signal": signal, "points": pts, "max": MAX, "label": label}


def _score_divergences(df: pd.DataFrame) -> tuple[float, dict]:
    """Pontua divergencias RSI + MACD (sinal de maior conviccao)."""
    MAX = SCORE_MAX_DIVERGENCE

    rsi_div = detect_rsi_divergence(df)
    macd_div = detect_macd_divergence(df)

    pts = 0
    signals = []
    half = MAX // 2

    # RSI divergence (ate metade dos pontos)
    if rsi_div["type"] == "bullish":
        pts += half
        signals.append(f"RSI: {rsi_div['description']}")
    elif rsi_div["type"] == "bearish":
        pts += 0
        signals.append(f"RSI: {rsi_div['description']}")

    # MACD divergence (ate metade dos pontos)
    if macd_div["type"] == "bullish":
        pts += half
        signals.append(f"MACD: {macd_div['description']}")
    elif macd_div["type"] == "bearish":
        pts += 0
        signals.append(f"MACD: {macd_div['description']}")

    if not signals:
        signal_text = "Nenhuma divergencia detectada - sem sinal extra de reversao"
        pts = int(MAX * 0.3)  # Pontuacao neutra
    else:
        signal_text = " | ".join(signals)

    return pts, {
        "value": pts,
        "signal": signal_text,
        "points": pts,
        "max": MAX,
        "details": {"rsi": rsi_div, "macd": macd_div},
    }


# =============================================================
# GESTAO DE RISCO
# =============================================================

def compute_risk_metrics(df: pd.DataFrame, portfolio_value: float = DEFAULT_PORTFOLIO_VALUE,
                         risk_pct: float = DEFAULT_RISK_PER_TRADE) -> dict:
    """
    Calcula metricas de gestao de risco baseadas em ATR.
    """
    if df.empty or "atr" not in df.columns:
        return {}

    close = df["Close"].iloc[-1]
    atr = df["atr"].iloc[-1]
    fib = df.attrs.get("fibonacci", {})
    levels = fib.get("levels", {})

    if pd.isna(atr) or atr == 0 or close == 0:
        return {}

    stop_loss = close - (atr * ATR_STOP_MULTIPLIER)
    risk_per_unit = close - stop_loss

    # Position sizing: quanto comprar para arriscar apenas X% do portfolio
    risk_amount = portfolio_value * (risk_pct / 100)
    position_size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
    position_value = position_size * close

    # Take profit baseado em Fibonacci
    tp1 = levels.get("38.2%", close * 1.05)
    tp2 = levels.get("23.6%", close * 1.10)
    tp3 = levels.get("0%", close * 1.15)

    rr_ratio = (tp1 - close) / risk_per_unit if risk_per_unit > 0 else 0

    return {
        "preco_atual": round(close, 2),
        "atr": round(atr, 4),
        "atr_percentual": round(atr / close * 100, 2),
        "stop_loss": round(stop_loss, 2),
        "risco_por_unidade": round(risk_per_unit, 4),
        "tamanho_posicao": round(position_size, 6),
        "valor_posicao": round(position_value, 2),
        "risco_maximo": round(risk_amount, 2),
        "take_profit_1": round(tp1, 2),
        "take_profit_2": round(tp2, 2),
        "take_profit_3": round(tp3, 2),
        "risco_retorno": round(rr_ratio, 2),
    }


def compute_dca_plan(df: pd.DataFrame, total_amount: float, num_tranches: int = 4) -> list[dict]:
    """
    Gera plano de DCA (Dollar Cost Averaging) usando niveis Fibonacci.
    Divide o investimento em parcelas em niveis de suporte.
    """
    if df.empty:
        return []

    close = df["Close"].iloc[-1]
    fib = df.attrs.get("fibonacci", {})
    levels = fib.get("levels", {})

    if not levels:
        # Fallback: dividir igualmente com quedas de 3%
        plan = []
        for i in range(num_tranches):
            price = close * (1 - i * 0.03)
            plan.append({
                "tranche": i + 1,
                "preco": round(price, 2),
                "valor": round(total_amount / num_tranches, 2),
                "percentual": round(100 / num_tranches, 1),
                "nivel": f"-{i*3}% do preco atual",
            })
        return plan

    # Usar Fibonacci para definir niveis de compra
    fib_buy_levels = [
        ("Preco Atual", close),
        ("Fibonacci 38.2%", levels.get("38.2%", close * 0.97)),
        ("Fibonacci 50%", levels.get("50%", close * 0.95)),
        ("Fibonacci 61.8%", levels.get("61.8%", close * 0.92)),
    ]

    # Distribuir mais dinheiro nos niveis mais baixos (mais baratos)
    weights = [0.15, 0.25, 0.30, 0.30]  # 15% agora, 25% em 38.2%, 30% em 50%, 30% em 61.8%

    plan = []
    for i, ((nivel, preco), peso) in enumerate(zip(fib_buy_levels[:num_tranches], weights[:num_tranches])):
        plan.append({
            "tranche": i + 1,
            "preco": round(preco, 2),
            "valor": round(total_amount * peso, 2),
            "percentual": round(peso * 100, 1),
            "nivel": nivel,
        })

    return plan


def compute_correlation_matrix(price_data: dict) -> pd.DataFrame:
    """
    Calcula matriz de correlacao entre multiplos ativos.
    price_data: dict de {nome: pd.Series de precos de fechamento}
    """
    if len(price_data) < 2:
        return pd.DataFrame()

    # Alinhar datas e calcular retornos
    combined = pd.DataFrame()
    for name, prices in price_data.items():
        if isinstance(prices, pd.Series) and len(prices) > 20:
            returns = prices.pct_change().dropna()
            combined[name] = returns

    if combined.empty or len(combined.columns) < 2:
        return pd.DataFrame()

    return combined.corr()


# =============================================================
# RECOMENDACOES
# =============================================================

def generate_recommendation(score_result: dict, dip_info: dict, asset_name: str) -> str:
    """Gera recomendacao detalhada em texto para o usuario."""
    score = score_result["score"]
    label = score_result["label"]
    trend = score_result["trend"]
    dip_type = dip_info["type"]
    confluence = score_result.get("confluence", {})
    agree = confluence.get("agree_buy", 0)
    total = confluence.get("total", 10)
    divergences = score_result.get("divergences", {})

    rsi_div = divergences.get("rsi", {}).get("type", "none")
    macd_div = divergences.get("macd", {}).get("type", "none")
    has_bullish_div = rsi_div == "bullish" or macd_div == "bullish"
    has_bearish_div = rsi_div == "bearish" or macd_div == "bearish"

    lines = [f"**{asset_name}** - Score: {score}/100 ({label})"]
    lines.append(f"Tendencia: {trend.replace('_', ' ').title()} | Confluencia: {agree}/{total} indicadores concordam")

    if has_bullish_div:
        lines.append("**DIVERGENCIA BULLISH DETECTADA** - Sinal forte de possivel reversao para alta!")

    if has_bearish_div:
        lines.append("**DIVERGENCIA BEARISH DETECTADA** - Atencao: possivel reversao para baixa!")

    lines.append(f"Contexto: {dip_info['explanation']}")

    if score >= BUY_CONFIDENCE_STRONG:
        if has_bullish_div and agree >= 6:
            lines.append(
                "RECOMENDACAO: **OPORTUNIDADE FORTE.** Score alto com divergencia bullish e "
                f"{agree} indicadores alinhados. Considere entrada com plano DCA."
            )
        elif dip_type in ("correcao", "ruido"):
            lines.append(
                "RECOMENDACAO: Momento muito favoravel. "
                "Multiplos indicadores apontam oportunidade. Use stop-loss baseado em ATR."
            )
        else:
            lines.append(
                "RECOMENDACAO: Sinais positivos convergindo. "
                "Considere compra em parcelas (DCA) para reduzir risco."
            )
    elif score >= BUY_CONFIDENCE_MODERATE:
        lines.append(
            "RECOMENDACAO: Sinais moderadamente favoraveis. "
            "Aguarde confirmacao ou inicie com posicao pequena."
        )
    elif score >= SELL_CONFIDENCE:
        lines.append(
            "RECOMENDACAO: Neutro. Sem sinais claros. "
            "Mantenha posicoes existentes e aguarde melhor oportunidade."
        )
    else:
        if trend in ("baixa", "baixa_fraca"):
            lines.append(
                "RECOMENDACAO: **NAO COMPRE AGORA.** Tendencia de baixa ativa"
                + (" com divergencia bearish." if has_bearish_div else ".")
                + " Espere reversao confirmada por EMA e divergencia bullish."
            )
        else:
            lines.append(
                "RECOMENDACAO: Sinais negativos. Evite novas compras. "
                "Aguarde melhora nos indicadores."
            )

    return "\n\n".join(lines)
