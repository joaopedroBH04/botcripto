# ============================================================
# BotCripto v2 - Motor de Reconhecimento de Padroes Graficos
# Detecta: H&S, Fundo/Topo Duplo, Triangulos, Bull/Bear Flag
# ============================================================

import pandas as pd
import numpy as np


def _find_pivots(series: pd.Series, order: int = 5) -> tuple[list, list]:
    """
    Identifica maximos e minimos locais (swing highs/lows).
    Retorna (highs, lows) como listas de (indice, valor).
    """
    values = series.dropna().values
    n = len(values)
    highs, lows = [], []

    for i in range(order, n - order):
        v = float(values[i])
        left  = values[i - order: i]
        right = values[i + 1: i + order + 1]

        if v >= float(left.max()) and v >= float(right.max()):
            highs.append((i, v))
        if v <= float(left.min()) and v <= float(right.min()):
            lows.append((i, v))

    return highs, lows


def detect_head_and_shoulders(df: pd.DataFrame) -> dict | None:
    """
    Detecta Ombro-Cabeca-Ombro (bearish) e H&S Invertido (bullish).
    Criterios: 3 pivots consecutivos, cabeca claramente maior/menor,
    ombros simetricos (diferenca < 6%) e espaco minimo entre eles.
    """
    if len(df) < 60:
        return None

    close = df["Close"]
    highs, lows = _find_pivots(close, order=5)

    # H&S Bearish: 3 maximos onde o do meio e o mais alto
    if len(highs) >= 3:
        i1, v1 = highs[-3]
        i2, v2 = highs[-2]
        i3, v3 = highs[-1]

        head_higher     = v2 > v1 * 1.02 and v2 > v3 * 1.02
        shoulders_close = abs(v1 - v3) / max(v1, v3) < 0.06
        spaced          = (i2 - i1) >= 8 and (i3 - i2) >= 8

        if head_higher and shoulders_close and spaced:
            neckline = float(close.iloc[i1:i3 + 1].nsmallest(5).mean())
            conf = min(90, 58 + int((v2 - max(v1, v3)) / v2 * 280))
            return {
                "name": "Ombro-Cabeca-Ombro",
                "type": "bearish",
                "confidence": conf,
                "description": (
                    f"H&S classico: ombros em ${v1:.2f} e ${v3:.2f}, cabeca em ${v2:.2f}. "
                    f"Sinal de reversao para BAIXA. Perda da neckline ${neckline:.2f} confirma queda."
                ),
                "key_levels": {
                    "Cabeca": round(v2, 2),
                    "Ombro Esq.": round(v1, 2),
                    "Ombro Dir.": round(v3, 2),
                    "Neckline": round(neckline, 2),
                },
            }

    # H&S Invertido Bullish: 3 minimos onde o do meio e o mais baixo
    if len(lows) >= 3:
        i1, v1 = lows[-3]
        i2, v2 = lows[-2]
        i3, v3 = lows[-1]

        head_lower      = v2 < v1 * 0.98 and v2 < v3 * 0.98
        shoulders_close = abs(v1 - v3) / max(v1, v3) < 0.06
        spaced          = (i2 - i1) >= 8 and (i3 - i2) >= 8

        if head_lower and shoulders_close and spaced:
            neckline = float(close.iloc[i1:i3 + 1].nlargest(5).mean())
            conf = min(90, 58 + int((min(v1, v3) - v2) / min(v1, v3) * 280))
            return {
                "name": "H&S Invertido",
                "type": "bullish",
                "confidence": conf,
                "description": (
                    f"H&S Invertido: ombros em ${v1:.2f} e ${v3:.2f}, fundo em ${v2:.2f}. "
                    f"Sinal de reversao para ALTA. Confirmacao acima de ${neckline:.2f}."
                ),
                "key_levels": {
                    "Fundo": round(v2, 2),
                    "Ombro Esq.": round(v1, 2),
                    "Ombro Dir.": round(v3, 2),
                    "Neckline": round(neckline, 2),
                },
            }

    return None


def detect_double_pattern(df: pd.DataFrame) -> dict | None:
    """
    Detecta Topo Duplo (bearish) e Fundo Duplo (bullish).
    Criterios: 2 pivots em precos similares (< 4%), espaco minimo de 12 barras.
    """
    if len(df) < 40:
        return None

    close = df["Close"]
    highs, lows = _find_pivots(close, order=5)

    if len(highs) >= 2:
        i1, v1 = highs[-2]
        i2, v2 = highs[-1]
        similar = abs(v1 - v2) / max(v1, v2) < 0.04
        spaced  = (i2 - i1) >= 12

        if similar and spaced:
            neckline = float(close.iloc[i1:i2 + 1].min())
            conf = min(88, 55 + int((1 - abs(v1 - v2) / max(v1, v2)) * 480))
            return {
                "name": "Topo Duplo",
                "type": "bearish",
                "confidence": conf,
                "description": (
                    f"Resistencia em ${max(v1, v2):.2f} testada duas vezes "
                    f"(diferenca de {abs(v1-v2)/v1*100:.1f}%). "
                    f"Perder o suporte ${neckline:.2f} confirma queda."
                ),
                "key_levels": {
                    "Topo 1": round(v1, 2),
                    "Topo 2": round(v2, 2),
                    "Suporte Critico": round(neckline, 2),
                },
            }

    if len(lows) >= 2:
        i1, v1 = lows[-2]
        i2, v2 = lows[-1]
        similar = abs(v1 - v2) / max(v1, v2) < 0.04
        spaced  = (i2 - i1) >= 12

        if similar and spaced:
            resistance = float(close.iloc[i1:i2 + 1].max())
            conf = min(88, 55 + int((1 - abs(v1 - v2) / max(v1, v2)) * 480))
            return {
                "name": "Fundo Duplo",
                "type": "bullish",
                "confidence": conf,
                "description": (
                    f"Suporte em ${min(v1, v2):.2f} segurou duas vezes "
                    f"(diferenca de {abs(v1-v2)/v1*100:.1f}%). "
                    f"Rompimento acima de ${resistance:.2f} confirma alta."
                ),
                "key_levels": {
                    "Fundo 1": round(v1, 2),
                    "Fundo 2": round(v2, 2),
                    "Resistencia": round(resistance, 2),
                },
            }

    return None


def detect_triangle(df: pd.DataFrame) -> dict | None:
    """
    Detecta Triangulo Ascendente (bullish), Descendente (bearish) e Simetrico (neutro).
    Usa a inclinacao normalizada dos pivots recentes.
    """
    if len(df) < 40:
        return None

    close = df["Close"]
    highs, lows = _find_pivots(close, order=5)

    if len(highs) < 2 or len(lows) < 2:
        return None

    rh = highs[-min(3, len(highs)):]
    rl = lows[-min(3, len(lows)):]

    h_dx = max(rh[-1][0] - rh[0][0], 1)
    l_dx = max(rl[-1][0] - rl[0][0], 1)
    h_slope = (rh[-1][1] - rh[0][1]) / h_dx
    l_slope = (rl[-1][1] - rl[0][1]) / l_dx

    avg_p = float(close.iloc[-1])
    hn = h_slope / avg_p * 100   # inclinacao dos maximos em %
    ln = l_slope / avg_p * 100   # inclinacao dos minimos em %

    if abs(hn) < 0.015 and ln > 0.015:
        return {
            "name": "Triangulo Ascendente",
            "type": "bullish",
            "confidence": 70,
            "description": (
                f"Resistencia plana em ${rh[-1][1]:.2f}, minimos subindo. "
                "Pressao compradora crescente. Rompimento esperado para cima."
            ),
            "key_levels": {
                "Resistencia Plana": round(rh[-1][1], 2),
                "Suporte Atual": round(rl[-1][1], 2),
            },
        }

    if hn < -0.015 and abs(ln) < 0.015:
        return {
            "name": "Triangulo Descendente",
            "type": "bearish",
            "confidence": 70,
            "description": (
                f"Suporte plano em ${rl[-1][1]:.2f}, maximos caindo. "
                "Pressao vendedora crescente. Rompimento esperado para baixo."
            ),
            "key_levels": {
                "Suporte Plano": round(rl[-1][1], 2),
                "Resistencia Atual": round(rh[-1][1], 2),
            },
        }

    if hn < -0.008 and ln > 0.008:
        return {
            "name": "Triangulo Simetrico",
            "type": "neutral",
            "confidence": 62,
            "description": (
                "Compressao de volatilidade: maximos caindo e minimos subindo. "
                "Rompimento iminente — direcao ainda indefinida. Aguarde o lado que vencer."
            ),
            "key_levels": {
                "Resistencia": round(rh[-1][1], 2),
                "Suporte": round(rl[-1][1], 2),
            },
        }

    return None


def detect_flag(df: pd.DataFrame) -> dict | None:
    """
    Detecta Bull Flag (bullish) e Bear Flag (bearish).
    Logica: movimento forte (pole) seguido de consolidacao (flag).
    """
    if len(df) < 30:
        return None

    close = df["Close"]
    pole      = close.iloc[-20:-10]
    flag_part = close.iloc[-10:]

    if len(pole) < 5 or len(flag_part) < 5:
        return None

    pole_ret = (float(pole.iloc[-1]) - float(pole.iloc[0])) / float(pole.iloc[0]) * 100
    flag_ret = (float(flag_part.iloc[-1]) - float(flag_part.iloc[0])) / float(flag_part.iloc[0]) * 100

    if pole_ret > 8 and -6 < flag_ret < 1:
        target = float(close.iloc[-1]) * (1 + abs(pole_ret) / 100)
        return {
            "name": "Bull Flag",
            "type": "bullish",
            "confidence": 68,
            "description": (
                f"Alta de {pole_ret:.1f}% (mastro) seguida de consolidacao ({flag_ret:.1f}%). "
                f"Padrao de continuacao bullish. Alvo projetado: ${target:.2f}."
            ),
            "key_levels": {
                "Alvo Projetado": round(target, 2),
                "Suporte da Flag": round(float(flag_part.min()), 2),
            },
        }

    if pole_ret < -8 and -1 < flag_ret < 6:
        target = float(close.iloc[-1]) * (1 + pole_ret / 100)
        return {
            "name": "Bear Flag",
            "type": "bearish",
            "confidence": 68,
            "description": (
                f"Queda de {abs(pole_ret):.1f}% (mastro) seguida de recuperacao ({flag_ret:.1f}%). "
                f"Padrao de continuacao bearish. Alvo projetado: ${target:.2f}."
            ),
            "key_levels": {
                "Alvo Projetado": round(target, 2),
                "Resistencia da Flag": round(float(flag_part.max()), 2),
            },
        }

    return None


def detect_all_patterns(df: pd.DataFrame) -> list[dict]:
    """
    Executa todos os detectores e retorna lista de padroes encontrados,
    ordenados por confianca decrescente.
    """
    if df.empty or len(df) < 30:
        return []

    patterns = []
    for fn in [detect_head_and_shoulders, detect_double_pattern, detect_triangle, detect_flag]:
        try:
            result = fn(df)
            if result is not None:
                patterns.append(result)
        except Exception:
            continue

    return sorted(patterns, key=lambda p: p["confidence"], reverse=True)
