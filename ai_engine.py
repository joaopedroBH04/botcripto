"""
ai_engine.py — Motor de Analise Inteligente BotCripto v2

Transforma dados quantitativos em narrativas tecnicas contextuais,
detecta padroes tecnicos, gera resumo de mercado, insights de portfolio
e interpretacao de backtests. Cada saida e construida a partir dos
valores reais dos indicadores — sem templates genericos.
"""

import pandas as pd
import numpy as np
from typing import Optional


# ─── Classificadores de estado dos indicadores ───────────────────────────────

def _rsi_state(v: float) -> tuple[str, str]:
    """Retorna (estado, descricao_curta)."""
    if v <= 20: return "sobrevenda_extrema",  f"RSI em sobrevenda extrema ({v:.1f})"
    if v <= 30: return "sobrevenda",           f"RSI em sobrevenda ({v:.1f})"
    if v <= 45: return "neutro_fraco",         f"RSI levemente baixo ({v:.1f})"
    if v <= 55: return "neutro",               f"RSI neutro ({v:.1f})"
    if v <= 68: return "neutro_forte",         f"RSI levemente elevado ({v:.1f})"
    if v <= 80: return "sobrecompra",          f"RSI em sobrecompra ({v:.1f})"
    return         "sobrecompra_extrema",      f"RSI em sobrecompra extrema ({v:.1f})"


def _macd_state(hist: float, prev_hist: float) -> tuple[str, str]:
    if hist > 0 and prev_hist <= 0:
        return "cruzamento_bullish",          "MACD com cruzamento bullish recente"
    if hist < 0 and prev_hist >= 0:
        return "cruzamento_bearish",          "MACD com cruzamento bearish recente"
    if hist > 0 and hist > prev_hist:
        return "momentum_acelerando_alta",    "MACD em momentum de alta acelerando"
    if hist > 0 and hist < prev_hist:
        return "momentum_desacelerando_alta", "MACD em alta mas perdendo forca"
    if hist < 0 and abs(hist) > abs(prev_hist):
        return "momentum_acelerando_baixa",   "MACD em queda acelerando"
    if hist < 0 and abs(hist) < abs(prev_hist):
        return "momentum_desacelerando_baixa","MACD em baixa mas desacelerando"
    return "neutro", "MACD neutro"


def _bb_state(price: float, upper: float, lower: float, middle: float) -> tuple[str, str]:
    width_pct = (upper - lower) / middle * 100 if middle else 0
    pos_pct   = (price - lower) / (upper - lower) * 100 if upper != lower else 50

    if width_pct < 3:  return "squeeze_extremo", "Bandas de Bollinger em squeeze extremo"
    if width_pct < 5:  return "squeeze",          "Bandas de Bollinger comprimidas"
    if pos_pct >= 95:  return "topo_banda",        "Preco tocando banda superior"
    if pos_pct <= 5:   return "fundo_banda",       "Preco tocando banda inferior"
    if pos_pct >= 70:  return "zona_superior",     "Preco na zona superior das Bandas"
    if pos_pct <= 30:  return "zona_inferior",     "Preco na zona inferior das Bandas"
    return "meio", "Preco no centro das Bandas"


def _trend_state(price, sma20, sma50, sma200) -> tuple[str, str]:
    smas  = [s for s in [sma20, sma50, sma200] if s is not None]
    bulls = sum(price > s for s in smas)
    align = sum(
        a > b
        for a, b in [(sma20, sma50), (sma50, sma200)]
        if a is not None and b is not None
    )
    if bulls == 3 and align == 2: return "tendencia_alta_perfeita", "Tendencia de alta em todas as medias"
    if bulls >= 2 and align >= 1: return "tendencia_alta",          "Tendencia de alta predominante"
    if bulls == 0 and align == 0: return "tendencia_baixa_perfeita","Tendencia de baixa em todas as medias"
    if bulls <= 1:                return "tendencia_baixa",         "Tendencia de baixa"
    return "tendencia_indefinida", "Tendencia indefinida / lateral"


def _adx_state(adx: Optional[float]) -> tuple[str, str]:
    if adx is None:   return "desconhecido",       ""
    if adx < 15:      return "sem_tendencia",      f"ADX {adx:.0f} — mercado lateral sem tendencia"
    if adx < 25:      return "tendencia_fraca",    f"ADX {adx:.0f} — tendencia fraca"
    if adx < 40:      return "tendencia_forte",    f"ADX {adx:.0f} — tendencia confirmada"
    return             "tendencia_muito_forte",    f"ADX {adx:.0f} — tendencia muito forte"


def _fib_context(price: float, fib_levels: dict) -> tuple[str, str]:
    if not fib_levels:
        return "sem_fibonacci", ""
    nearest = min(fib_levels, key=lambda k: abs(fib_levels[k] - price))
    np_price = fib_levels[nearest]
    dist = abs(price - np_price) / np_price * 100
    if dist <= 1.5:
        tipo = "suporte" if price >= np_price else "resistencia"
        return f"em_{tipo}_fib", f"Confluencia com nivel Fibonacci {nearest} (${np_price:,.2f}) como {tipo}"
    return "longe_fibonacci", f"Fibonacci mais proximo: {nearest} em ${np_price:,.2f} ({dist:.1f}% de distancia)"


# ─── Catalogos de narrativa ───────────────────────────────────────────────────

_RSI_INTERPRETATION = {
    "sobrevenda_extrema": "Nivel historicamente raro — vendedores possivelmente esgotados. Frequentemente precede recuperacoes tecnicas expressivas.",
    "sobrevenda":         "Zona de sobrevenda indica pressao vendedora excessiva. Monitorar sinais de reversao para entrada.",
    "neutro_fraco":       "Leve pressao vendedora sem excessos — nao configura oportunidade nem risco imediato.",
    "neutro":             "RSI em territorio neutro, sem vieses extremos em nenhuma direcao.",
    "neutro_forte":       "Levemente elevado, mas sem configurar sobrecompra — compradores ainda controlam.",
    "sobrecompra":        "Zona de sobrecompra aumenta o risco de correcao de curto prazo.",
    "sobrecompra_extrema":"Nivel extremo — historicamente associado a correcoes ou consolidacoes fortes.",
}

_MACD_INTERPRETATION = {
    "cruzamento_bullish":           "O cruzamento do MACD para cima e um dos sinais tecnicos mais relevantes de reversao de momentum.",
    "cruzamento_bearish":           "Cruzamento bearish do MACD — mudanca de momentum. Sinal de cautela para posicoes compradas.",
    "momentum_acelerando_alta":     "Histograma em expansao positiva — momentum de alta se intensificando.",
    "momentum_desacelerando_alta":  "Histograma diminuindo — momentum de alta perdendo forca, possivel exaustao a vista.",
    "momentum_acelerando_baixa":    "Pressao vendedora crescente e acelerando.",
    "momentum_desacelerando_baixa": "Pressao vendedora diminuindo — possivel esgotamento dos vendedores.",
    "neutro":                       "MACD sem direcao clara — mercado em equilibrio temporario.",
}

_TREND_INTERPRETATION = {
    "tendencia_alta_perfeita": (
        "O ativo esta alinhado em alta em todas as medias moveis — preco acima das SMAs 20, 50 e 200, "
        "todas em sequencia correta. Esta e a configuracao tecnica mais favoravel para tendencias de alta."
    ),
    "tendencia_alta": "Maioria das medias moveis alinhadas positivamente — predominancia de alta.",
    "tendencia_indefinida": "Medias moveis mistas, sem direcao clara. Mercado em fase de definicao.",
    "tendencia_baixa": "Estrutura bearish nas medias moveis — contexto desfavoravel para posicoes compradas.",
    "tendencia_baixa_perfeita": (
        "Alinhamento bearish perfeito — preco abaixo de todas as medias em sequencia descendente. "
        "Risco elevado para compradores."
    ),
}

_BB_INTERPRETATION = {
    "squeeze_extremo": " As Bandas de Bollinger estao em squeeze extremo — volatilidade comprimida ao maximo. Historicamente, squeezes assim precedem movimentos violentos.",
    "squeeze":         " Bandas comprimidas — o mercado acumula energia antes de um movimento mais expressivo.",
    "topo_banda":      " Preco tocando a banda superior — resistencia estatistica que costuma provocar realizacao.",
    "fundo_banda":     " Preco na banda inferior — zona de suporte estatistico onde compradores historicamente aparecem.",
    "zona_superior":   " Preco na metade superior das Bandas, levemente favoravel para comprados.",
    "zona_inferior":   " Preco na metade inferior das Bandas — zona de potencial acumulacao.",
    "meio":            "",
}

PATTERN_NAMES = {
    "REVERSAO_BULLISH_CONFLUENTE": "Reversao Bullish Confluente",
    "REVERSAO_BULLISH":            "Reversao Bullish",
    "CONTINUACAO_BULLISH":         "Continuacao de Alta",
    "ACUMULACAO":                  "Zona de Acumulacao",
    "COMPRESSAO_PRE_MOVIMENTO":    "Compressao Pre-Movimento",
    "EXAUSTAO_COMPRADORES":        "Exaustao de Compradores",
    "REVERSAO_BEARISH":            "Reversao Bearish",
    "TENDENCIA_BAIXA_ACELERANDO":  "Tendencia de Baixa Acelerando",
    "DIVERGENCIA_BULLISH":         "Divergencia Bullish",
    "NEUTRO_INDEFINIDO":           "Mercado Indefinido",
}

_PATTERN_CONCLUSIONS = {
    "REVERSAO_BULLISH_CONFLUENTE": "Multiplos indicadores convergem para um setup de reversao de alta probabilidade. Entradas escalonadas (DCA) podem ser consideradas com stop abaixo do suporte.",
    "DIVERGENCIA_BULLISH":         "A divergencia bullish e considerada um dos sinais mais confiaveis de reversao tecnica — preco cai mas indicadores nao confirmam. Setup de alta qualidade.",
    "REVERSAO_BULLISH":            "Setup de reversao em formacao. Confirmar rompimento de resistencia proxima antes de entrar.",
    "CONTINUACAO_BULLISH":         "Tendencia de alta saudavel sem sinais de exaustao. Posicoes existentes podem ser mantidas; novas entradas em pullbacks.",
    "ACUMULACAO":                  "Preco em zona de acumulacao. Sem confirmacao de reversao ainda, mas risco/retorno favoravel para quem tolera volatilidade.",
    "COMPRESSAO_PRE_MOVIMENTO":    "Compressao precede movimentos expressivos. Monitorar o lado do rompimento para direcionar posicao.",
    "EXAUSTAO_COMPRADORES":        "Sinais de exaustao — cautela para novas entradas compradas. Considerar reducao gradual.",
    "REVERSAO_BEARISH":            "Configuracao de reversao bearish. Posicoes compradas em risco — considerar reducao.",
    "TENDENCIA_BAIXA_ACELERANDO":  "Tendencia de baixa com momentum crescente. Aguardar sinais de reversao antes de posicionar.",
    "NEUTRO_INDEFINIDO":           "Sem padrao claro — mercado em fase de definicao. Aguardar confirmacao antes de operar.",
}


# ─── Detector de Padroes ──────────────────────────────────────────────────────

def detect_pattern(df: pd.DataFrame, score_result: dict) -> tuple[str, str, list[str]]:
    """
    Detecta o padrao primario baseado na combinacao de indicadores.
    Retorna (pattern_id, confianca, [razoes]).
    confianca: 'alta' | 'media' | 'baixa'
    """
    if df.empty or score_result is None:
        return "NEUTRO_INDEFINIDO", "baixa", []

    price     = df["Close"].iloc[-1]
    rsi       = df["rsi_14"].iloc[-1]      if "rsi_14"    in df.columns else 50.0
    mh        = df["macd_hist"].iloc[-1]   if "macd_hist" in df.columns else 0.0
    mh_prev   = df["macd_hist"].iloc[-2]   if "macd_hist" in df.columns and len(df) > 1 else 0.0
    bb_upper  = df["bb_upper"].iloc[-1]    if "bb_upper"  in df.columns else None
    bb_lower  = df["bb_lower"].iloc[-1]    if "bb_lower"  in df.columns else None
    bb_mid    = df["bb_mid"].iloc[-1]      if "bb_mid"    in df.columns else None
    sma20     = df["sma_20"].iloc[-1]      if "sma_20"    in df.columns else None
    sma50     = df["sma_50"].iloc[-1]      if "sma_50"    in df.columns else None
    sma200    = df["sma_200"].iloc[-1]     if "sma_200"   in df.columns else None
    adx       = df["adx"].iloc[-1]         if "adx"       in df.columns else None
    fib_levels = df.attrs.get("fibonacci", {}).get("levels", {})

    rsi_st,   _ = _rsi_state(rsi)
    macd_st,  _ = _macd_state(mh, mh_prev)
    trend_st, _ = _trend_state(price, sma20, sma50, sma200)
    adx_st,   _ = _adx_state(adx)
    fib_st,   _ = _fib_context(price, fib_levels)

    bb_st = "meio"
    if all(v is not None for v in [bb_upper, bb_lower, bb_mid]):
        bb_st, _ = _bb_state(price, bb_upper, bb_lower, bb_mid)

    divs    = score_result.get("divergences", {})
    rsi_div  = (divs.get("rsi")  or {}).get("type", "none")
    macd_div = (divs.get("macd") or {}).get("type", "none")
    score   = score_result.get("score", 50)

    razoes: list[str] = []

    # Divergencias bullish — sinal de maior qualidade
    if rsi_div == "bullish" or macd_div == "bullish":
        tipos = []
        if rsi_div == "bullish":  tipos.append("RSI")
        if macd_div == "bullish": tipos.append("MACD")
        razoes.append(f"Divergencia bullish detectada em {' e '.join(tipos)} — preco faz novas minimas mas indicadores nao confirmam")
        if bb_st == "fundo_banda":
            razoes.append("Confluencia adicional: preco na banda inferior de Bollinger")
        if "suporte" in fib_st:
            razoes.append("Confluencia com suporte Fibonacci")
        return "DIVERGENCIA_BULLISH", "alta", razoes

    # Reversao bullish
    if rsi_st in ("sobrevenda", "sobrevenda_extrema"):
        razoes.append(f"RSI em {rsi:.1f} — zona de sobrevenda{'extrema' if rsi <= 20 else ''}")
        if macd_st == "cruzamento_bullish":
            razoes.append("MACD cruzou para cima — confirmacao de reversao")
            if bb_st == "fundo_banda":
                razoes.append("Tripla confluencia: RSI sobrevenda + cruzamento MACD + banda Bollinger inferior")
                if "suporte" in fib_st: razoes.append("Fibonacci como suporte adicional")
                return "REVERSAO_BULLISH_CONFLUENTE", "alta", razoes
            return "REVERSAO_BULLISH", "alta" if score >= 60 else "media", razoes
        if macd_st in ("momentum_acelerando_alta", "momentum_desacelerando_baixa"):
            razoes.append("MACD indica reducao de pressao vendedora")
            return "REVERSAO_BULLISH", "media", razoes
        if bb_st == "fundo_banda":
            razoes.append("Banda Bollinger inferior como suporte estatistico")
            return "ACUMULACAO", "media", razoes
        razoes.append("Aguardar confirmacao de MACD para maior confianca")
        return "REVERSAO_BULLISH", "baixa", razoes

    # Continuacao bullish
    if trend_st in ("tendencia_alta_perfeita", "tendencia_alta"):
        razoes.append("Tendencia de alta confirmada nas medias moveis")
        if macd_st in ("cruzamento_bullish", "momentum_acelerando_alta"):
            razoes.append("MACD confirma momentum positivo")
            if adx_st in ("tendencia_forte", "tendencia_muito_forte"):
                razoes.append(f"ADX em {adx:.0f} — forca da tendencia confirmada")
            if rsi_st not in ("sobrecompra", "sobrecompra_extrema"):
                razoes.append("RSI com espaco para subir — sem sobrecompra")
                return "CONTINUACAO_BULLISH", "alta", razoes
            return "CONTINUACAO_BULLISH", "media", razoes

    # Compressao de volatilidade
    if bb_st in ("squeeze", "squeeze_extremo"):
        razoes.append("Bandas de Bollinger em compressao — volatilidade historicamente baixa")
        razoes.append("Squeezes precedem movimentos expressivos de preco")
        if trend_st in ("tendencia_alta_perfeita", "tendencia_alta"):
            razoes.append("Contexto de tendencia de alta favorece rompimento para cima")
        return "COMPRESSAO_PRE_MOVIMENTO", "media", razoes

    # Exaustao / reversao bearish
    if rsi_st in ("sobrecompra", "sobrecompra_extrema"):
        razoes.append(f"RSI em {rsi:.1f} — zona de sobrecompra")
        if macd_st == "momentum_desacelerando_alta" and bb_st == "topo_banda":
            razoes.append("MACD perdendo forca na banda superior — divergencia de momentum classica")
            return "EXAUSTAO_COMPRADORES", "alta", razoes
        if macd_st == "cruzamento_bearish":
            razoes.append("MACD com cruzamento bearish — confirmacao de reversao")
            return "REVERSAO_BEARISH", "alta", razoes
        razoes.append("Sem confirmacao de reversao ainda — cautela, nao panique")
        return "EXAUSTAO_COMPRADORES", "media", razoes

    if macd_st == "momentum_acelerando_baixa" and trend_st in ("tendencia_baixa", "tendencia_baixa_perfeita"):
        razoes.append("Tendencia de baixa com MACD acelerando para baixo")
        razoes.append("Posicoes compradas em alto risco neste contexto")
        return "TENDENCIA_BAIXA_ACELERANDO", "alta", razoes

    razoes.append("Indicadores sem consenso — aguardar confirmacao antes de agir")
    return "NEUTRO_INDEFINIDO", "baixa", razoes


# ─── Gerador de Narrativa Tecnica ─────────────────────────────────────────────

def generate_technical_narrative(
    df: pd.DataFrame,
    score_result: dict,
    asset_name: str,
) -> str:
    """
    Gera narrativa tecnica completa em HTML para o ativo fornecido.
    Cada paragrafo e construido a partir dos valores reais dos indicadores.
    """
    if df.empty or score_result is None:
        return "<p style='color:#8B9AB0;'>Dados insuficientes para analise narrativa.</p>"

    price      = df["Close"].iloc[-1]
    prev_5d    = df["Close"].iloc[-6] if len(df) > 5 else df["Close"].iloc[0]
    change_5d  = (price - prev_5d) / prev_5d * 100

    rsi    = df["rsi_14"].iloc[-1]    if "rsi_14"    in df.columns else 50.0
    mh     = df["macd_hist"].iloc[-1] if "macd_hist" in df.columns else 0.0
    mh_p   = df["macd_hist"].iloc[-2] if "macd_hist" in df.columns and len(df) > 1 else 0.0
    atr    = df["atr_14"].iloc[-1]    if "atr_14"    in df.columns else None
    sma20  = df["sma_20"].iloc[-1]    if "sma_20"    in df.columns else None
    sma50  = df["sma_50"].iloc[-1]    if "sma_50"    in df.columns else None
    sma200 = df["sma_200"].iloc[-1]   if "sma_200"   in df.columns else None
    adx    = df["adx"].iloc[-1]       if "adx"       in df.columns else None

    bb_upper = df["bb_upper"].iloc[-1] if "bb_upper" in df.columns else None
    bb_lower = df["bb_lower"].iloc[-1] if "bb_lower" in df.columns else None
    bb_mid   = df["bb_mid"].iloc[-1]   if "bb_mid"   in df.columns else None

    fib_levels = df.attrs.get("fibonacci", {}).get("levels", {})
    score      = score_result.get("score", 50)
    divs       = score_result.get("divergences", {})

    rsi_st,   rsi_desc   = _rsi_state(rsi)
    macd_st,  macd_desc  = _macd_state(mh, mh_p)
    trend_st, trend_desc = _trend_state(price, sma20, sma50, sma200)
    adx_st,   adx_desc   = _adx_state(adx)
    fib_st,   fib_desc   = _fib_context(price, fib_levels)

    bb_st, bb_desc = "meio", ""
    if all(v is not None for v in [bb_upper, bb_lower, bb_mid]):
        bb_st, bb_desc = _bb_state(price, bb_upper, bb_lower, bb_mid)

    pattern, confidence, razoes = detect_pattern(df, score_result)

    # ── P1: Contexto de preco ────────────────────────────────────────────────
    vol_desc = ""
    if atr is not None:
        atr_pct = atr / price * 100
        if atr_pct > 5:   vol_desc = f", com volatilidade diaria elevada ({atr_pct:.1f}%)"
        elif atr_pct > 2: vol_desc = f", com volatilidade diaria moderada ({atr_pct:.1f}%)"
        else:             vol_desc = f", com volatilidade diaria baixa ({atr_pct:.1f}%)"

    if change_5d > 1:      dir5 = f"valorizou <span style='color:#00E5C3;font-weight:600;'>{change_5d:+.1f}%</span>"
    elif change_5d < -1:   dir5 = f"recuou <span style='color:#FF4757;font-weight:600;'>{abs(change_5d):.1f}%</span>"
    else:                  dir5 = "ficou praticamente estavel"

    p1 = (
        f"<strong>{asset_name}</strong> e negociado a "
        f"<strong style='color:#E8EDF5;'>${price:,.2f}</strong>{vol_desc}. "
        f"Nos ultimos 5 dias, o ativo {dir5}."
    )

    # ── P2: Momentum (RSI + MACD) ────────────────────────────────────────────
    rsi_txt  = _RSI_INTERPRETATION.get(rsi_st, "")
    macd_txt = _MACD_INTERPRETATION.get(macd_st, "")
    adx_txt  = f" {adx_desc}." if adx_st not in ("desconhecido", "sem_tendencia") else ""

    p2 = (
        f"<strong>Momentum:</strong> {rsi_desc}. {rsi_txt} "
        f"{macd_desc}. {macd_txt}{adx_txt}"
    )

    # ── P3: Estrutura (tendencia + BB + Fibonacci) ───────────────────────────
    trend_txt = _TREND_INTERPRETATION.get(trend_st, trend_desc)
    bb_txt    = _BB_INTERPRETATION.get(bb_st, "")
    fib_txt   = f" {fib_desc}." if fib_st not in ("longe_fibonacci", "sem_fibonacci") else ""

    p3 = f"<strong>Estrutura:</strong> {trend_txt}{bb_txt}{fib_txt}"

    # ── P4: Padrao + Conclusao ───────────────────────────────────────────────
    conf_labels  = {"alta": "Alta confianca", "media": "Confianca media", "baixa": "Baixa confianca"}
    pattern_name = PATTERN_NAMES.get(pattern, pattern)
    conclusion   = _PATTERN_CONCLUSIONS.get(pattern, "")
    conf_colors  = {"alta": "#00E5C3", "media": "#FFB800", "baixa": "#8B9AB0"}
    conf_color   = conf_colors.get(confidence, "#8B9AB0")

    # Divergencias no resumo
    rsi_div  = (divs.get("rsi")  or {}).get("type", "none")
    macd_div = (divs.get("macd") or {}).get("type", "none")
    div_html = ""
    if rsi_div == "bullish" or macd_div == "bullish":
        fontes = " e ".join(f for f, t in [("RSI", rsi_div), ("MACD", macd_div)] if t == "bullish")
        div_html = (
            f"<div style='margin-top:10px;padding:8px 14px;border-left:3px solid #4A9EFF;"
            f"background:rgba(74,158,255,0.06);border-radius:6px;font-size:0.82rem;color:#A0C0D8;'>"
            f"Divergencia bullish detectada em <strong>{fontes}</strong> — "
            f"sinal classico de potencial reversao de fundo.</div>"
        )

    p4 = (
        f"<strong>Padrao detectado:</strong> "
        f"<span style='color:{conf_color};font-weight:700;'>{pattern_name}</span> "
        f"<span style='color:#607888;font-size:0.78rem;'>({conf_labels.get(confidence, '')})</span><br>"
        f"<span style='color:#A0B5C5;'>{conclusion} Score atual: <strong style='color:{conf_color};'>{score}/100</strong>.</span>"
        f"{div_html}"
    )

    return f"""
<div style="line-height:1.9;font-size:0.86rem;color:#B0C5D5;">
  <p style="margin:0 0 14px 0;">{p1}</p>
  <p style="margin:0 0 14px 0;">{p2}</p>
  <p style="margin:0 0 14px 0;">{p3}</p>
  <p style="margin:0;">{p4}</p>
</div>"""


# ─── Resumo Executivo do Mercado ──────────────────────────────────────────────

def generate_market_pulse(scores: list[dict], fg_val: int, btc_dom: float) -> str:
    """Gera resumo executivo do estado geral do mercado em HTML."""
    if not scores:
        return ""

    vals     = [s.get("Score", 0) for s in scores]
    avg      = float(np.mean(vals))
    n_strong = sum(1 for v in vals if v >= 72)
    n_sell   = sum(1 for v in vals if v <  30)
    n_watch  = sum(1 for v in vals if 55 <= v < 72)
    n        = len(vals)

    rsi_divs  = sum(1 for s in scores if (s.get("_score_result") or {}).get("divergences", {}).get("rsi",  {}).get("type") == "bullish")
    macd_divs = sum(1 for s in scores if (s.get("_score_result") or {}).get("divergences", {}).get("macd", {}).get("type") == "bullish")

    # Sentimento
    if   fg_val <= 20: fg_txt = f"medo extremo (<strong>{fg_val}</strong>) — historicamente um momento de oportunidade para compradores contrarios"
    elif fg_val <= 40: fg_txt = f"medo prevalente (<strong>{fg_val}</strong>) — mercado cauteloso, correcoes de medo criam oportunidades"
    elif fg_val <= 60: fg_txt = f"sentimento neutro (<strong>{fg_val}</strong>) — sem vieses extremos no mercado"
    elif fg_val <= 75: fg_txt = f"ganancia (<strong>{fg_val}</strong>) — momentum positivo, mas monitorar sinais de sobrecompra"
    else:              fg_txt = f"ganancia extrema (<strong>{fg_val}</strong>) — historicamente proximo de topos de curto prazo"

    # Breadth
    if   n_strong >= n * 0.4: breadth = f"Ampla maioria (<strong>{n_strong}/{n}</strong>) dos ativos em compra forte — mercado bull em andamento"
    elif n_strong >= n * 0.2: breadth = f"<strong>{n_strong}</strong> de {n} ativos em compra forte — seletividade favoravel"
    elif n_sell   >= n * 0.4: breadth = f"Deterioracao ampla: <strong>{n_sell}/{n}</strong> ativos em zona de venda"
    elif avg >= 60:            breadth = f"Score medio <strong>{avg:.0f}/100</strong> — momentum positivo generalizado"
    elif avg <= 40:            breadth = f"Score medio <strong>{avg:.0f}/100</strong> — fraqueza generalizada nos ativos monitorados"
    else:                      breadth = f"Score medio <strong>{avg:.0f}/100</strong> — mercado equilibrado, sem excessos"

    # Divergencias
    total_divs = rsi_divs + macd_divs
    if   total_divs >= 3: div_txt = f" <strong>{total_divs}</strong> divergencias bullish simultaneas detectadas — sinal classico de potencial reversao de fundo em multiplos ativos."
    elif total_divs >= 1: div_txt = f" <strong>{total_divs}</strong> ativo(s) com divergencia bullish — monitorar para possiveis reversoes."
    else:                 div_txt = ""

    # BTC dominance
    if   btc_dom >= 55: btc_txt = f" Bitcoin domina <strong>{btc_dom:.1f}%</strong> do mercado cripto — fluxo concentrado em BTC, altcoins em posicao secundaria."
    elif btc_dom <= 45: btc_txt = f" Dominancia BTC em <strong>{btc_dom:.1f}%</strong> — possivel rotacao para altcoins em formacao."
    else:               btc_txt = f" Dominancia BTC em <strong>{btc_dom:.1f}%</strong> — equilibrio entre Bitcoin e altcoins."

    # Watch
    watch_txt = f" <strong>{n_watch}</strong> ativos em observacao podem migrar para compra forte se o momentum continuar." if n_watch >= 3 else ""

    return f"""
<div style="font-size:0.85rem;color:#B0C5D5;line-height:1.9;">
  <p style="margin:0 0 8px 0;">O mercado opera em {fg_txt}.{btc_txt}</p>
  <p style="margin:0;">{breadth}.{div_txt}{watch_txt}</p>
</div>"""


# ─── Insights de Portfolio ────────────────────────────────────────────────────

def generate_portfolio_insights(portfolio: dict, scores: list[dict]) -> list[str]:
    """
    Retorna lista de insights HTML acionaveis sobre a composicao do portfolio.
    """
    if not portfolio:
        return []

    assets       = list(portfolio.keys())
    n            = len(assets)
    crypto_count = sum(1 for v in portfolio.values() if v.get("type") in ("crypto",))
    stock_count  = n - crypto_count
    insights     = []

    # Concentracao
    if n == 1:
        insights.append(
            "<strong>Concentracao maxima:</strong> Portfolio com apenas 1 ativo. "
            "Diversifique com pelo menos 3-5 ativos descorrelacionados para reduzir o risco nao-sistematico."
        )
    elif crypto_count == n and n > 1:
        insights.append(
            f"<strong>Diversificacao:</strong> Portfolio 100% em criptomoedas ({n} ativos). "
            "Criptos tem alta correlacao entre si — em correcoes de mercado, todos tendem a cair juntos. "
            "Acoes ou ETFs de renda fixa podem reduzir essa correlacao."
        )
    elif stock_count == n:
        insights.append(
            "<strong>Diversificacao:</strong> Portfolio 100% em acoes/ETFs. "
            "Uma pequena alocacao em Bitcoin (5-10%) pode reduzir a correlacao com o mercado tradicional."
        )
    elif crypto_count > 0 and stock_count > 0:
        pct_crypto = crypto_count / n * 100
        insights.append(
            f"<strong>Composicao:</strong> {crypto_count} cripto + {stock_count} acao/ETF "
            f"({pct_crypto:.0f}% em cripto). Portfolio diversificado entre classes de ativos."
        )

    # Scores dos ativos do portfolio
    score_map = {}
    for s in scores:
        aid = s.get("id", "")
        if aid:
            score_map[aid] = s.get("Score", 50)

    portfolio_scores, weak_assets, strong_assets = [], [], []
    for aid in assets:
        sc = score_map.get(aid)
        if sc is not None:
            portfolio_scores.append(sc)
            if sc < 30:  weak_assets.append((aid, sc))
            elif sc >= 72: strong_assets.append((aid, sc))

    if weak_assets:
        names = ", ".join(f"<strong>{a.upper()}</strong> ({s}/100)" for a, s in weak_assets)
        insights.append(
            f"<span style='color:#FF8090;'>Zona de venda:</span> {names} "
            "com fraqueza tecnica confirmada. Avalie reducao de posicao ou stop-loss."
        )

    if strong_assets:
        names = ", ".join(f"<strong>{a.upper()}</strong> ({s}/100)" for a, s in strong_assets)
        insights.append(
            f"<span style='color:#00E5C3;'>Posicao favoravel:</span> {names} "
            "em zona de compra forte. Mantenha ou considere incremento."
        )

    if portfolio_scores:
        avg = float(np.mean(portfolio_scores))
        if avg < 40:
            insights.append(
                f"<strong>Score medio do portfolio: {avg:.0f}/100</strong> — carteira com fraqueza tecnica. "
                "Revise as posicoes e considere rotacao para ativos com scores mais elevados."
            )
        elif avg >= 65:
            insights.append(
                f"<strong>Score medio do portfolio: {avg:.0f}/100</strong> — carteira bem posicionada tecnicamente."
            )
        else:
            insights.append(
                f"<strong>Score medio do portfolio: {avg:.0f}/100</strong> — posicionamento neutro a moderadamente positivo."
            )

    return insights


# ─── Narrativa de Backtest ────────────────────────────────────────────────────

def generate_backtest_narrative(
    trades: list[dict],
    final_equity: float,
    initial_capital: float,
    max_drawdown: float,
    asset_name: str,
) -> str:
    """Gera analise narrativa completa do resultado do backtest em HTML."""

    if not trades and abs(final_equity - initial_capital) < 1:
        return (
            "<p style='color:#8B9AB0;font-size:0.85rem;line-height:1.8;'>"
            "Nenhuma operacao foi realizada no periodo simulado. "
            "A estrategia nao gerou sinais suficientes com os parametros atuais. "
            "Considere reduzir o limiar de compra ou aguardar mais historico de score."
            "</p>"
        )

    total_return = (final_equity / initial_capital - 1) * 100
    n_trades     = len(trades)
    wins         = sum(1 for t in trades if t.get("_pnl_raw", 0) >= 0)
    losses       = n_trades - wins
    win_rate     = wins / n_trades * 100 if n_trades else 0.0

    win_vals  = [t["_pnl_raw"] for t in trades if t.get("_pnl_raw", 0) > 0]
    loss_vals = [abs(t["_pnl_raw"]) for t in trades if t.get("_pnl_raw", 0) < 0]
    avg_win   = float(np.mean(win_vals))  if win_vals  else 0.0
    avg_loss  = float(np.mean(loss_vals)) if loss_vals else 0.0

    pf = (wins * avg_win) / (losses * avg_loss) if losses > 0 and avg_loss > 0 else None

    # Avaliacoes qualitativas
    if   total_return >  50: ret_eval = "retorno excepcional"
    elif total_return >  20: ret_eval = "retorno forte"
    elif total_return >   5: ret_eval = "retorno positivo moderado"
    elif total_return >   0: ret_eval = "retorno levemente positivo"
    elif total_return > -10: ret_eval = "leve perda"
    elif total_return > -30: ret_eval = "perda moderada"
    else:                    ret_eval = "perda expressiva"

    if   max_drawdown < -5:   dd_eval, dd_color = "controlado",   "#00E5C3"
    elif max_drawdown < -15:  dd_eval, dd_color = "aceitavel",    "#FFB800"
    elif max_drawdown < -30:  dd_eval, dd_color = "elevado",      "#FF8800"
    else:                     dd_eval, dd_color = "severo",       "#FF4757"

    if   win_rate >= 60: wr_eval = "taxa de acerto alta"
    elif win_rate >= 45: wr_eval = "taxa de acerto razoavel"
    else:                wr_eval = "taxa de acerto abaixo do ideal"

    ret_color = "#00E5C3" if total_return >= 0 else "#FF4757"

    pf_txt = ""
    if pf is not None:
        if   pf >= 2.0: pf_txt = f" Profit factor de <strong>{pf:.2f}x</strong> — ganhos superam amplamente as perdas."
        elif pf >= 1.0: pf_txt = f" Profit factor de <strong>{pf:.2f}x</strong> — ganhos superam perdas por margem pequena."
        else:           pf_txt = f" Profit factor de <strong>{pf:.2f}x</strong> — perdas superam ganhos em magnitude: ajuste os parametros."

    # Consistencia recente
    consistency = ""
    if n_trades >= 4:
        recent_wins = sum(1 for t in trades[-3:] if t.get("_pnl_raw", 0) >= 0)
        if recent_wins == 3:   consistency = " As 3 ultimas operacoes foram lucrativas — estrategia em bom momento."
        elif recent_wins == 0: consistency = " As 3 ultimas operacoes foram perdedoras — possivel degradacao da estrategia no periodo recente."

    # Recomendacao final
    if total_return > 10 and max_drawdown > -20:
        rec = "Estrategia demonstrou bom equilibrio risco/retorno no periodo testado."
    elif total_return > 0 and max_drawdown < -30:
        rec = "Retorno positivo, mas drawdown elevado sugere ajuste nos niveis de stop ou reducao do tamanho de posicao."
    elif total_return <= 0 and win_rate < 40:
        rec = "Estrategia nao lucrativa neste periodo. Revise os limiares de compra/venda ou aguarde mais historico."
    else:
        rec = "Resultados mistos — considere otimizar os parametros ou ampliar o periodo de teste."

    return f"""
<div style="font-size:0.86rem;color:#B0C5D5;line-height:1.9;">
  <p style="margin:0 0 12px 0;">
    Simulacao de <strong>{asset_name}</strong>:
    retorno de <strong style="color:{ret_color};">{total_return:+.1f}%</strong> ({ret_eval})
    em <strong>{n_trades}</strong> operacao(es).
    Capital ${initial_capital:,.0f} → <strong style="color:{ret_color};">${final_equity:,.0f}</strong>.
  </p>
  <p style="margin:0 0 12px 0;">
    <strong>Qualidade:</strong> {wins} ganho(s) e {losses} perda(s)
    (<em>{wr_eval}</em>: {win_rate:.0f}%).
    Ganho medio: <span style="color:#00E5C3;">${avg_win:,.2f}</span> /
    Perda media: <span style="color:#FF4757;">${avg_loss:,.2f}</span>.{pf_txt}
  </p>
  <p style="margin:0;">
    Drawdown maximo: <strong style="color:{dd_color};">{max_drawdown:.1f}%</strong> ({dd_eval}).{consistency}
    <em style="color:#8B9AB0;">{rec}</em>
  </p>
</div>"""
