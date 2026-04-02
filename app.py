# ============================================================
# BotCripto v2 - Dashboard de Monitoramento Financeiro
# Execute com: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

from config import (
    CRYPTO_IDS, STOCK_TICKERS,
    BUY_CONFIDENCE_STRONG, BUY_CONFIDENCE_MODERATE, SELL_CONFIDENCE,
    FIBONACCI_LEVELS, FIBONACCI_LABELS,
    DEFAULT_PORTFOLIO_VALUE, DEFAULT_RISK_PER_TRADE,
)
from data_fetcher import (
    fetch_crypto_current, fetch_crypto_history,
    fetch_all_crypto_histories,
    fetch_stock_current, fetch_stock_history,
    fetch_fear_greed, get_fear_greed_current,
    fetch_btc_dominance, fetch_news,
)
from analysis import (
    compute_indicators, score_asset, detect_trend,
    classify_dip, generate_recommendation,
    compute_risk_metrics, compute_dca_plan, compute_correlation_matrix,
)

# -------------------------------------------------------
# Config da pagina
# -------------------------------------------------------
st.set_page_config(
    page_title="BotCripto v2 - Monitor Financeiro",
    page_icon="\U0001f4ca",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------
# CSS customizado
# -------------------------------------------------------
st.markdown("""
<style>
    .recommendation-box {
        background: #1a1f2e;
        border-left: 4px solid #00d4aa;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        line-height: 1.6;
    }
    .alert-buy {
        background: #0a2e1a;
        border: 1px solid #00d4aa;
        padding: 12px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .alert-sell {
        background: #2e0a0a;
        border: 1px solid #ff4444;
        padding: 12px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .divergence-box {
        background: #1a2e1a;
        border: 2px solid #00ff88;
        padding: 12px;
        border-radius: 5px;
        margin: 10px 0;
        font-weight: bold;
    }
    .confluence-box {
        background: #1a1f2e;
        border: 1px solid #3498db;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
    .risk-metric {
        background: #1a1f2e;
        border-radius: 5px;
        padding: 10px;
        margin: 3px 0;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------
st.sidebar.title("\U0001f4ca BotCripto v2")
st.sidebar.markdown("Monitor Inteligente de Mercado")
st.sidebar.caption("Scoring v2: 10 indicadores + Confluencia")

page = st.sidebar.radio(
    "Navegacao",
    [
        "\U0001f3e0 Visao Geral",
        "\U0001f50d Analise Detalhada",
        "\U0001f6e1 Gestao de Risco",
        "\U0001f514 Alertas",
        "\U0001f4f0 Noticias",
        "\U0001f4bc Portfolio",
    ],
)

st.sidebar.markdown("---")
if st.sidebar.button("\U0001f504 Atualizar Dados"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(
    "Aviso: Este sistema e apenas informativo. "
    "Nao constitui recomendacao de investimento. "
    "Sempre faca sua propria analise."
)


# -------------------------------------------------------
# Funcoes auxiliares
# -------------------------------------------------------

def get_all_assets() -> pd.DataFrame:
    """Busca dados atuais de todos os ativos monitorados."""
    crypto_df = fetch_crypto_current(CRYPTO_IDS)
    stock_df = fetch_stock_current(STOCK_TICKERS)
    frames = [df for df in [crypto_df, stock_df] if not df.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def get_history_and_analysis(asset_id: str, asset_type: str):
    """Busca historico e computa analise completa de um ativo."""
    if asset_type == "crypto":
        df = fetch_crypto_history(asset_id)
    else:
        df = fetch_stock_history(asset_id)

    if df.empty:
        return df, None, None

    df = compute_indicators(df)
    fg_val, _ = get_fear_greed_current()
    score_result = score_asset(df, fg_val)
    dip_info = classify_dip(df)

    return df, score_result, dip_info


def score_emoji(score: int) -> str:
    if score >= BUY_CONFIDENCE_STRONG:
        return "\U0001f7e2"
    elif score >= BUY_CONFIDENCE_MODERATE:
        return "\U0001f7e1"
    elif score >= SELL_CONFIDENCE:
        return "\u26aa"
    return "\U0001f534"


def format_number(n):
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "N/A"
    if abs(n) >= 1e12:
        return f"${n/1e12:.2f}T"
    if abs(n) >= 1e9:
        return f"${n/1e9:.2f}B"
    if abs(n) >= 1e6:
        return f"${n/1e6:.2f}M"
    if abs(n) >= 1e3:
        return f"${n/1e3:.2f}K"
    return f"${n:.2f}"


# -------------------------------------------------------
# GRAFICOS
# -------------------------------------------------------

def create_price_chart(df: pd.DataFrame, asset_name: str) -> go.Figure:
    """Cria grafico completo com todos os indicadores."""
    has_adx = "adx" in df.columns
    num_rows = 4 if has_adx else 3
    heights = [0.5, 0.17, 0.17, 0.16] if has_adx else [0.55, 0.22, 0.23]
    titles = [f"{asset_name} - Preco", "RSI / Stoch RSI", "MACD"]
    if has_adx:
        titles.append("ADX (Forca da Tendencia)")

    fig = make_subplots(
        rows=num_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=heights,
        subplot_titles=titles,
    )

    # --- Preco ---
    fig.add_trace(
        go.Scatter(x=df.index, y=df["Close"], name="Preco",
                   line=dict(color="#00d4aa", width=2)),
        row=1, col=1,
    )

    # SMAs
    sma_colors = {20: "#3498db", 50: "#f39c12", 200: "#e74c3c"}
    for period, color in sma_colors.items():
        col_name = f"sma_{period}"
        if col_name in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[col_name], name=f"SMA {period}",
                           line=dict(color=color, width=1, dash="dot"), opacity=0.7),
                row=1, col=1,
            )

    # EMAs
    if "ema_9" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["ema_9"], name="EMA 9",
                       line=dict(color="#e91e63", width=1.5)),
            row=1, col=1,
        )
    if "ema_21" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["ema_21"], name="EMA 21",
                       line=dict(color="#9c27b0", width=1.5)),
            row=1, col=1,
        )

    # Bollinger Bands
    if "bb_upper" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["bb_upper"], name="BB Sup",
                       line=dict(color="rgba(150,150,150,0.3)", width=1), showlegend=False),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["bb_lower"], name="BB Inf",
                       line=dict(color="rgba(150,150,150,0.3)", width=1),
                       fill="tonexty", fillcolor="rgba(150,150,150,0.05)", showlegend=False),
            row=1, col=1,
        )

    # Fibonacci levels
    fib = df.attrs.get("fibonacci", {})
    fib_levels = fib.get("levels", {})
    fib_colors = {"23.6%": "#b388ff", "38.2%": "#82b1ff", "50%": "#ffd54f",
                  "61.8%": "#ff8a65", "78.6%": "#ef5350"}
    for label, price in fib_levels.items():
        if label in fib_colors:
            fig.add_hline(y=price, line_dash="dash", line_color=fib_colors[label],
                          opacity=0.5, row=1, col=1,
                          annotation_text=f"Fib {label}", annotation_position="right")

    # --- RSI + Stoch RSI ---
    if "rsi_14" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["rsi_14"], name="RSI 14",
                       line=dict(color="#9b59b6", width=1.5)),
            row=2, col=1,
        )
    if "stoch_rsi_k" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["stoch_rsi_k"], name="Stoch RSI K",
                       line=dict(color="#00bcd4", width=1)),
            row=2, col=1,
        )
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)

    # --- MACD ---
    if "macd" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["macd"], name="MACD",
                       line=dict(color="#3498db", width=1.5)),
            row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["macd_signal"], name="Sinal",
                       line=dict(color="#e74c3c", width=1)),
            row=3, col=1,
        )
        if "macd_hist" in df.columns:
            colors = ["#00d4aa" if v >= 0 else "#ff4444" for v in df["macd_hist"].fillna(0)]
            fig.add_trace(
                go.Bar(x=df.index, y=df["macd_hist"], name="Histograma",
                       marker_color=colors, opacity=0.6),
                row=3, col=1,
            )

    # --- ADX ---
    if has_adx:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["adx"], name="ADX",
                       line=dict(color="#ffd54f", width=2)),
            row=4, col=1,
        )
        if "adx_pos" in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df["adx_pos"], name="DI+",
                           line=dict(color="#00d4aa", width=1)),
                row=4, col=1,
            )
        if "adx_neg" in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df["adx_neg"], name="DI-",
                           line=dict(color="#ff4444", width=1)),
                row=4, col=1,
            )
        fig.add_hline(y=25, line_dash="dash", line_color="white", opacity=0.3, row=4, col=1,
                      annotation_text="Tendencia forte", annotation_position="right")
        fig.add_hline(y=20, line_dash="dot", line_color="gray", opacity=0.3, row=4, col=1)

    fig.update_layout(
        height=850 if has_adx else 700,
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
        margin=dict(l=50, r=20, t=40, b=20),
    )
    fig.update_xaxes(rangeslider_visible=False)

    return fig


def create_gauge(score: int) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "Score de Compra", "font": {"size": 14, "color": "white"}},
        number={"font": {"size": 36, "color": "white"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "white"},
            "bar": {"color": "#00d4aa" if score >= 55 else ("#f0ad4e" if score >= 30 else "#ff4444")},
            "bgcolor": "#1a1f2e",
            "steps": [
                {"range": [0, 30], "color": "rgba(255,68,68,0.2)"},
                {"range": [30, 55], "color": "rgba(240,173,78,0.2)"},
                {"range": [55, 72], "color": "rgba(52,152,219,0.2)"},
                {"range": [72, 100], "color": "rgba(0,212,170,0.2)"},
            ],
        },
    ))
    fig.update_layout(
        height=220, template="plotly_dark",
        paper_bgcolor="#0e1117", margin=dict(l=20, r=20, t=35, b=10),
    )
    return fig


# -------------------------------------------------------
# Funcao para analisar todos os ativos de uma vez
# -------------------------------------------------------

def analyze_all_assets():
    """Busca e analisa todos os ativos. Retorna lista de dicts com scores."""
    all_assets = get_all_assets()
    if all_assets.empty:
        return [], all_assets

    crypto_assets = all_assets[all_assets["type"] == "crypto"]
    stock_assets = all_assets[all_assets["type"] == "stock"]

    with st.spinner("Analisando criptomoedas... (respeitando limites da API)"):
        crypto_ids = crypto_assets["id"].tolist() if not crypto_assets.empty else []
        crypto_histories = fetch_all_crypto_histories(crypto_ids) if crypto_ids else {}

    fg_val, _ = get_fear_greed_current()
    scores = []

    # Criptos
    for _, row in crypto_assets.iterrows():
        df = crypto_histories.get(row["id"], pd.DataFrame())
        if df.empty:
            continue
        df = compute_indicators(df)
        score_result = score_asset(df, fg_val)
        dip_info = classify_dip(df)
        if score_result:
            confluence = score_result.get("confluence", {})
            scores.append({
                "Ativo": f"{row['name']} ({row['symbol']})",
                "Tipo": "\U0001f4b0 Crypto",
                "Preco": f"${row['price']:.4f}" if row["price"] < 1 else (f"${row['price']:.2f}" if row["price"] < 1000 else format_number(row["price"])),
                "24h": f"{row['change_24h']:+.2f}%",
                "Score": score_result["score"],
                "Sinal": f"{score_emoji(score_result['score'])} {score_result['label']}",
                "Confluencia": f"{confluence.get('agree_buy', 0)}/{confluence.get('total', 10)}",
                "Tendencia": score_result["trend"].replace("_", " ").title(),
                "id": row["id"],
                "type": "crypto",
                "_score_result": score_result,
                "_dip_info": dip_info,
                "_df": df,
            })

    # Acoes
    with st.spinner("Analisando acoes e ETFs..."):
        for _, row in stock_assets.iterrows():
            df = fetch_stock_history(row["id"])
            if df.empty:
                continue
            df = compute_indicators(df)
            score_result = score_asset(df, fg_val)
            dip_info = classify_dip(df)
            if score_result:
                confluence = score_result.get("confluence", {})
                scores.append({
                    "Ativo": f"{row['name']} ({row['symbol']})",
                    "Tipo": "\U0001f4c8 Acao",
                    "Preco": f"${row['price']:.2f}" if row["price"] < 1000 else format_number(row["price"]),
                    "24h": f"{row['change_24h']:+.2f}%",
                    "Score": score_result["score"],
                    "Sinal": f"{score_emoji(score_result['score'])} {score_result['label']}",
                    "Confluencia": f"{confluence.get('agree_buy', 0)}/{confluence.get('total', 10)}",
                    "Tendencia": score_result["trend"].replace("_", " ").title(),
                    "id": row["id"],
                    "type": "stock",
                    "_score_result": score_result,
                    "_dip_info": dip_info,
                    "_df": df,
                })

    return scores, all_assets


# -------------------------------------------------------
# PAGINA: Visao Geral
# -------------------------------------------------------
def render_overview():
    st.title("\U0001f3e0 Visao Geral do Mercado")

    # Pulse do mercado
    col1, col2, col3 = st.columns(3)
    fg_val, fg_class = get_fear_greed_current()
    btc_dom = fetch_btc_dominance()

    with col1:
        fg_color = "normal" if 40 <= fg_val <= 60 else ("off" if fg_val > 60 else "inverse")
        st.metric("Fear & Greed Index", f"{fg_val} - {fg_class}",
                   delta=f"{'Medo' if fg_val < 40 else ('Ganancia' if fg_val > 60 else 'Neutro')}",
                   delta_color=fg_color)
    with col2:
        st.metric("Dominancia BTC", f"{btc_dom:.1f}%")
    with col3:
        st.metric("Ativos Monitorados", f"{len(CRYPTO_IDS) + len(STOCK_TICKERS)}")

    st.markdown("---")

    scores, all_assets = analyze_all_assets()

    scores_df = pd.DataFrame(scores)
    if scores_df.empty:
        st.warning("Nao foi possivel calcular scores. Tente novamente em 1 minuto.")
        return
    scores_df = scores_df.sort_values("Score", ascending=False)

    # Alertas de divergencia no topo
    divergence_alerts = []
    for _, row in scores_df.iterrows():
        sr = row.get("_score_result", {})
        divs = sr.get("divergences", {})
        rsi_div = divs.get("rsi", {}).get("type", "none")
        macd_div = divs.get("macd", {}).get("type", "none")
        if rsi_div == "bullish" or macd_div == "bullish":
            divergence_alerts.append(row["Ativo"])

    if divergence_alerts:
        st.markdown(
            '<div class="divergence-box">\U0001f4a1 DIVERGENCIA BULLISH DETECTADA em: '
            + ", ".join(divergence_alerts)
            + " — Sinal forte de possivel reversao para alta!</div>",
            unsafe_allow_html=True,
        )

    # Alertas de score
    strong_buys = scores_df[scores_df["Score"] >= BUY_CONFIDENCE_STRONG]
    strong_sells = scores_df[scores_df["Score"] <= SELL_CONFIDENCE]

    if not strong_buys.empty:
        st.success(
            "\U0001f7e2 **OPORTUNIDADES:** "
            + ", ".join([f"{r['Ativo']} (Score: {r['Score']}, {r['Confluencia']})" for _, r in strong_buys.iterrows()])
        )
    if not strong_sells.empty:
        st.error(
            "\U0001f534 **ZONA DE VENDA:** "
            + ", ".join([f"{r['Ativo']} (Score: {r['Score']})" for _, r in strong_sells.iterrows()])
        )

    # Tabela principal
    st.subheader("\U0001f4ca Ranking de Oportunidades")
    display_cols = ["Ativo", "Tipo", "Preco", "24h", "Score", "Confluencia", "Sinal", "Tendencia"]
    display_df = scores_df[[c for c in display_cols if c in scores_df.columns]].reset_index(drop=True)
    display_df.index += 1

    st.dataframe(
        display_df,
        use_container_width=True,
        height=min(len(display_df) * 40 + 40, 600),
        column_config={
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
        },
    )

    # Top 3 recomendacoes
    st.subheader("\U0001f4a1 Top Recomendacoes")
    top3 = scores_df.head(3)
    cols = st.columns(min(len(top3), 3))
    for i, (_, row) in enumerate(top3.iterrows()):
        if i >= 3:
            break
        with cols[i]:
            sr = row.get("_score_result")
            di = row.get("_dip_info")
            if sr and di:
                rec = generate_recommendation(sr, di, row["Ativo"])
                st.markdown(f'<div class="recommendation-box">{rec}</div>', unsafe_allow_html=True)


# -------------------------------------------------------
# PAGINA: Analise Detalhada
# -------------------------------------------------------
def render_deep_dive():
    st.title("\U0001f50d Analise Detalhada")

    col1, col2 = st.columns(2)
    with col1:
        asset_type = st.selectbox("Tipo", ["Criptomoeda", "Acao/ETF"])
    with col2:
        if asset_type == "Criptomoeda":
            asset_id = st.selectbox("Ativo", CRYPTO_IDS, format_func=lambda x: x.title())
            a_type = "crypto"
        else:
            asset_id = st.selectbox("Ativo", STOCK_TICKERS)
            a_type = "stock"

    with st.spinner("Carregando analise completa..."):
        df, score_result, dip_info = get_history_and_analysis(asset_id, a_type)

    if df is None or df.empty:
        st.error(f"Nao foi possivel carregar dados de {asset_id}")
        return

    # Header
    current_price = df["Close"].iloc[-1]
    prev_price = df["Close"].iloc[-2] if len(df) > 1 else current_price
    change = ((current_price - prev_price) / prev_price) * 100

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Preco Atual", f"${current_price:.2f}", f"{change:+.2f}%")
    with col2:
        st.metric("Maxima (Periodo)", f"${df['Close'].max():.2f}")
    with col3:
        st.metric("Minima (Periodo)", f"${df['Close'].min():.2f}")
    with col4:
        from_high = ((current_price - df['Close'].max()) / df['Close'].max()) * 100
        st.metric("Dist. da Maxima", f"{from_high:.1f}%")

    # Score + Confluencia + Divergencias
    if score_result:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.plotly_chart(create_gauge(score_result["score"]), use_container_width=True)

        with col2:
            # Confluencia
            conf = score_result.get("confluence", {})
            agree = conf.get("agree_buy", 0)
            total = conf.get("total", 10)
            pct = conf.get("percentage", 0)
            st.markdown(f"""
            <div class="confluence-box">
                <h3 style="margin:0">{agree}/{total}</h3>
                <p style="margin:0">indicadores concordam ({pct}%)</p>
                <p style="margin:5px 0 0 0; font-size:0.85em; color:#aaa">
                    {"Alta confianca" if pct >= 60 else ("Moderada" if pct >= 40 else "Baixa confianca")}
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"**Tendencia:** {score_result['trend'].replace('_', ' ').title()}")
            st.markdown(f"**Classificacao:** {score_result['label']}")

        with col3:
            # Divergencias
            divs = score_result.get("divergences", {})
            rsi_div = divs.get("rsi", {})
            macd_div = divs.get("macd", {})

            if rsi_div.get("type") == "bullish" or macd_div.get("type") == "bullish":
                st.markdown(
                    '<div class="divergence-box">\U0001f4a1 DIVERGENCIA BULLISH<br>'
                    'Sinal forte de reversao!</div>',
                    unsafe_allow_html=True,
                )
            elif rsi_div.get("type") == "bearish" or macd_div.get("type") == "bearish":
                st.markdown(
                    '<div class="alert-sell">\u26a0\ufe0f DIVERGENCIA BEARISH<br>'
                    'Risco de reversao para baixa!</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.info("Nenhuma divergencia detectada")

            if dip_info:
                st.info(f"\U0001f4c9 {dip_info['explanation']}")

    # Recomendacao
    if score_result and dip_info:
        rec = generate_recommendation(score_result, dip_info, asset_id.title())
        st.markdown(f'<div class="recommendation-box">{rec}</div>', unsafe_allow_html=True)

    # Grafico principal
    st.subheader("\U0001f4c8 Grafico com Indicadores")
    fig = create_price_chart(df, asset_id.title())
    st.plotly_chart(fig, use_container_width=True)

    # Breakdown dos sinais
    if score_result:
        st.subheader("\U0001f9e0 Detalhamento dos 10 Indicadores")
        signals = score_result.get("signals", {})
        signal_list = list(signals.items())

        # Exibir em grid 2x5
        for row_start in range(0, len(signal_list), 5):
            row_signals = signal_list[row_start:row_start + 5]
            cols = st.columns(len(row_signals))
            for j, (name, info) in enumerate(row_signals):
                with cols[j]:
                    pct = info["points"] / info["max"] if info["max"] > 0 else 0
                    emoji = "\U0001f7e2" if pct >= 0.6 else ("\U0001f7e1" if pct >= 0.3 else "\U0001f534")
                    st.markdown(f"**{emoji} {name}**")
                    st.progress(pct)
                    st.caption(f"{info['points']}/{info['max']} pts")
                    with st.expander("Detalhe"):
                        st.markdown(f"**Valor:** {info.get('value', 'N/A')}")
                        st.markdown(info['signal'])

    # Niveis Fibonacci
    fib = df.attrs.get("fibonacci", {})
    fib_levels = fib.get("levels", {})
    if fib_levels:
        st.subheader("\U0001f4d0 Niveis Fibonacci")
        fib_df = pd.DataFrame([
            {"Nivel": label, "Preco": f"${price:.2f}",
             "Distancia": f"{((current_price - price) / current_price * 100):+.1f}%"}
            for label, price in fib_levels.items()
        ])
        st.dataframe(fib_df, use_container_width=True, hide_index=True)


# -------------------------------------------------------
# PAGINA: Gestao de Risco
# -------------------------------------------------------
def render_risk():
    st.title("\U0001f6e1 Gestao de Risco")

    tab1, tab2, tab3 = st.tabs(["\U0001f4b0 Calculadora de Posicao", "\U0001f4c9 Plano DCA", "\U0001f504 Correlacao"])

    # --- TAB 1: Calculadora de Posicao ---
    with tab1:
        st.subheader("Calculadora de Tamanho de Posicao (ATR)")
        st.info(
            "Esta calculadora usa o ATR (Average True Range) para definir stop-loss automatico "
            "e calcular quanto investir sem arriscar demais do seu portfolio."
        )

        col1, col2 = st.columns(2)
        with col1:
            asset_type_r = st.selectbox("Tipo", ["Criptomoeda", "Acao/ETF"], key="risk_type")
            if asset_type_r == "Criptomoeda":
                asset_id_r = st.selectbox("Ativo", CRYPTO_IDS, format_func=lambda x: x.title(), key="risk_asset")
                a_type_r = "crypto"
            else:
                asset_id_r = st.selectbox("Ativo", STOCK_TICKERS, key="risk_asset")
                a_type_r = "stock"
        with col2:
            portfolio_val = st.number_input("Valor do seu Portfolio (USD)", value=float(DEFAULT_PORTFOLIO_VALUE),
                                           min_value=100.0, step=500.0)
            risk_pct = st.slider("Risco por operacao (%)", min_value=0.5, max_value=10.0,
                                value=DEFAULT_RISK_PER_TRADE, step=0.5,
                                help="Percentual maximo do portfolio que voce aceita perder nesta operacao")

        with st.spinner("Calculando..."):
            df_r, sr_r, _ = get_history_and_analysis(asset_id_r, a_type_r)

        if df_r is not None and not df_r.empty:
            risk = compute_risk_metrics(df_r, portfolio_val, risk_pct)

            if risk:
                st.markdown("### Resultado")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Preco Atual", f"${risk['preco_atual']}")
                c2.metric("ATR (14)", f"${risk['atr']}", f"{risk['atr_percentual']}% volatilidade")
                c3.metric("Stop-Loss Sugerido", f"${risk['stop_loss']}")
                c4.metric("Risco/Retorno", f"{risk['risco_retorno']}x")

                st.markdown("---")
                c1, c2, c3 = st.columns(3)
                c1.metric("Tamanho da Posicao", f"{risk['tamanho_posicao']:.4f} unidades")
                c2.metric("Valor da Posicao", f"${risk['valor_posicao']:.2f}")
                c3.metric("Perda Maxima", f"${risk['risco_maximo']:.2f}",
                          f"{risk_pct}% do portfolio", delta_color="inverse")

                st.markdown("---")
                st.markdown("### Alvos de Lucro (Fibonacci)")
                c1, c2, c3 = st.columns(3)
                c1.metric("Alvo 1 (38.2%)", f"${risk['take_profit_1']}")
                c2.metric("Alvo 2 (23.6%)", f"${risk['take_profit_2']}")
                c3.metric("Alvo 3 (Topo)", f"${risk['take_profit_3']}")

                # Explicacao
                with st.expander("Como interpretar"):
                    st.markdown(f"""
                    **O que significa:**
                    - O ATR de ${risk['atr']:.2f} ({risk['atr_percentual']}%) indica a volatilidade diaria media
                    - O stop-loss em ${risk['stop_loss']:.2f} esta a 2x ATR do preco atual
                    - Se o preco cair ate o stop-loss, voce perde no maximo ${risk['risco_maximo']:.2f} ({risk_pct}% do portfolio)
                    - Para isso, compre no maximo {risk['tamanho_posicao']:.4f} unidades (${risk['valor_posicao']:.2f})

                    **Regra de ouro:** Nunca arrisque mais de 2% do portfolio em uma unica operacao.
                    """)

    # --- TAB 2: Plano DCA ---
    with tab2:
        st.subheader("Plano DCA (Dollar Cost Averaging)")
        st.info(
            "O DCA divide sua compra em parcelas em niveis de preco diferentes. "
            "Se o preco cair, voce compra mais barato automaticamente. Reduz risco de comprar no topo."
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            dca_type = st.selectbox("Tipo", ["Criptomoeda", "Acao/ETF"], key="dca_type")
            if dca_type == "Criptomoeda":
                dca_asset = st.selectbox("Ativo", CRYPTO_IDS, format_func=lambda x: x.title(), key="dca_asset")
                dca_a_type = "crypto"
            else:
                dca_asset = st.selectbox("Ativo", STOCK_TICKERS, key="dca_asset")
                dca_a_type = "stock"
        with col2:
            dca_amount = st.number_input("Valor total a investir (USD)", value=1000.0, min_value=50.0, step=100.0)
        with col3:
            dca_tranches = st.slider("Numero de parcelas", min_value=2, max_value=6, value=4)

        with st.spinner("Gerando plano..."):
            df_dca, _, _ = get_history_and_analysis(dca_asset, dca_a_type)

        if df_dca is not None and not df_dca.empty:
            plan = compute_dca_plan(df_dca, dca_amount, dca_tranches)

            if plan:
                st.markdown(f"### Plano DCA para {dca_asset.title()}")

                plan_df = pd.DataFrame(plan)
                plan_df.columns = ["Parcela", "Preco de Compra", "Valor (USD)", "% do Total", "Nivel"]

                st.dataframe(plan_df, use_container_width=True, hide_index=True)

                # Grafico visual
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[f"Parcela {p['tranche']}" for p in plan],
                    y=[p["valor"] for p in plan],
                    text=[f"${p['valor']:.0f} @ ${p['preco']:.2f}" for p in plan],
                    textposition="outside",
                    marker_color=["#00d4aa", "#3498db", "#f39c12", "#e74c3c", "#9b59b6", "#1abc9c"][:len(plan)],
                ))
                fig.update_layout(
                    height=300, template="plotly_dark", paper_bgcolor="#0e1117",
                    yaxis_title="Valor (USD)", showlegend=False,
                    margin=dict(l=50, r=20, t=20, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("""
                **Estrategia:** Coloque ordens limite nos precos indicados.
                Se o preco cair, suas ordens executam automaticamente comprando mais barato.
                Se o preco subir, voce ja tem a primeira parcela investida.
                """)

    # --- TAB 3: Correlacao ---
    with tab3:
        st.subheader("Matriz de Correlacao")
        st.info(
            "Mostra quais ativos se movem juntos. Correlacao alta (>0.8) = risco duplicado. "
            "Correlacao baixa (<0.3) = boa diversificacao."
        )

        with st.spinner("Calculando correlacoes..."):
            price_data = {}

            # Criptos
            crypto_histories = fetch_all_crypto_histories(CRYPTO_IDS[:5])  # Limitar para velocidade
            for coin_id, df_c in crypto_histories.items():
                if not df_c.empty:
                    price_data[coin_id.title()] = df_c["Close"]

            # Top acoes
            for ticker in STOCK_TICKERS[:6]:
                df_s = fetch_stock_history(ticker)
                if not df_s.empty:
                    price_data[ticker] = df_s["Close"]

        corr_matrix = compute_correlation_matrix(price_data)

        if not corr_matrix.empty:
            fig = px.imshow(
                corr_matrix,
                text_auto=".2f",
                color_continuous_scale="RdYlGn",
                zmin=-1, zmax=1,
                aspect="auto",
            )
            fig.update_layout(
                height=500, template="plotly_dark", paper_bgcolor="#0e1117",
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Alertas de correlacao
            high_corr = []
            low_corr = []
            for i in range(len(corr_matrix)):
                for j in range(i + 1, len(corr_matrix)):
                    val = corr_matrix.iloc[i, j]
                    name_i = corr_matrix.index[i]
                    name_j = corr_matrix.columns[j]
                    if val > 0.8:
                        high_corr.append((name_i, name_j, val))
                    elif val < 0.3:
                        low_corr.append((name_i, name_j, val))

            if high_corr:
                st.warning(
                    "**Correlacao ALTA (risco duplicado):** "
                    + ", ".join([f"{a} + {b} ({v:.2f})" for a, b, v in high_corr])
                )
            if low_corr:
                st.success(
                    "**Boa diversificacao:** "
                    + ", ".join([f"{a} + {b} ({v:.2f})" for a, b, v in low_corr[:5]])
                )
        else:
            st.warning("Dados insuficientes para calcular correlacao.")


# -------------------------------------------------------
# PAGINA: Alertas
# -------------------------------------------------------
def render_alerts():
    st.title("\U0001f514 Central de Alertas")

    scores, all_assets = analyze_all_assets()

    if not scores:
        st.warning("Sem dados disponiveis. Tente novamente em 1 minuto.")
        return

    buy_alerts = []
    sell_alerts = []
    watch_alerts = []
    div_alerts = []

    for s in scores:
        sr = s.get("_score_result", {})
        di = s.get("_dip_info", {})
        score_val = sr.get("score", 0)

        alert = {
            "name": s["Ativo"],
            "score": score_val,
            "label": sr.get("label", ""),
            "trend": sr.get("trend", ""),
            "confluence": sr.get("confluence", {}),
            "explanation": di.get("explanation", "") if di else "",
            "price": s.get("Preco", ""),
            "change_24h": s.get("24h", ""),
        }

        # Verificar divergencias
        divs = sr.get("divergences", {})
        rsi_div = divs.get("rsi", {}).get("type", "none")
        macd_div = divs.get("macd", {}).get("type", "none")
        if rsi_div == "bullish" or macd_div == "bullish":
            alert["divergence"] = "bullish"
            div_alerts.append(alert)
        elif rsi_div == "bearish" or macd_div == "bearish":
            alert["divergence"] = "bearish"

        if score_val >= BUY_CONFIDENCE_STRONG:
            buy_alerts.append(alert)
        elif score_val <= SELL_CONFIDENCE:
            sell_alerts.append(alert)
        elif score_val >= BUY_CONFIDENCE_MODERATE:
            watch_alerts.append(alert)

    # Divergencias (sinal mais forte)
    if div_alerts:
        st.subheader(f"\U0001f4a1 Divergencias Bullish Detectadas ({len(div_alerts)} ativos)")
        for alert in div_alerts:
            conf = alert.get("confluence", {})
            st.markdown(f"""
            <div class="divergence-box">
                <strong>{alert['name']}</strong> - Score: {alert['score']}/100 |
                Confluencia: {conf.get('agree_buy', 0)}/{conf.get('total', 10)}<br>
                Preco: {alert['price']} | 24h: {alert['change_24h']}<br>
                DIVERGENCIA BULLISH - Sinal de reversao para alta!
            </div>
            """, unsafe_allow_html=True)

    # Compra forte
    st.subheader(f"\U0001f7e2 Zona de Compra Forte ({len(buy_alerts)} ativos)")
    if buy_alerts:
        for alert in sorted(buy_alerts, key=lambda x: x["score"], reverse=True):
            conf = alert.get("confluence", {})
            st.markdown(f"""
            <div class="alert-buy">
                <strong>{alert['name']}</strong> - Score: {alert['score']}/100 |
                Confluencia: {conf.get('agree_buy', 0)}/{conf.get('total', 10)}<br>
                Preco: {alert['price']} | 24h: {alert['change_24h']}<br>
                {alert['explanation']}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Nenhum ativo em zona de compra forte no momento.")

    # Venda
    st.subheader(f"\U0001f534 Zona de Alerta/Venda ({len(sell_alerts)} ativos)")
    if sell_alerts:
        for alert in sorted(sell_alerts, key=lambda x: x["score"]):
            st.markdown(f"""
            <div class="alert-sell">
                <strong>{alert['name']}</strong> - Score: {alert['score']}/100<br>
                Preco: {alert['price']} | 24h: {alert['change_24h']}<br>
                {alert['explanation']}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Nenhum ativo em zona de venda no momento.")

    # Observacao
    st.subheader(f"\U0001f7e1 Em Observacao ({len(watch_alerts)} ativos)")
    if watch_alerts:
        for alert in sorted(watch_alerts, key=lambda x: x["score"], reverse=True):
            conf = alert.get("confluence", {})
            st.info(
                f"**{alert['name']}** - Score: {alert['score']} | "
                f"Confluencia: {conf.get('agree_buy', 0)}/{conf.get('total', 10)} | "
                f"{alert['explanation']}"
            )


# -------------------------------------------------------
# PAGINA: Noticias
# -------------------------------------------------------
def render_news():
    st.title("\U0001f4f0 Noticias e Sentimento do Mercado")

    fg_df = fetch_fear_greed()
    if not fg_df.empty:
        st.subheader("Fear & Greed Index - Historico (30 dias)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fg_df["date"], y=fg_df["value"],
            mode="lines+markers",
            line=dict(color="#00d4aa", width=2),
            fill="tozeroy", fillcolor="rgba(0,212,170,0.1)",
        ))
        fig.add_hline(y=20, line_dash="dash", line_color="red", annotation_text="Medo Extremo")
        fig.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="Ganancia Extrema")
        fig.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="Neutro")
        fig.update_layout(
            height=300, template="plotly_dark",
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            yaxis_range=[0, 100], margin=dict(l=50, r=20, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Ultimas Noticias")
    news = fetch_news()

    if news:
        pos = sum(1 for n in news if n["sentiment_label"] == "Positivo")
        neg = sum(1 for n in news if n["sentiment_label"] == "Negativo")
        neu = sum(1 for n in news if n["sentiment_label"] == "Neutro")

        c1, c2, c3 = st.columns(3)
        c1.metric("\U0001f7e2 Positivas", pos)
        c2.metric("\u26aa Neutras", neu)
        c3.metric("\U0001f534 Negativas", neg)

        for article in news:
            icon = {
                "Positivo": "\U0001f7e2", "Negativo": "\U0001f534", "Neutro": "\u26aa",
            }.get(article["sentiment_label"], "\u26aa")
            with st.expander(f"{icon} {article['title'][:100]}"):
                st.markdown(f"**Fonte:** {article['source']}")
                st.markdown(f"**Sentimento:** {article['sentiment_label']}")
                st.markdown(f"**Data:** {article['date']}")
                if article["url"]:
                    st.markdown(f"[Ler noticia completa]({article['url']})")
    else:
        st.info("Nao foi possivel carregar noticias.")


# -------------------------------------------------------
# PAGINA: Portfolio
# -------------------------------------------------------
def render_portfolio():
    st.title("\U0001f4bc Meu Portfolio")
    st.info("Adicione seus ativos para acompanhar valor total, P&L e alocacao.")

    if "portfolio" not in st.session_state:
        st.session_state.portfolio = {}

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        p_type = st.selectbox("Tipo", ["crypto", "stock"], key="p_type",
                              format_func=lambda x: "Cripto" if x == "crypto" else "Acao")
    with col2:
        if p_type == "crypto":
            p_asset = st.selectbox("Ativo", CRYPTO_IDS, key="p_asset")
        else:
            p_asset = st.selectbox("Ativo", STOCK_TICKERS, key="p_asset")
    with col3:
        p_qty = st.number_input("Quantidade", min_value=0.0, step=0.01, key="p_qty")
    with col4:
        p_buy = st.number_input("Preco de Compra ($)", min_value=0.0, step=0.01, key="p_buy")

    if st.button("Adicionar"):
        if p_qty > 0:
            st.session_state.portfolio[p_asset] = {"type": p_type, "quantity": p_qty, "buy_price": p_buy}
            st.success(f"{p_asset} adicionado!")
            st.rerun()

    if not st.session_state.portfolio:
        st.info("Portfolio vazio.")
        return

    portfolio_data = []
    total_invested = 0
    total_current = 0

    for asset_id, info in st.session_state.portfolio.items():
        if info["type"] == "crypto":
            cdf = fetch_crypto_current([asset_id])
            price = cdf.iloc[0]["price"] if not cdf.empty else 0
        else:
            sdf = fetch_stock_history(asset_id, period="5d")
            price = sdf["Close"].iloc[-1] if not sdf.empty else 0

        current_value = price * info["quantity"]
        invested = info["buy_price"] * info["quantity"]
        pnl = current_value - invested
        pnl_pct = ((price - info["buy_price"]) / info["buy_price"] * 100) if info["buy_price"] > 0 else 0

        total_invested += invested
        total_current += current_value
        portfolio_data.append({
            "Ativo": asset_id, "Qtd": info["quantity"],
            "Compra": f"${info['buy_price']:.2f}", "Atual": f"${price:.2f}",
            "Valor": f"${current_value:.2f}", "P&L": f"${pnl:.2f}", "P&L%": f"{pnl_pct:+.2f}%",
        })

    total_pnl = total_current - total_invested
    total_pnl_pct = ((total_current - total_invested) / total_invested * 100) if total_invested > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Investido", f"${total_invested:.2f}")
    c2.metric("Atual", f"${total_current:.2f}")
    c3.metric("P&L", f"${total_pnl:.2f}", f"{total_pnl_pct:+.2f}%")
    c4.metric("Ativos", len(st.session_state.portfolio))

    st.dataframe(pd.DataFrame(portfolio_data), use_container_width=True, hide_index=True)

    if portfolio_data:
        fig = go.Figure(data=[go.Pie(
            labels=[p["Ativo"] for p in portfolio_data],
            values=[float(p["Valor"].replace("$", "")) for p in portfolio_data],
            hole=0.4,
            marker_colors=["#00d4aa", "#3498db", "#f39c12", "#e74c3c", "#9b59b6",
                          "#1abc9c", "#e67e22", "#2ecc71", "#f1c40f", "#34495e"],
        )])
        fig.update_layout(title="Alocacao", template="plotly_dark", paper_bgcolor="#0e1117", height=350)
        st.plotly_chart(fig, use_container_width=True)

    if st.button("\U0001f5d1 Limpar Portfolio"):
        st.session_state.portfolio = {}
        st.rerun()


# -------------------------------------------------------
# Roteamento
# -------------------------------------------------------
if page == "\U0001f3e0 Visao Geral":
    render_overview()
elif page == "\U0001f50d Analise Detalhada":
    render_deep_dive()
elif page == "\U0001f6e1 Gestao de Risco":
    render_risk()
elif page == "\U0001f514 Alertas":
    render_alerts()
elif page == "\U0001f4f0 Noticias":
    render_news()
elif page == "\U0001f4bc Portfolio":
    render_portfolio()
