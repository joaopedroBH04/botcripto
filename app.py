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
from datetime import datetime

from config import (
    CRYPTO_IDS, STOCK_TICKERS,
    BUY_CONFIDENCE_STRONG, BUY_CONFIDENCE_MODERATE, SELL_CONFIDENCE,
    FIBONACCI_LEVELS, FIBONACCI_LABELS,
    DEFAULT_PORTFOLIO_VALUE, DEFAULT_RISK_PER_TRADE,
    score_label,
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
# CSS customizado — Design System BotCripto
# -------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ─── CRITICO: aplicar Inter SEM sobrescrever Material Icons do Streamlit ─── */
/* O !important no * quebra as ligaduras de icone, causando "_arrow_right" como texto */
body, .stApp { font-family: 'Inter', sans-serif; }
p, span:not([class*="material"]), div:not([class*="material"]), h1, h2, h3, h4, h5, h6,
button, label, input, select, textarea, a, td, th, li,
[data-testid="stSidebar"] *, [data-testid="stMainBlockContainer"] *,
[data-baseweb], .stMarkdown, .stText {
    font-family: 'Inter', sans-serif;
}
/* Restaurar Material Icons (necessario para expanders, botoes, tabs) */
.material-icons, .material-icons-round, .material-icons-outlined,
[class*="material-icons"] {
    font-family: 'Material Icons', 'Material Icons Round', 'Material Icons Outlined' !important;
    font-feature-settings: 'liga' !important;
    -webkit-font-feature-settings: 'liga' !important;
    text-rendering: optimizeLegibility !important;
}

/* ─── Background ─── */
.stApp {
    background:
        radial-gradient(ellipse at 15% 0%, rgba(99,102,241,0.07) 0%, transparent 45%),
        radial-gradient(ellipse at 85% 95%, rgba(56,189,248,0.05) 0%, transparent 45%),
        #07080F;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A0C1C 0%, #07080F 100%) !important;
    border-right: 1px solid #181C30 !important;
}
section[data-testid="stSidebar"] > div { background: transparent !important; }

/* ─── Ocultar branding Streamlit ─── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* ─── Cards ─── */
.card {
    background: linear-gradient(145deg, #0D0F1E, #0A0C18);
    border: 1px solid #1C2038;
    border-radius: 14px;
    padding: 20px 24px;
    margin: 8px 0;
    transition: border-color 0.25s, box-shadow 0.25s;
}
.card:hover {
    border-color: rgba(99, 102, 241, 0.3);
    box-shadow: 0 4px 32px rgba(99, 102, 241, 0.1);
}

/* ─── Caixas de Alerta ─── */
.alert-buy {
    background: rgba(0, 229, 195, 0.05);
    border: 1px solid rgba(0, 229, 195, 0.16);
    border-left: 3px solid #00E5C3;
    padding: 16px 20px;
    border-radius: 10px;
    margin: 8px 0;
    color: #C8D8E8;
    font-size: 0.88rem;
    line-height: 1.65;
}
.alert-sell {
    background: rgba(255, 71, 87, 0.05);
    border: 1px solid rgba(255, 71, 87, 0.16);
    border-left: 3px solid #FF4757;
    padding: 16px 20px;
    border-radius: 10px;
    margin: 8px 0;
    color: #C8D8E8;
    font-size: 0.88rem;
    line-height: 1.65;
}
.divergence-box {
    background: rgba(74, 158, 255, 0.05);
    border: 1px solid rgba(74, 158, 255, 0.25);
    border-left: 3px solid #4A9EFF;
    padding: 14px 20px;
    border-radius: 10px;
    margin: 10px 0;
    font-weight: 600;
    color: #4A9EFF;
    font-size: 0.88rem;
    line-height: 1.65;
}

/* ─── Caixa de Recomendacao ─── */
.recommendation-box {
    background: linear-gradient(140deg, #0D0F1E 0%, #0A0C18 100%);
    border: 1px solid #1C2038;
    border-top: 2px solid #6366F1;
    padding: 18px 22px;
    border-radius: 12px;
    margin: 12px 0;
    line-height: 1.75;
    font-size: 0.9rem;
    color: #8B9AB0;
}

/* ─── Caixa de Confluencia ─── */
.confluence-box {
    background: #0F1923;
    border: 1px solid #1A2A40;
    border-radius: 12px;
    padding: 22px 16px;
    text-align: center;
}
.confluence-box h3 {
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0 0 6px 0;
    background: linear-gradient(135deg, #818CF8, #38BDF8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.confluence-box p {
    color: #6A8898;
    font-size: 0.78rem;
    margin: 4px 0 0 0;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ─── Risk Metric ─── */
.risk-metric {
    background: #0F1923;
    border: 1px solid #1A2A40;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 4px 0;
}

/* ─── Badges de Status ─── */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}
.badge-buy    { background: rgba(0,229,195,0.12); color: #00E5C3; border: 1px solid rgba(0,229,195,0.28); }
.badge-watch  { background: rgba(255,184,0,0.12);  color: #FFB800; border: 1px solid rgba(255,184,0,0.28); }
.badge-sell   { background: rgba(255,71,87,0.12);  color: #FF4757; border: 1px solid rgba(255,71,87,0.28); }
.badge-neutral{ background: rgba(139,154,176,0.10); color: #8B9AB0; border: 1px solid rgba(139,154,176,0.25); }

/* ─── Dots de Status ─── */
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }
.dot-green  { background: #00E5C3; box-shadow: 0 0 6px rgba(0,229,195,0.7); }
.dot-yellow { background: #FFB800; box-shadow: 0 0 6px rgba(255,184,0,0.7); }
.dot-red    { background: #FF4757; box-shadow: 0 0 6px rgba(255,71,87,0.7); }
.dot-gray   { background: #2A3A50; }

/* ─── Metrics ─── */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #0D0F1E, #0A0C18) !important;
    border: 1px solid #1C2038 !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
}
[data-testid="stMetricLabel"] > div {
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #7A9AB0 !important;
}
[data-testid="stMetricValue"] { color: #E8EDF5 !important; font-weight: 700 !important; }

/* ─── Botoes ─── */
.stButton > button {
    background: linear-gradient(135deg, #6366F1, #4F46E5) !important;
    color: #fff !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.3px;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 22px rgba(99, 102, 241, 0.35) !important;
}

/* ─── Fix: esconder botao de colapso (keyboard_double bug) ─── */
[data-testid="stSidebarCollapseButton"] { display: none !important; }

/* ─── Navegacao customizada no sidebar ─── */
div[data-testid="stSidebar"] .stRadio > div { gap: 2px !important; margin-top: 4px !important; }

/* Item de navegacao — base */
div[data-testid="stSidebar"] .stRadio > div > label {
    display: flex !important;
    align-items: center !important;
    padding: 9px 12px 9px 14px !important;
    border-radius: 8px !important;
    border-left: 2px solid transparent !important;
    cursor: pointer !important;
    transition: background 0.15s, border-color 0.15s, color 0.15s !important;
    font-size: 0.855rem !important;
    font-weight: 400 !important;
    color: #506070 !important;
    letter-spacing: 0.3px !important;
    margin: 1px 0 !important;
    white-space: nowrap !important;
}
/* Hover */
div[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: rgba(99,102,241,0.06) !important;
    border-left-color: rgba(99,102,241,0.4) !important;
    color: #A0B5C5 !important;
}
/* Ativo — detecta o radio checked via :has() */
div[data-testid="stSidebar"] .stRadio > div > label:has(input:checked) {
    background: rgba(99,102,241,0.1) !important;
    border-left-color: #818CF8 !important;
    color: #E8EDF5 !important;
    font-weight: 600 !important;
}
/* Esconder o circulo do radio nativo */
div[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] > div:first-child { display: none !important; }
div[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] { gap: 0 !important; }

/* Botao de atualizar no sidebar — estilo diferente do botao principal */
div[data-testid="stSidebar"] .stButton > button {
    background: rgba(99,102,241,0.08) !important;
    color: #818CF8 !important;
    border: 1px solid rgba(99,102,241,0.22) !important;
    font-weight: 500 !important;
    font-size: 0.83rem !important;
    letter-spacing: 0.3px !important;
    width: 100% !important;
    box-shadow: none !important;
    transform: none !important;
    border-radius: 8px !important;
}
div[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(99,102,241,0.14) !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ─── Abas ─── */
[data-baseweb="tab-list"] {
    background: #0A0C1A !important;
    border-radius: 10px !important;
    gap: 6px !important;
    padding: 6px !important;
    border: 1px solid #1C2038 !important;
}
[data-baseweb="tab"] {
    border-radius: 7px !important;
    font-size: 0.84rem !important;
    padding: 8px 20px !important;
    font-weight: 500 !important;
    letter-spacing: 0.2px !important;
    color: #506070 !important;
    transition: all 0.15s !important;
}
[data-baseweb="tab"][aria-selected="true"] {
    background: rgba(99,102,241,0.12) !important;
    color: #818CF8 !important;
    font-weight: 600 !important;
}

/* ─── Animacoes ─── */
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 0 0 rgba(74,158,255,0.35); }
    50%       { box-shadow: 0 0 0 6px rgba(74,158,255,0); }
}
@keyframes live-blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
}
@keyframes slide-in {
    from { opacity: 0; transform: translateY(-4px); }
    to   { opacity: 1; transform: translateY(0); }
}
.divergence-box { animation: pulse-glow 2.5s infinite, slide-in 0.3s ease; }
.alert-buy      { animation: slide-in 0.25s ease; }
.alert-sell     { animation: slide-in 0.25s ease; }

/* ─── Expander ─── */
[data-testid="stExpander"] { border: 1px solid #1C2038 !important; border-radius: 10px !important; }

/* ─── DataFrames ─── */
[data-testid="stDataFrame"] { border: 1px solid #1C2038 !important; border-radius: 10px !important; overflow: hidden; }

/* ─── Scrollbar ─── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #060B14; }
::-webkit-scrollbar-thumb { background: #1A2A40; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2A3A50; }

/* ─── Divisor ─── */
hr { border-color: #1A2A40 !important; margin: 20px 0 !important; }

/* ─── Label de secao ─── */
.section-label {
    font-size: 0.67rem;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    background: linear-gradient(90deg, #818CF8, #38BDF8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 28px 0 12px 0;
    display: block;
    padding-bottom: 8px;
    border-bottom: 1px solid #161C30;
}

/* ─── Header de pagina com barra lateral ─── */
.page-header {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    margin-bottom: 28px;
    animation: slide-in 0.3s ease;
}
.page-header-bar {
    width: 3px;
    height: 36px;
    background: linear-gradient(180deg, #818CF8, #38BDF8);
    border-radius: 2px;
    flex-shrink: 0;
    margin-top: 3px;
}
.page-header h2 {
    font-size: 1.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #E8EDF5 0%, #A5B4FC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
    margin: 0 0 4px 0;
    line-height: 1.2;
}
.page-header p {
    font-size: 0.8rem;
    color: #5A7888;
    margin: 0;
    letter-spacing: 0.3px;
}

/* ─── Alerts nativos Streamlit ─── */
[data-testid="stAlert"] { border-radius: 10px !important; border-width: 1px !important; }

/* ══════════════════════════════════════════
   VISUAIS INOVADORES
   ══════════════════════════════════════════ */

/* ─── Ticker horizontal animado ─── */
@keyframes ticker-scroll {
    0%   { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}
.ticker-wrap {
    background: #060B14;
    border-top: 1px solid #0F1E2E;
    border-bottom: 1px solid #0F1E2E;
    overflow: hidden;
    padding: 9px 0;
    margin: 16px 0 20px 0;
    position: relative;
}
.ticker-wrap::before, .ticker-wrap::after {
    content: '';
    position: absolute;
    top: 0; bottom: 0;
    width: 60px;
    z-index: 2;
}
.ticker-wrap::before { left:0; background: linear-gradient(90deg,#060B14,transparent); }
.ticker-wrap::after  { right:0; background: linear-gradient(90deg,transparent,#060B14); }
.ticker-inner {
    display: flex;
    width: max-content;
    animation: ticker-scroll 55s linear infinite;
}
.ticker-inner:hover { animation-play-state: paused; }

/* ─── Anel SVG de score ─── */
@keyframes ring-draw {
    from { stroke-dashoffset: 340; }
    to   { stroke-dashoffset: var(--ring-offset, 0); }
}
@keyframes ring-fade {
    from { opacity: 0; transform: scale(0.9); }
    to   { opacity: 1; transform: scale(1); }
}
.score-ring-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 14px 0 8px 0;
    animation: ring-fade 0.5s ease forwards;
}

/* ─── Heatmap grid de ativos ─── */
.heat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(172px, 1fr));
    gap: 10px;
    margin: 14px 0 24px 0;
}
.heat-tile {
    background: linear-gradient(145deg, #0D0F1E 0%, #0A0C18 100%);
    border: 1px solid #1C2038;
    border-radius: 12px;
    padding: 14px 15px;
    transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
    position: relative;
    overflow: hidden;
}
.heat-tile::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--tile-accent, #1A2A40);
    border-radius: 10px 10px 0 0;
}
.heat-tile:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 32px rgba(0,0,0,0.55);
    border-color: var(--tile-border, #2A3A50);
}
.heat-tile.t-buy  { --tile-accent: #00E5C3; --tile-border: rgba(0,229,195,0.3); }
.heat-tile.t-watch{ --tile-accent: #FFB800; --tile-border: rgba(255,184,0,0.25); }
.heat-tile.t-sell { --tile-accent: #FF4757; --tile-border: rgba(255,71,87,0.25); }
.heat-tile.t-neut { --tile-accent: #2A3A50; --tile-border: #2A3A50; }

/* ─── Glass metric cards ─── */
.g-metric {
    background: linear-gradient(145deg, #0D0F1E 0%, #0A0C18 100%);
    border: 1px solid #1C2038;
    border-radius: 14px;
    padding: 20px 22px 18px 22px;
    position: relative;
    overflow: hidden;
    transition: box-shadow 0.2s, border-color 0.2s;
}
.g-metric:hover {
    border-color: rgba(99,102,241,0.25);
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
}
.g-metric::after {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 20% 0%, var(--gm-glow, rgba(99,102,241,0.06)) 0%, transparent 70%);
    pointer-events: none;
}
.g-metric .gm-label {
    font-size: 0.64rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #6A90A8;
    margin-bottom: 10px;
    display: block;
}
.g-metric .gm-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--gm-color, #E8EDF5);
    line-height: 1;
    margin-bottom: 6px;
    text-shadow: 0 0 28px var(--gm-glow, transparent);
}
.g-metric .gm-sub {
    font-size: 0.75rem;
    color: #607888;
}
.g-metric .gm-accent-bar {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--gm-color, #1A2A40), transparent 60%);
}

/* ─── Numero de destaque com glow ─── */
.glow-number {
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -2px;
    line-height: 1;
}

/* ─── Confluencia visual ─── */
.conf-arc-wrap {
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    gap: 10px;
    padding: 16px;
    background: #0C1522;
    border: 1px solid #1A2A40;
    border-radius: 12px;
}

/* ─── Barra de score inline ─── */
.inline-score-bar {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 4px 0;
}
.inline-score-bar .bar-track {
    flex: 1;
    background: #0A1520;
    border-radius: 4px;
    height: 5px;
    overflow: hidden;
}
.inline-score-bar .bar-fill {
    height: 100%;
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------
# Registra hora da ultima renderizacao (nao do cache)
if "last_render_time" not in st.session_state:
    st.session_state["last_render_time"] = datetime.now()

_now = datetime.now()
_last = st.session_state.get("last_render_time", _now)
_mins_ago = int((_now - _last).total_seconds() / 60)
_age_label = "agora" if _mins_ago == 0 else f"ha {_mins_ago} min"

st.sidebar.markdown(f"""
<div style="padding:20px 4px 18px 4px; border-bottom:1px solid #0F1E2E; margin-bottom:16px;">
    <div style="font-size:1.6rem; font-weight:700;
                background:linear-gradient(135deg,#818CF8 0%,#38BDF8 100%);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                background-clip:text; letter-spacing:-0.8px; line-height:1;">BotCripto</div>
    <div style="font-size:0.6rem; color:#5A7888; text-transform:uppercase;
                letter-spacing:2.2px; margin-top:5px; margin-bottom:10px;">Monitor Financeiro v2</div>
    <div style="display:flex;align-items:center;gap:7px;margin-bottom:4px;">
        <div style="width:6px;height:6px;background:#818CF8;border-radius:50%;
                    flex-shrink:0;
                    animation:live-blink 1.8s ease-in-out infinite;
                    box-shadow:0 0 6px rgba(129,140,248,0.8);"></div>
        <span style="font-size:0.62rem;color:#5A7888;text-transform:uppercase;
                     letter-spacing:1.5px;">Cache 10 min &bull; Cripto</span>
    </div>
    <div style="font-size:0.6rem;color:#3A5568;padding-left:13px;">
        Pagina carregada {_age_label} &nbsp;&bull;&nbsp; {_now.strftime('%H:%M')}
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="font-size:0.62rem;color:#4A6275;text-transform:uppercase;
            letter-spacing:2px;margin-bottom:6px;padding:0 4px;">Menu</div>
""", unsafe_allow_html=True)

_nav_options = {
    "▸  01   Visao Geral":       "Visao Geral",
    "▸  02   Analise Detalhada": "Analise Detalhada",
    "▸  03   Gestao de Risco":   "Gestao de Risco",
    "▸  04   Alertas":           "Alertas",
    "▸  05   Noticias":          "Noticias",
    "▸  06   Portfolio":         "Portfolio",
    "▸  07   Backtesting":       "Backtesting",
}
_nav_descs = {
    "Visao Geral":       "Panorama de todos os ativos",
    "Analise Detalhada": "10 indicadores tecnicos",
    "Gestao de Risco":   "Posicao, DCA e correlacao",
    "Alertas":           "Sinais ativos agora",
    "Noticias":          "Sentimento do mercado",
    "Portfolio":         "Seus ativos e P&L",
    "Backtesting":       "Simule estrategias historicas",
}
_selected_nav = st.sidebar.radio(
    "nav",
    list(_nav_options.keys()),
    label_visibility="collapsed",
)
page = _nav_options[_selected_nav]

# Descricao da pagina ativa
st.sidebar.markdown(
    f'<div style="font-size:0.75rem;color:#5A7888;padding:4px 14px 12px 14px;'
    f'border-bottom:1px solid #0F1E2E;">{_nav_descs[page]}</div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")
if st.sidebar.button("Atualizar Dados", use_container_width=True):
    st.session_state["last_render_time"] = datetime.now()
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("""
<div style="margin-top:24px;padding:14px;background:#090F1C;border-radius:8px;
            border:1px solid #0F1E2E;">
    <div style="font-size:0.65rem;color:#5A7888;text-transform:uppercase;
                letter-spacing:1.2px;margin-bottom:6px;">Aviso Legal</div>
    <div style="font-size:0.72rem;color:#4A6275;line-height:1.6;">
        Este sistema e apenas informativo e nao constitui recomendacao de investimento.
        Sempre faca sua propria analise antes de operar.
    </div>
</div>
<div style="margin-top:14px;font-size:0.62rem;color:#3A5060;text-align:center;
            letter-spacing:0.5px;">BotCripto v2 &nbsp;·&nbsp; open source</div>
""", unsafe_allow_html=True)


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


@st.cache_data(ttl=600, show_spinner=False)
def get_history_and_analysis(asset_id: str, asset_type: str):
    """
    Busca historico + indicadores + score de UM ativo.

    Resultado cacheado por 10 min: enquanto o usuario interagir com
    widgets da mesma pagina (filtros, dropdowns), essa funcao nao
    recalcula todos os indicadores nem refaz a chamada a API.
    """
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
    """Retorna texto simples para uso em dataframes (sem HTML)."""
    label, _ = score_label(score)
    return label


def score_badge(score: int) -> str:
    """Retorna badge HTML para uso em markdown/HTML."""
    _, css = score_label(score)
    labels = {"buy": "Compra Forte", "watch": "Observacao", "neutral": "Neutro", "sell": "Venda"}
    return f'<span class="badge badge-{css}">{labels[css]}</span>'


def score_dot(score: int) -> str:
    """Retorna dot HTML colorido para indicadores."""
    _, css = score_label(score)
    dot_classes = {"buy": "dot-green", "watch": "dot-yellow", "neutral": "dot-gray", "sell": "dot-red"}
    return f'<span class="dot {dot_classes[css]}"></span>'


def format_number(n):
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "N/A"
    if abs(n) >= 1e12:
        return f"${n/1e12:,.2f}T"
    if abs(n) >= 1e9:
        return f"${n/1e9:,.2f}B"
    if abs(n) >= 1e6:
        return f"${n/1e6:,.2f}M"
    return f"${n:,.2f}"


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

@st.cache_data(ttl=600, show_spinner=False)
def _compute_all_scores(all_assets: pd.DataFrame, fg_val: int) -> list[dict]:
    """
    Nucleo puro e cacheavel de analise em lote.

    Recebe a lista de ativos + valor atual do Fear&Greed e devolve uma
    lista de dicts com scores + DataFrames anexados. Como esta funcao nao
    depende de widgets Streamlit, pode ser cacheada por 10 min — o ganho
    e substancial porque render_overview() e render_alerts() a chamam.
    """
    if all_assets.empty:
        return []

    crypto_assets = all_assets[all_assets["type"] == "crypto"]
    stock_assets = all_assets[all_assets["type"] == "stock"]

    crypto_ids = crypto_assets["id"].tolist() if not crypto_assets.empty else []
    crypto_histories = fetch_all_crypto_histories(crypto_ids) if crypto_ids else {}

    scores = []

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
                "Tipo": "Cripto",
                "Preco": f"${row['price']:.4f}" if row["price"] < 1 else format_number(row["price"]),
                "24h": f"{row['change_24h']:+.2f}%",
                "Score": score_result["score"],
                "Sinal": score_result['label'],
                "Confluencia": f"{confluence.get('agree_buy', 0)}/{confluence.get('total', 10)}",
                "Tendencia": score_result["trend"].replace("_", " ").title(),
                "id": row["id"],
                "type": "crypto",
                "_score_result": score_result,
                "_dip_info": dip_info,
                "_df": df,
            })

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
                "Tipo": "Acao",
                "Preco": format_number(row["price"]),
                "24h": f"{row['change_24h']:+.2f}%",
                "Score": score_result["score"],
                "Sinal": score_result['label'],
                "Confluencia": f"{confluence.get('agree_buy', 0)}/{confluence.get('total', 10)}",
                "Tendencia": score_result["trend"].replace("_", " ").title(),
                "id": row["id"],
                "type": "stock",
                "_score_result": score_result,
                "_dip_info": dip_info,
                "_df": df,
            })

    return scores


def analyze_all_assets():
    """
    Wrapper com UI (spinner) ao redor do nucleo cacheavel _compute_all_scores.

    A separacao permite que o trabalho pesado (fetch + indicadores + scoring)
    seja cacheado via @st.cache_data, enquanto mantem o feedback visual
    para o usuario na primeira execucao.
    """
    all_assets = get_all_assets()
    if all_assets.empty:
        return [], all_assets

    fg_val, _ = get_fear_greed_current()

    with st.spinner("Analisando ativos... (respeitando limites da API)"):
        scores = _compute_all_scores(all_assets, fg_val)

    # Persistencia opcional: salva snapshot dos scores do dia
    try:
        from database import save_scores_snapshot
        save_scores_snapshot(scores)
    except Exception:
        pass  # Persistencia e opcional, nao pode quebrar a UI

    # Dispara notificacoes de sinais fortes (sem bloquear a UI)
    try:
        from notifications import dispatch_strong_signals
        dispatch_strong_signals(scores)
    except Exception:
        pass

    return scores, all_assets


# -------------------------------------------------------
# COMPONENTES VISUAIS INOVADORES
# -------------------------------------------------------

def _score_colors(score: int):
    if score >= 72:
        return "#00E5C3", "rgba(0,229,195,0.45)", "rgba(0,229,195,0.07)", "t-buy"
    elif score >= 55:
        return "#4A9EFF", "rgba(74,158,255,0.45)", "rgba(74,158,255,0.07)", "t-watch"
    elif score >= 30:
        return "#FFB800", "rgba(255,184,0,0.4)",  "rgba(255,184,0,0.06)",  "t-neut"
    return "#FF4757", "rgba(255,71,87,0.45)", "rgba(255,71,87,0.07)", "t-sell"


def render_score_ring_html(score: int) -> str:
    """Anel SVG animado que substitui o gauge plotly."""
    radius = 52
    circ   = 326.7   # 2π × 52
    offset = circ * (1 - score / 100)
    color, glow, _, _ = _score_colors(score)
    label = score_emoji(score)

    return f"""
<div class="score-ring-wrap">
    <div style="position:relative;width:148px;height:148px;">
        <svg width="148" height="148" viewBox="0 0 148 148"
             style="transform:rotate(-90deg);overflow:visible;">
            <circle cx="74" cy="74" r="{radius}" fill="none"
                    stroke="#0A1520" stroke-width="13"/>
            <circle cx="74" cy="74" r="{radius}" fill="none"
                    stroke="{color}" stroke-width="13"
                    stroke-linecap="round"
                    stroke-dasharray="{circ:.1f}"
                    style="--ring-offset:{offset:.1f};
                           stroke-dashoffset:{offset:.1f};
                           filter:drop-shadow(0 0 10px {glow});
                           animation:ring-draw 1.4s cubic-bezier(0.22,1,0.36,1) forwards;"/>
        </svg>
        <div style="position:absolute;inset:0;display:flex;flex-direction:column;
                    align-items:center;justify-content:center;gap:2px;">
            <div style="font-size:2.5rem;font-weight:800;color:{color};line-height:1;
                        text-shadow:0 0 30px {glow};letter-spacing:-2px;">{score}</div>
            <div style="font-size:0.56rem;color:#5A7888;text-transform:uppercase;
                        letter-spacing:2px;">/ 100</div>
        </div>
    </div>
    <div style="margin-top:8px;font-size:0.68rem;font-weight:600;color:{color};
                text-transform:uppercase;letter-spacing:1.8px;
                background:{'rgba(0,229,195,0.08)' if score>=72 else 'rgba(74,158,255,0.08)' if score>=55 else 'rgba(255,184,0,0.08)' if score>=30 else 'rgba(255,71,87,0.08)'};
                padding:4px 12px;border-radius:20px;">Score de Compra</div>
</div>
"""


def render_ticker_html(scores: list) -> str:
    """Banner de ticker horizontal com variacao 24h de todos os ativos."""
    if not scores:
        return ""
    items = ""
    for s in scores:
        change = s.get("24h", "0%")
        is_pos = change.startswith("+")
        color  = "#00E5C3" if is_pos else "#FF4757"
        name   = s["Ativo"].split("(")[0].strip()[:14]
        symbol = "▲" if is_pos else "▼"
        items += f"""<span style="display:inline-flex;align-items:center;gap:8px;
                                   margin:0 22px;white-space:nowrap;">
            <span style="font-size:0.72rem;color:#5A7888;font-family:'Courier New',monospace;
                         letter-spacing:0.5px;">{name}</span>
            <span style="font-size:0.75rem;font-weight:700;color:{color};
                         font-family:'Courier New',monospace;">{symbol} {change}</span>
        </span>
        <span style="color:#0F1E2E;font-size:0.6rem;">·</span>"""
    doubled = items * 2  # duplicar para loop seamless
    return f"""
<div class="ticker-wrap">
    <div class="ticker-inner">{doubled}</div>
</div>"""


def render_heatmap_html(scores_df) -> str:
    """Grade de tiles coloridos por score — substitui o dataframe tabular."""
    tiles = ""
    for _, row in scores_df.iterrows():
        sv     = row["Score"]
        change = row["24h"]
        is_pos = change.startswith("+")
        color, glow, bg, tile_cls = _score_colors(sv)
        ch_color = "#00E5C3" if is_pos else "#FF4757"
        ch_bg    = "rgba(0,229,195,0.08)" if is_pos else "rgba(255,71,87,0.08)"
        ch_sym   = "▲" if is_pos else "▼"
        name     = row["Ativo"]
        pct      = sv / 100
        conf     = row["Confluencia"]

        tiles += f"""
<div class="heat-tile {tile_cls}">
    <div style="margin-bottom:9px;">
        <div style="font-size:0.78rem;font-weight:600;color:#C0CDD8;
                    line-height:1.3;margin-bottom:3px;">{name}</div>
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="font-size:0.68rem;color:#5A7888;">{row['Tipo']} &middot; {row['Preco']}</span>
            <span style="font-size:0.7rem;font-weight:700;color:{ch_color};
                         background:{ch_bg};padding:1px 6px;border-radius:5px;">
                {ch_sym} {change}</span>
        </div>
    </div>
    <div style="background:#0A1520;border-radius:3px;height:3px;margin-bottom:10px;overflow:hidden;">
        <div style="background:{color};width:{pct*100:.0f}%;height:100%;border-radius:3px;
                    box-shadow:0 0 8px {glow};"></div>
    </div>
    <div style="display:flex;justify-content:space-between;align-items:baseline;">
        <span style="font-size:1.4rem;font-weight:800;color:{color};
                     text-shadow:0 0 16px {glow};letter-spacing:-1px;">{sv}</span>
        <span style="font-size:0.64rem;color:#5A7888;">{conf} indic.</span>
    </div>
</div>"""
    return f'<div class="heat-grid">{tiles}</div>'


def render_glass_metric(label: str, value: str, sub: str,
                         color: str = "#E8EDF5",
                         glow: str  = "rgba(0,229,195,0.06)") -> str:
    """Card de metrica com efeito glassmorphism e glow."""
    return f"""
<div class="g-metric" style="--gm-color:{color};--gm-glow:{glow};">
    <span class="gm-label">{label}</span>
    <div class="gm-value">{value}</div>
    <div class="gm-sub">{sub}</div>
    <div class="gm-accent-bar"></div>
</div>"""


# -------------------------------------------------------
# PAGINA: Visao Geral
# -------------------------------------------------------
def render_overview():
    st.markdown("""
<div class="page-header">
    <div class="page-header-bar"></div>
    <div><h2>Visao Geral do Mercado</h2>
    <p>Panorama em tempo real &mdash; criptomoedas, acoes e sentimento</p></div>
</div>""", unsafe_allow_html=True)

    # ── Pulse do mercado — glass metrics ──
    fg_val, fg_class = get_fear_greed_current()
    btc_dom = fetch_btc_dominance()
    total_assets = len(CRYPTO_IDS) + len(STOCK_TICKERS)

    if fg_val <= 25:
        fg_color, fg_glow = "#FF4757", "rgba(255,71,87,0.3)"
        fg_sub = "Medo Extremo"
    elif fg_val <= 45:
        fg_color, fg_glow = "#FFB800", "rgba(255,184,0,0.28)"
        fg_sub = "Medo"
    elif fg_val <= 55:
        fg_color, fg_glow = "#8B9AB0", "rgba(139,154,176,0.2)"
        fg_sub = "Neutro"
    elif fg_val <= 75:
        fg_color, fg_glow = "#00E5C3", "rgba(0,229,195,0.28)"
        fg_sub = "Ganancia"
    else:
        fg_color, fg_glow = "#FFB800", "rgba(255,184,0,0.3)"
        fg_sub = "Ganancia Extrema"

    btc_color = "#00E5C3" if btc_dom >= 50 else "#4A9EFF"
    btc_glow  = "rgba(0,229,195,0.25)" if btc_dom >= 50 else "rgba(74,158,255,0.25)"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(render_glass_metric(
            "Fear &amp; Greed Index",
            f"{fg_val} <span style='font-size:1rem;font-weight:400;color:#5A7888;'>{fg_class}</span>",
            fg_sub, fg_color, fg_glow
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(render_glass_metric(
            "Dominancia BTC",
            f"{btc_dom:.1f}<span style='font-size:1rem;font-weight:400;color:#5A7888;'>%</span>",
            "Dominancia do Bitcoin no mercado cripto",
            btc_color, btc_glow
        ), unsafe_allow_html=True)
    with c3:
        st.markdown(render_glass_metric(
            "Ativos Monitorados",
            str(total_assets),
            f"{len(CRYPTO_IDS)} criptos &middot; {len(STOCK_TICKERS)} acoes/ETFs",
            "#4A9EFF", "rgba(74,158,255,0.22)"
        ), unsafe_allow_html=True)

    scores, all_assets = analyze_all_assets()

    scores_df = pd.DataFrame(scores)
    if scores_df.empty:
        st.error(
            "**Nao foi possivel calcular scores dos ativos.**\n\n"
            "Possiveis causas:\n"
            "- Limite de requisicoes da API CoinGecko (max 5/min) — aguarde 60 s\n"
            "- Conexao com a internet instavel\n"
            "- Nenhum ativo configurado em `config.py`\n\n"
            "Clique em **Atualizar Dados** no menu lateral e aguarde 1-2 minutos."
        )
        return
    scores_df = scores_df.sort_values("Score", ascending=False)

    # ── Ticker animado ──
    st.markdown(render_ticker_html(scores), unsafe_allow_html=True)

    # ── Estatisticas de zona ──────────────────────────────────────────────
    n_strong = int((scores_df["Score"] >= BUY_CONFIDENCE_STRONG).sum())
    n_buy    = int(((scores_df["Score"] >= BUY_CONFIDENCE_MODERATE) & (scores_df["Score"] < BUY_CONFIDENCE_STRONG)).sum())
    n_neutral= int(((scores_df["Score"] >= SELL_CONFIDENCE) & (scores_df["Score"] < BUY_CONFIDENCE_MODERATE)).sum())
    n_sell   = int((scores_df["Score"] < SELL_CONFIDENCE).sum())
    avg_score= round(float(scores_df["Score"].mean()), 1)

    zc1, zc2, zc3, zc4, zc5 = st.columns(5)
    zc1.metric("Compra Forte", n_strong, help="Score >= 72")
    zc2.metric("Compra",       n_buy,    help="Score 55-71")
    zc3.metric("Neutro",       n_neutral, help="Score 30-54")
    zc4.metric("Venda",        n_sell,   help="Score < 30")
    zc5.metric("Media Score",  avg_score)
    st.markdown("---")

    # ── Divergencias bullish ──────────────────────────────────────────────
    divergence_alerts = []
    for _, row in scores_df.iterrows():
        sr = row.get("_score_result", {})
        divs = sr.get("divergences", {})
        if divs.get("rsi", {}).get("type") == "bullish" or divs.get("macd", {}).get("type") == "bullish":
            divergence_alerts.append(row["Ativo"])

    if divergence_alerts:
        st.markdown(
            '<div class="divergence-box"><strong>DIVERGENCIA BULLISH DETECTADA</strong> &mdash; '
            + ", ".join(divergence_alerts)
            + "<br><span style='font-weight:400;font-size:0.82rem;color:#8B9AB0;'>Sinal forte de possivel reversao para alta.</span></div>",
            unsafe_allow_html=True,
        )

    # ── Alertas de score ──────────────────────────────────────────────────
    strong_buys = scores_df[scores_df["Score"] >= BUY_CONFIDENCE_STRONG]
    strong_sells = scores_df[scores_df["Score"] <= SELL_CONFIDENCE]
    if not strong_buys.empty:
        st.success("**OPORTUNIDADES:** " + ", ".join(
            [f"{r['Ativo']} (Score: {r['Score']}, {r['Confluencia']})" for _, r in strong_buys.iterrows()]))
    if not strong_sells.empty:
        st.error("**ZONA DE VENDA:** " + ", ".join(
            [f"{r['Ativo']} (Score: {r['Score']})" for _, r in strong_sells.iterrows()]))

    # ── Enriquece com delta de score do dia anterior (banco) ──────────────
    try:
        from database import get_score_trend
        for item in scores:
            t = get_score_trend(item.get("id", ""))
            delta = t.get("delta")
            if delta is None:
                item["Δ Score"] = "—"
            elif delta > 0:
                item["Δ Score"] = f"+{delta}"
            elif delta < 0:
                item["Δ Score"] = str(delta)
            else:
                item["Δ Score"] = "="
        scores_df = pd.DataFrame(scores).sort_values("Score", ascending=False)
    except Exception:
        pass

    # ── Heatmap visual ────────────────────────────────────────────────────
    st.markdown('<span class="section-label">Ranking de Oportunidades</span>', unsafe_allow_html=True)
    st.markdown(render_heatmap_html(scores_df), unsafe_allow_html=True)

    # Barra de filtros
    fc1, fc2, fc3 = st.columns([2, 2, 3])
    with fc1:
        tipo_filter = st.multiselect(
            "Tipo", ["Cripto", "Acao"], default=["Cripto", "Acao"],
            key="rank_tipo", label_visibility="collapsed",
            placeholder="Filtrar por tipo..."
        )
    with fc2:
        min_score_filter = st.slider("Score minimo", 0, 100, 0, key="rank_score", label_visibility="collapsed")
    with fc3:
        search_text = st.text_input(
            "Busca", "", key="rank_search",
            placeholder="Buscar ativo (ex: Bitcoin, PETR4...)",
            label_visibility="collapsed",
        )

    # ── Tabela detalhada + export CSV ─────────────────────────────────────
    display_cols = ["Ativo", "Tipo", "Preco", "24h", "Score", "Δ Score", "Confluencia", "Sinal", "Tendencia"]
    display_df = scores_df[[c for c in display_cols if c in scores_df.columns]].reset_index(drop=True)

    # Aplica filtros
    if tipo_filter:
        display_df = display_df[display_df["Tipo"].isin(tipo_filter)]
    if min_score_filter > 0:
        display_df = display_df[display_df["Score"] >= min_score_filter]
    if search_text.strip():
        display_df = display_df[display_df["Ativo"].str.contains(search_text.strip(), case=False, na=False)]

    if display_df.empty:
        st.info("Nenhum ativo corresponde aos filtros selecionados.")
    else:
        display_df = display_df.reset_index(drop=True)
        display_df.index += 1

        col_rank, col_export = st.columns([5, 1])
        with col_rank:
            st.dataframe(
                display_df,
                use_container_width=True,
                height=min(len(display_df) * 40 + 48, 600),
                column_config={
                    "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
                    "Δ Score": st.column_config.TextColumn("Δ Score", help="Variacao do score vs ontem"),
                },
            )
        with col_export:
            st.download_button(
                label="CSV",
                data=display_df.to_csv(index=False),
                file_name="ranking_botcripto.csv",
                mime="text/csv",
                use_container_width=True,
                help="Baixar ranking filtrado como CSV",
            )

    # Top 3 recomendacoes
    st.markdown('<span class="section-label">Top Recomendacoes</span>', unsafe_allow_html=True)
    top3 = scores_df.head(3)
    cols = st.columns(min(len(top3), 3))
    for i, (_, row) in enumerate(top3.iterrows()):
        if i >= 3:
            break
        with cols[i]:
            sr = row.get("_score_result")
            di = row.get("_dip_info")
            if sr and di:
                score_val = sr.get("score", 0)
                conf = sr.get("confluence", {})
                rec = generate_recommendation(sr, di, row["Ativo"])
                st.markdown(f"""
<div style="background:#0F1923;border:1px solid #1A2A40;border-radius:12px;
            padding:16px 18px 12px 18px;height:100%;">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;">
        <div>
            <div style="font-weight:700;font-size:0.88rem;color:#E8EDF5;margin-bottom:2px;">{row['Ativo']}</div>
            <div style="font-size:0.72rem;color:#5A7888;">{row['Preco']} &nbsp;·&nbsp; {row['24h']} (24h)</div>
        </div>
        {score_badge(score_val)}
    </div>
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
        <div>
            <div style="font-size:0.68rem;color:#5A7888;text-transform:uppercase;letter-spacing:1px;">Score</div>
            <div style="font-size:1.6rem;font-weight:700;color:#00E5C3;line-height:1;">{score_val}<span style="font-size:0.8rem;color:#5A7888;font-weight:400;">/100</span></div>
        </div>
        <div style="text-align:right;">
            <div style="font-size:0.68rem;color:#5A7888;text-transform:uppercase;letter-spacing:1px;">Confluencia</div>
            <div style="font-size:1.2rem;font-weight:700;color:#4A9EFF;">{conf.get('agree_buy',0)}<span style="font-size:0.8rem;color:#5A7888;font-weight:400;">/{conf.get('total',10)}</span></div>
        </div>
    </div>
    <div style="font-size:0.82rem;color:#7A9AB0;line-height:1.65;border-top:1px solid #1A2A40;padding-top:10px;">{rec}</div>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# PAGINA: Analise Detalhada
# -------------------------------------------------------
def render_deep_dive():
    st.markdown("""
<div class="page-header">
    <div class="page-header-bar"></div>
    <div><h2>Analise Detalhada</h2>
    <p>10 indicadores tecnicos + scoring + gestao de risco por ativo</p></div>
</div>""", unsafe_allow_html=True)

    asset_type = st.radio("Tipo de ativo", ["Criptomoeda", "Acao / ETF"], horizontal=True)
    if asset_type == "Criptomoeda":
        asset_id = st.selectbox("Ativo", CRYPTO_IDS, format_func=lambda x: x.title())
        a_type = "crypto"
    else:
        asset_id = st.selectbox("Ativo", STOCK_TICKERS)
        a_type = "stock"

    with st.spinner(f"Carregando analise de {asset_id.title()}..."):
        df, score_result, dip_info = get_history_and_analysis(asset_id, a_type)

    if df is None or df.empty:
        st.error(
            f"**Sem dados para {asset_id.upper()}**\n\n"
            "Possiveis causas:\n"
            "- Ativo nao disponivel na API neste momento\n"
            "- Limite de requisicoes atingido (aguarde 1 min)\n\n"
            "Tente selecionar outro ativo ou clique em **Atualizar Dados**."
        )
        return

    if not score_result:
        st.warning(
            f"Dados carregados, mas nao foi possivel calcular o score de **{asset_id.upper()}** "
            f"(apenas {len(df)} candles — minimo: 50). "
            "Tente novamente apos alguns minutos."
        )
        return

    # Header
    current_price = df["Close"].iloc[-1]
    prev_price = df["Close"].iloc[-2] if len(df) > 1 else current_price
    change = ((current_price - prev_price) / prev_price) * 100

    price_color  = "#00E5C3" if change >= 0 else "#FF4757"
    price_glow   = "rgba(0,229,195,0.28)" if change >= 0 else "rgba(255,71,87,0.28)"
    from_high    = ((current_price - df['Close'].max()) / df['Close'].max()) * 100
    from_h_color = "#FF4757" if from_high < -20 else ("#FFB800" if from_high < -5 else "#00E5C3")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(render_glass_metric(
            "Preco Atual",
            f"<span style='font-size:1.5rem;'>${current_price:,.2f}</span>",
            f"{'▲' if change>=0 else '▼'} {change:+.2f}% (24h)",
            price_color, price_glow
        ), unsafe_allow_html=True)
    with col2:
        st.markdown(render_glass_metric(
            "Maxima (Periodo)",
            f"<span style='font-size:1.5rem;'>${df['Close'].max():,.2f}</span>",
            "Maior preco nos ultimos 365 dias",
            "#4A9EFF", "rgba(74,158,255,0.22)"
        ), unsafe_allow_html=True)
    with col3:
        st.markdown(render_glass_metric(
            "Minima (Periodo)",
            f"<span style='font-size:1.5rem;'>${df['Close'].min():,.2f}</span>",
            "Menor preco nos ultimos 365 dias",
            "#8B9AB0", "rgba(139,154,176,0.15)"
        ), unsafe_allow_html=True)
    with col4:
        st.markdown(render_glass_metric(
            "Dist. da Maxima",
            f"<span style='font-size:1.5rem;'>{from_high:.1f}%</span>",
            "Quanto esta abaixo do topo historico",
            from_h_color, f"rgba(255,71,87,0.2)" if from_high < -20 else "rgba(0,229,195,0.15)"
        ), unsafe_allow_html=True)

    # Score + Confluencia + Divergencias
    if score_result:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.markdown(render_score_ring_html(score_result["score"]), unsafe_allow_html=True)

        with col2:
            conf  = score_result.get("confluence", {})
            agree = conf.get("agree_buy", 0)
            total = conf.get("total", 10)
            pct   = conf.get("percentage", 0)
            conf_color = "#00E5C3" if pct >= 60 else ("#FFB800" if pct >= 40 else "#FF4757")
            conf_glow  = "rgba(0,229,195,0.4)" if pct >= 60 else ("rgba(255,184,0,0.4)" if pct >= 40 else "rgba(255,71,87,0.4)")
            trend_txt  = score_result['trend'].replace('_', ' ').title()
            label_txt  = score_result['label']
            st.markdown(f"""
<div class="g-metric" style="--gm-color:{conf_color};--gm-glow:{conf_glow};margin-bottom:10px;">
    <span class="gm-label">Confluencia</span>
    <div class="gm-value" style="font-size:2.6rem;letter-spacing:-2px;">
        {agree}<span style="font-size:1rem;font-weight:400;color:#5A7888;">/{total}</span>
    </div>
    <div class="gm-sub">indicadores concordam &middot; {pct}%</div>
    <div class="gm-accent-bar"></div>
</div>
<div style="background:#0C1522;border:1px solid #1A2A40;border-radius:10px;
            padding:12px 16px;font-size:0.82rem;color:#7A9AB0;line-height:1.8;">
    <div><span style="color:#5A7888;font-size:0.68rem;text-transform:uppercase;
                       letter-spacing:1px;">Tendencia</span><br>
         <span style="color:#C0CDD8;font-weight:600;">{trend_txt}</span></div>
    <div style="margin-top:6px;"><span style="color:#5A7888;font-size:0.68rem;text-transform:uppercase;
                       letter-spacing:1px;">Classificacao</span><br>
         <span style="color:#C0CDD8;font-weight:600;">{label_txt}</span></div>
</div>
""", unsafe_allow_html=True)

        with col3:
            # Divergencias
            divs = score_result.get("divergences", {})
            rsi_div = divs.get("rsi", {})
            macd_div = divs.get("macd", {})

            if rsi_div.get("type") == "bullish" or macd_div.get("type") == "bullish":
                st.markdown(
                    '<div class="divergence-box"><strong>DIVERGENCIA BULLISH</strong><br>'
                    '<span style="font-weight:400;font-size:0.82rem;">Sinal forte de reversao para alta.</span></div>',
                    unsafe_allow_html=True,
                )
            elif rsi_div.get("type") == "bearish" or macd_div.get("type") == "bearish":
                st.markdown(
                    '<div class="alert-sell"><strong>DIVERGENCIA BEARISH</strong><br>'
                    '<span style="font-size:0.82rem;">Risco de reversao para baixa.</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.info("Nenhuma divergencia detectada")

            if dip_info:
                st.info(dip_info['explanation'])

    # Recomendacao
    if score_result and dip_info:
        rec = generate_recommendation(score_result, dip_info, asset_id.title())
        st.markdown(f'<div class="recommendation-box">{rec}</div>', unsafe_allow_html=True)

    # Grafico principal
    st.markdown('<span class="section-label">Grafico com Indicadores</span>', unsafe_allow_html=True)
    fig = create_price_chart(df, asset_id.title())
    st.plotly_chart(fig, use_container_width=True)

    # Breakdown dos sinais
    if score_result:
        st.markdown('<span class="section-label">Detalhamento dos 10 Indicadores</span>', unsafe_allow_html=True)
        signals = score_result.get("signals", {})
        signal_list = list(signals.items())

        # Exibir em grid 2x5
        for row_start in range(0, len(signal_list), 5):
            row_signals = signal_list[row_start:row_start + 5]
            cols = st.columns(len(row_signals))
            for j, (name, info) in enumerate(row_signals):
                with cols[j]:
                    pct = info["points"] / info["max"] if info["max"] > 0 else 0
                    if pct >= 0.6:
                        bar_color = "#00E5C3"
                        dot_html = '<span class="dot dot-green"></span>'
                    elif pct >= 0.3:
                        bar_color = "#FFB800"
                        dot_html = '<span class="dot dot-yellow"></span>'
                    else:
                        bar_color = "#FF4757"
                        dot_html = '<span class="dot dot-red"></span>'
                    signal_text = info.get('signal', '').replace('<','&lt;').replace('>','&gt;')
                    value_text  = str(info.get('value', 'N/A'))
                    st.markdown(f"""
<div style="background:#0F1923;border:1px solid #1A2A40;border-radius:10px;
            padding:14px 16px;margin:4px 0;height:100%;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:9px;">
        <span style="font-size:0.78rem;font-weight:600;color:#C0CDD8;display:flex;align-items:center;gap:5px;">{dot_html}{name}</span>
        <span style="font-size:0.72rem;color:#5A7888;font-weight:600;
                     background:#0A1520;padding:2px 7px;border-radius:10px;
                     border:1px solid #1A2A40;">{info['points']}/{info['max']}</span>
    </div>
    <div style="background:#0A1520;border-radius:3px;height:4px;overflow:hidden;margin-bottom:10px;">
        <div style="background:{bar_color};width:{pct*100:.0f}%;height:100%;border-radius:3px;
                    box-shadow:0 0 8px {bar_color}44;"></div>
    </div>
    <div style="font-size:0.7rem;color:#5A7888;margin-bottom:3px;">
        Valor: <span style="color:#7A9AB0;">{value_text}</span>
    </div>
    <div style="font-size:0.72rem;color:#5A7888;line-height:1.5;">{signal_text}</div>
</div>
""", unsafe_allow_html=True)

    # Niveis Fibonacci
    fib = df.attrs.get("fibonacci", {})
    fib_levels = fib.get("levels", {})
    if fib_levels:
        st.markdown('<span class="section-label">Niveis Fibonacci</span>', unsafe_allow_html=True)
        fib_df = pd.DataFrame([
            {"Nivel": label, "Preco": f"${price:,.2f}",
             "Distancia": f"{((current_price - price) / current_price * 100):+.1f}%"}
            for label, price in fib_levels.items()
        ])
        st.dataframe(fib_df, use_container_width=True, hide_index=True)

    # Historico de Score (banco de dados)
    try:
        from database import load_score_history
        score_hist = load_score_history(asset_id)
        if not score_hist.empty and len(score_hist) >= 2:
            st.markdown('<span class="section-label">Historico de Score</span>', unsafe_allow_html=True)
            fig_sh = go.Figure()
            fig_sh.add_trace(go.Scatter(
                x=score_hist["date"], y=score_hist["score"],
                mode="lines+markers",
                name="Score",
                line=dict(color="#00E5C3", width=2),
                fill="tozeroy",
                fillcolor="rgba(0,229,195,0.06)",
                marker=dict(size=5),
            ))
            fig_sh.add_hline(y=BUY_CONFIDENCE_STRONG, line_dash="dash",
                             line_color="#00E5C3", opacity=0.5,
                             annotation_text="Compra Forte", annotation_position="right")
            fig_sh.add_hline(y=BUY_CONFIDENCE_MODERATE, line_dash="dash",
                             line_color="#4A9EFF", opacity=0.4,
                             annotation_text="Compra", annotation_position="right")
            fig_sh.add_hline(y=SELL_CONFIDENCE, line_dash="dash",
                             line_color="#FF4757", opacity=0.4,
                             annotation_text="Venda", annotation_position="right")
            fig_sh.update_layout(
                height=240, template="plotly_dark",
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                yaxis=dict(range=[0, 100], title="Score"),
                xaxis=dict(title=""),
                margin=dict(l=50, r=80, t=20, b=20),
                showlegend=False,
            )
            st.plotly_chart(fig_sh, use_container_width=True)
            st.caption(f"{len(score_hist)} snapshots registrados desde {score_hist['date'].min()}")
    except Exception:
        pass


# -------------------------------------------------------
# PAGINA: Gestao de Risco
# -------------------------------------------------------
def render_risk():
    st.markdown("""
<div class="page-header">
    <div class="page-header-bar"></div>
    <div><h2>Gestao de Risco</h2>
    <p>Calculadora de posicao baseada em ATR, plano DCA e correlacao de ativos</p></div>
</div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Calculadora de Posicao", "Plano DCA", "Correlacao"])

    # --- TAB 1: Calculadora de Posicao ---
    with tab1:
        st.markdown('<span class="section-label">Calculadora de Posicao baseada em ATR</span>', unsafe_allow_html=True)
        st.info(
            "Esta calculadora usa o ATR (Average True Range) para definir stop-loss automatico "
            "e calcular quanto investir sem arriscar demais do seu portfolio."
        )

        asset_type_r = st.radio("Tipo de ativo", ["Criptomoeda", "Acao / ETF"], horizontal=True, key="risk_type")
        col1, col2 = st.columns(2)
        with col1:
            if asset_type_r == "Criptomoeda":
                asset_id_r = st.selectbox("Ativo", CRYPTO_IDS, format_func=lambda x: x.title(), key="risk_asset")
                a_type_r = "crypto"
            else:
                asset_id_r = st.selectbox("Ativo", STOCK_TICKERS, key="risk_asset")
                a_type_r = "stock"
        with col2:
            portfolio_val = st.number_input("Valor do seu Portfolio (USD)", value=float(DEFAULT_PORTFOLIO_VALUE),
                                           min_value=100.0, max_value=10_000_000.0, step=500.0)
            risk_pct = st.slider("Risco por operacao (%)", min_value=0.5, max_value=5.0,
                                value=min(float(DEFAULT_RISK_PER_TRADE), 5.0), step=0.5,
                                help="Percentual maximo do portfolio a arriscar. Acima de 5% e considerado alto risco.")

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
                c2.metric("Valor da Posicao", f"${risk['valor_posicao']:,.2f}")
                c3.metric("Perda Maxima", f"${risk['risco_maximo']:,.2f}",
                          f"{risk_pct}% do portfolio", delta_color="inverse")

                st.markdown("---")
                st.markdown("### Alvos de Lucro (Fibonacci)")
                c1, c2, c3 = st.columns(3)
                c1.metric("Alvo 1 (38.2%)", f"${risk['take_profit_1']}")
                c2.metric("Alvo 2 (23.6%)", f"${risk['take_profit_2']}")
                c3.metric("Alvo 3 (Topo)", f"${risk['take_profit_3']}")

                # Explicacao inline (sem expander para evitar bug de icone)
                st.markdown(f"""
<div style="background:#090F1C;border:1px solid #0F1E2E;border-radius:10px;
            padding:16px 20px;margin-top:12px;">
    <div style="font-size:0.68rem;color:#5A7888;text-transform:uppercase;
                letter-spacing:1.5px;margin-bottom:10px;">Como interpretar</div>
    <div style="font-size:0.82rem;color:#5A7888;line-height:1.8;">
        &mdash; O ATR de <span style="color:#8B9AB0;">${risk['atr']:,.2f}</span> ({risk['atr_percentual']}%) indica a volatilidade diaria media<br>
        &mdash; O stop-loss em <span style="color:#FF4757;">${risk['stop_loss']:,.2f}</span> esta a 2&times; ATR do preco atual<br>
        &mdash; Se o preco cair ate o stop-loss, voce perde no maximo <span style="color:#FF4757;">${risk['risco_maximo']:,.2f}</span> ({risk_pct}% do portfolio)<br>
        &mdash; Para isso, compre no maximo <span style="color:#8B9AB0;">{risk['tamanho_posicao']:.4f} unidades</span> (${risk['valor_posicao']:,.2f})
    </div>
    <div style="margin-top:10px;padding-top:10px;border-top:1px solid #0F1E2E;
                font-size:0.78rem;color:#5A7888;">
        Regra de ouro: nunca arrisque mais de 2% do portfolio em uma unica operacao.
    </div>
</div>
""", unsafe_allow_html=True)

    # --- TAB 2: Plano DCA ---
    with tab2:
        st.markdown('<span class="section-label">Plano DCA — Dollar Cost Averaging</span>', unsafe_allow_html=True)
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
                    text=[f"Alocar ${p['valor']:,.0f}" for p in plan],
                    textposition="outside",
                    customdata=[[p["preco"], p["valor"], p["tranche"]] for p in plan],
                    hovertemplate=(
                        "<b>Parcela %{customdata[2]}</b><br>"
                        "Valor a investir: <b>$%{customdata[1]:,.0f}</b><br>"
                        "Preço alvo de compra: <b>$%{customdata[0]:,.2f}</b>"
                        "<extra></extra>"
                    ),
                    marker_color=["#6366F1", "#38BDF8", "#F59E0B", "#F43F5E", "#A78BFA", "#34D399"][:len(plan)],
                ))
                fig.update_layout(
                    height=300, template="plotly_dark",
                    paper_bgcolor="#07080F", plot_bgcolor="#07080F",
                    yaxis_title="Valor a Investir (USD)", showlegend=False,
                    margin=dict(l=50, r=20, t=30, b=20),
                    yaxis=dict(gridcolor="#161C30"),
                    xaxis=dict(gridcolor="#161C30"),
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("""
                **Estrategia:** Coloque ordens limite nos precos indicados.
                Se o preco cair, suas ordens executam automaticamente comprando mais barato.
                Se o preco subir, voce ja tem a primeira parcela investida.
                """)

    # --- TAB 3: Correlacao ---
    with tab3:
        st.markdown('<span class="section-label">Matriz de Correlacao entre Ativos</span>', unsafe_allow_html=True)
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
    st.markdown("""
<div class="page-header">
    <div class="page-header-bar"></div>
    <div><h2>Central de Alertas</h2>
    <p>Sinais de compra, venda e divergencias bullish/bearish em tempo real</p></div>
</div>""", unsafe_allow_html=True)

    scores, all_assets = analyze_all_assets()

    if not scores:
        st.error(
            "**Sem dados de score disponíveis.**\n\n"
            "Acesse a pagina **Visao Geral** primeiro para carregar os scores, "
            "depois volte para os Alertas."
        )
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
        st.markdown(f'<span class="section-label">Divergencias Bullish Detectadas &mdash; {len(div_alerts)} ativo(s)</span>', unsafe_allow_html=True)
        for alert in div_alerts:
            conf = alert.get("confluence", {})
            st.markdown(f"""
            <div class="divergence-box">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                    <strong style="font-size:0.95rem;">{alert['name']}</strong>
                    <span style="font-size:0.78rem;font-weight:400;color:#8B9AB0;">Score {alert['score']}/100 &nbsp;|&nbsp; {conf.get('agree_buy',0)}/{conf.get('total',10)} indicadores</span>
                </div>
                <div style="font-size:0.8rem;font-weight:400;color:#8B9AB0;">{alert['price']} &nbsp;&mdash;&nbsp; {alert['change_24h']} (24h)</div>
                <div style="margin-top:6px;font-size:0.82rem;">DIVERGENCIA BULLISH &mdash; Sinal de possivel reversao para alta</div>
            </div>
            """, unsafe_allow_html=True)

    # Compra forte
    st.markdown(f'<span class="section-label">Zona de Compra Forte &mdash; {len(buy_alerts)} ativo(s)</span>', unsafe_allow_html=True)
    if buy_alerts:
        for alert in sorted(buy_alerts, key=lambda x: x["score"], reverse=True):
            conf = alert.get("confluence", {})
            st.markdown(f"""
            <div class="alert-buy">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                    <strong style="font-size:0.95rem;color:#E8EDF5;">{alert['name']}</strong>
                    <span style="font-size:0.78rem;color:#4A5568;">Score {alert['score']}/100 &nbsp;|&nbsp; {conf.get('agree_buy',0)}/{conf.get('total',10)} indicadores</span>
                </div>
                <div style="font-size:0.8rem;color:#4A5568;margin-bottom:6px;">{alert['price']} &nbsp;&mdash;&nbsp; {alert['change_24h']} (24h)</div>
                <div style="font-size:0.83rem;color:#8B9AB0;">{alert['explanation']}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info(
            "Nenhum ativo em zona de compra forte agora (Score >= 72). "
            "Scores acima de 72 sao raros — exigem multiplos indicadores simultaneamente alinhados. "
            "Veja os ativos em **Observacao** abaixo para oportunidades proximas do limiar."
        )

    # Venda
    st.markdown(f'<span class="section-label">Zona de Alerta / Venda &mdash; {len(sell_alerts)} ativo(s)</span>', unsafe_allow_html=True)
    if sell_alerts:
        for alert in sorted(sell_alerts, key=lambda x: x["score"]):
            st.markdown(f"""
            <div class="alert-sell">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                    <strong style="font-size:0.95rem;color:#E8EDF5;">{alert['name']}</strong>
                    <span style="font-size:0.78rem;color:#4A5568;">Score {alert['score']}/100</span>
                </div>
                <div style="font-size:0.8rem;color:#4A5568;margin-bottom:6px;">{alert['price']} &nbsp;&mdash;&nbsp; {alert['change_24h']} (24h)</div>
                <div style="font-size:0.83rem;color:#8B9AB0;">{alert['explanation']}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Nenhum ativo em zona de venda (Score < 30) no momento — mercado sem sinais de saida.")

    # Observacao
    st.markdown(f'<span class="section-label">Em Observacao &mdash; {len(watch_alerts)} ativo(s)</span>', unsafe_allow_html=True)
    if watch_alerts:
        for alert in sorted(watch_alerts, key=lambda x: x["score"], reverse=True):
            conf = alert.get("confluence", {})
            pct = conf.get("percentage", 0)
            st.markdown(f"""
<div style="background:#0F1923;border:1px solid #1A2A40;border-left:3px solid #4A9EFF;
            border-radius:10px;padding:14px 18px;margin:6px 0;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
        <strong style="font-size:0.92rem;color:#E8EDF5;">{alert['name']}</strong>
        <span style="font-size:0.76rem;color:#4A5568;">Score {alert['score']}/100 &nbsp;|&nbsp; {conf.get('agree_buy',0)}/{conf.get('total',10)} indicadores ({pct}%)</span>
    </div>
    <div style="font-size:0.78rem;color:#4A5568;margin-bottom:4px;">{alert['price']} &nbsp;&mdash;&nbsp; {alert['change_24h']} (24h)</div>
    <div style="font-size:0.8rem;color:#7A9AB0;">{alert['explanation']}</div>
</div>""", unsafe_allow_html=True)
    else:
        st.info("Nenhum ativo em observacao (Score 55-71) no momento.")

    # Historico de alertas disparados (banco de dados)
    st.markdown("---")
    st.markdown('<span class="section-label">Historico de Alertas — Ultimos 30 dias</span>', unsafe_allow_html=True)
    try:
        from database import load_alert_history
        hist_df = load_alert_history(days=30)
        if not hist_df.empty:
            type_labels = {
                "strong_buy": "Compra Forte",
                "bullish_divergence_rsi": "Divergencia RSI",
                "bullish_divergence_macd": "Divergencia MACD",
            }
            hist_df["Tipo"] = hist_df["Tipo"].map(lambda t: type_labels.get(t, t))
            st.dataframe(hist_df[["Ativo", "Tipo", "Data", "Hora"]], use_container_width=True, hide_index=True)
        else:
            st.info("Nenhum alerta registrado nos ultimos 30 dias. Os alertas aparecem aqui quando notificacoes sao disparadas.")
    except Exception:
        pass


# -------------------------------------------------------
# PAGINA: Noticias
# -------------------------------------------------------
def render_news():
    st.markdown("""
<div class="page-header">
    <div class="page-header-bar"></div>
    <div><h2>Noticias e Sentimento</h2>
    <p>Fear &amp; Greed Index historico e manchetes com analise de sentimento</p></div>
</div>""", unsafe_allow_html=True)

    fg_df = fetch_fear_greed()
    if not fg_df.empty:
        st.markdown('<span class="section-label">Fear & Greed Index — Historico 30 dias</span>', unsafe_allow_html=True)
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

    st.markdown('<span class="section-label">Ultimas Noticias</span>', unsafe_allow_html=True)
    news = fetch_news()

    if news:
        pos = sum(1 for n in news if n["sentiment_label"] == "Positivo")
        neg = sum(1 for n in news if n["sentiment_label"] == "Negativo")
        neu = sum(1 for n in news if n["sentiment_label"] == "Neutro")

        c1, c2, c3 = st.columns(3)
        c1.metric("Positivas", pos)
        c2.metric("Neutras", neu)
        c3.metric("Negativas", neg)

        news_colors = {
            "Positivo": ("#00E5C3", "rgba(0,229,195,0.06)", "rgba(0,229,195,0.18)", "+"),
            "Negativo": ("#FF4757", "rgba(255,71,87,0.06)",  "rgba(255,71,87,0.18)",  "−"),
            "Neutro":   ("#4A9EFF", "rgba(74,158,255,0.05)", "rgba(74,158,255,0.15)", "~"),
        }
        for article in news:
            color, bg, border, symbol = news_colors.get(
                article["sentiment_label"],
                ("#4A9EFF", "rgba(74,158,255,0.05)", "rgba(74,158,255,0.15)", "~")
            )
            link_html = f'<a href="{article["url"]}" target="_blank" style="color:{color};font-size:0.75rem;text-decoration:none;font-weight:600;letter-spacing:0.3px;">Ler noticia &rarr;</a>' if article["url"] else ""
            st.markdown(f"""
<div style="background:{bg};border:1px solid {border};border-left:3px solid {color};
            border-radius:10px;padding:14px 18px;margin:6px 0;">
    <div style="display:flex;align-items:flex-start;gap:10px;">
        <span style="font-size:0.7rem;font-weight:700;color:{color};background:rgba(255,255,255,0.05);
                     border:1px solid {border};border-radius:4px;padding:2px 6px;
                     flex-shrink:0;margin-top:2px;">{symbol}</span>
        <div style="flex:1;">
            <div style="font-size:0.86rem;color:#C0CDD8;font-weight:500;line-height:1.4;margin-bottom:6px;">{article['title']}</div>
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="font-size:0.72rem;color:#5A7888;">{article['source']} &nbsp;·&nbsp; {article['date']}</span>
                {link_html}
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
    else:
        st.warning(
            "**Feeds RSS nao responderam.**\n\n"
            "Os feeds de noticias (CoinDesk, CoinTelegraph, Yahoo Finance) podem estar "
            "temporariamente indisponiveis. Tente novamente em 2-3 minutos clicando em "
            "**Atualizar Dados** no menu lateral."
        )


# -------------------------------------------------------
# PAGINA: Portfolio
# -------------------------------------------------------
def render_portfolio():
    st.markdown("""
<div class="page-header">
    <div class="page-header-bar"></div>
    <div><h2>Meu Portfolio</h2>
    <p>Acompanhe valor total, P&amp;L e alocacao dos seus ativos</p></div>
</div>""", unsafe_allow_html=True)
    st.info("Adicione seus ativos para acompanhar valor total, P&L e alocacao.")

    if "portfolio" not in st.session_state:
        try:
            from database import load_portfolio
            st.session_state.portfolio = load_portfolio()
            if st.session_state.portfolio:
                st.caption(f"Portfolio carregado do banco: {len(st.session_state.portfolio)} ativo(s).")
        except Exception:
            st.session_state.portfolio = {}

    # ── Formulario de adicao ──────────────────────────────────────────────
    with st.expander("Adicionar ativo ao portfolio", expanded=not bool(st.session_state.portfolio)):
        p_type = st.radio("Tipo", ["crypto", "stock"], key="p_type", horizontal=True,
                          format_func=lambda x: "Cripto" if x == "crypto" else "Acao")
        col1, col2, col3 = st.columns(3)
        with col1:
            if p_type == "crypto":
                p_asset = st.selectbox("Ativo", CRYPTO_IDS, key="p_asset",
                                       format_func=lambda x: x.title())
            else:
                p_asset = st.selectbox("Ativo", STOCK_TICKERS, key="p_asset")
        with col2:
            p_qty = st.number_input("Quantidade", min_value=0.0, step=0.01, key="p_qty",
                                    help="Numero de unidades/acoes adquiridas")
        with col3:
            p_buy = st.number_input("Preco de Compra (USD)", min_value=0.0, step=0.01, key="p_buy",
                                    help="Preco medio de compra por unidade em dolares")

        if st.button("Adicionar ao Portfolio", use_container_width=True):
            if p_qty <= 0:
                st.error("Quantidade deve ser maior que zero.")
            elif p_buy <= 0:
                st.error("Preco de compra deve ser maior que zero.")
            else:
                st.session_state.portfolio[p_asset] = {
                    "type": p_type, "quantity": p_qty, "buy_price": p_buy
                }
                db_ok = False
                try:
                    from database import save_portfolio_entry
                    db_ok = save_portfolio_entry(p_asset, p_type, p_qty, p_buy)
                except Exception:
                    pass
                db_note = " (salvo no banco)" if db_ok else " (apenas em memoria — banco offline)"
                st.success(f"{p_asset.upper()}: {p_qty:.6g} un. @ ${p_buy:,.2f}{db_note}")
                st.rerun()

    if not st.session_state.portfolio:
        st.info(
            "Seu portfolio esta vazio. "
            "Clique em **Adicionar ativo ao portfolio** acima para comecar a acompanhar seus investimentos."
        )
        return

    portfolio_data = []
    total_invested = 0.0
    total_current = 0.0

    for asset_id, info in st.session_state.portfolio.items():
        if info["type"] == "crypto":
            cdf = fetch_crypto_current([asset_id])
            price = float(cdf.iloc[0]["price"]) if not cdf.empty else 0.0
        else:
            sdf = fetch_stock_history(asset_id, period="5d")
            price = float(sdf["Close"].iloc[-1]) if (not sdf.empty and len(sdf) > 0) else 0.0

        current_value = price * info["quantity"]
        invested = info["buy_price"] * info["quantity"]
        pnl = current_value - invested
        pnl_pct = ((price - info["buy_price"]) / info["buy_price"] * 100) if info["buy_price"] > 0 else 0.0

        total_invested += invested
        total_current += current_value
        portfolio_data.append({
            "Ativo": asset_id, "Qtd": info["quantity"],
            "Compra": f"${info['buy_price']:,.2f}", "Atual": f"${price:,.2f}",
            "Valor": f"${current_value:,.2f}", "P&L": f"${pnl:,.2f}", "P&L%": f"{pnl_pct:+.2f}%",
            "_current_value": current_value,   # valor numerico para o grafico de pizza
        })

    total_pnl = total_current - total_invested
    total_pnl_pct = ((total_current - total_invested) / total_invested * 100) if total_invested > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Investido", f"${total_invested:,.2f}")
    c2.metric("Atual", f"${total_current:,.2f}")
    c3.metric("P&L", f"${total_pnl:,.2f}", f"{total_pnl_pct:+.2f}%")
    c4.metric("Ativos", len(st.session_state.portfolio))

    display_df = pd.DataFrame([{k: v for k, v in p.items() if not k.startswith("_")} for p in portfolio_data])
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    if portfolio_data:
        fig = go.Figure(data=[go.Pie(
            labels=[p["Ativo"] for p in portfolio_data],
            values=[p["_current_value"] for p in portfolio_data],   # numerico direto — sem parse de string
            hole=0.4,
            marker_colors=["#00d4aa", "#3498db", "#f39c12", "#e74c3c", "#9b59b6",
                          "#1abc9c", "#e67e22", "#2ecc71", "#f1c40f", "#34495e"],
        )])
        fig.update_layout(title="Alocacao", template="plotly_dark", paper_bgcolor="#0e1117", height=350)
        st.plotly_chart(fig, use_container_width=True)

    col_rem, col_clear = st.columns([3, 1])
    with col_rem:
        if st.session_state.portfolio:
            rem_asset = st.selectbox("Remover ativo", list(st.session_state.portfolio.keys()), key="p_remove")
            if st.button("Remover selecionado"):
                del st.session_state.portfolio[rem_asset]
                try:
                    from database import delete_portfolio_entry
                    delete_portfolio_entry(rem_asset)
                except Exception:
                    pass
                st.rerun()
    with col_clear:
        if st.button("Limpar Portfolio"):
            try:
                from database import clear_portfolio
                clear_portfolio()
            except Exception:
                pass
            st.session_state.portfolio = {}
            st.rerun()


# -------------------------------------------------------
# PAGINA: Backtesting
# -------------------------------------------------------
def _simulate_strategy(
    score_hist: pd.DataFrame,
    price_hist: pd.DataFrame,
    buy_thresh: int,
    sell_thresh: int,
    capital: float,
) -> tuple[list[dict], float, pd.DataFrame]:
    """
    Simula uma estrategia simples baseada em score:
      - Compra quando score cruza acima de buy_thresh
      - Vende quando score cruza abaixo de sell_thresh

    Retorna (trades, capital_final, equity_curve).
    """
    # Alinha score com preco pela data
    price_hist = price_hist.copy()
    price_hist.index = pd.to_datetime(price_hist.index)
    score_hist = score_hist.copy()
    score_hist["date"] = pd.to_datetime(score_hist["date"])
    merged = score_hist.set_index("date").join(price_hist[["Close"]], how="inner").dropna()
    merged = merged.sort_index()

    if len(merged) < 2:
        return [], capital, pd.DataFrame()

    trades: list[dict] = []
    equity = capital
    position: dict | None = None
    equity_curve = []

    prev_score = merged["score"].iloc[0]

    for dt, row in merged.iterrows():
        cur_score = int(row["score"])
        close = float(row["Close"])
        cur_equity = equity if position is None else (position["shares"] * close)

        # Sinal de compra: score cruza ACIMA do threshold
        if position is None and prev_score < buy_thresh <= cur_score:
            shares = equity / close
            position = {"entry_date": dt, "entry_price": close, "shares": shares}

        # Sinal de venda: score cruza ABAIXO do threshold
        elif position is not None and cur_score <= sell_thresh < prev_score:
            exit_value = position["shares"] * close
            pnl = exit_value - equity
            pnl_pct = (exit_value / equity - 1) * 100
            trades.append({
                "Entrada": position["entry_date"].strftime("%Y-%m-%d"),
                "Saida": dt.strftime("%Y-%m-%d"),
                "Preco Entrada": f"${position['entry_price']:,.2f}",
                "Preco Saida": f"${close:,.2f}",
                "P&L": f"${pnl:+,.2f}",
                "P&L %": f"{pnl_pct:+.2f}%",
                "Resultado": "Lucro" if pnl >= 0 else "Prejuizo",
                "_pnl_raw": pnl,
            })
            equity = exit_value
            position = None

        equity_curve.append({"date": dt, "equity": cur_equity})
        prev_score = cur_score

    # Fecha posicao aberta no ultimo preco
    if position is not None:
        last_close = float(merged["Close"].iloc[-1])
        cur_equity = position["shares"] * last_close
        equity_curve[-1]["equity"] = cur_equity
        equity = cur_equity

    eq_df = pd.DataFrame(equity_curve).set_index("date") if equity_curve else pd.DataFrame()
    return trades, equity, eq_df


def render_backtesting():
    st.markdown("""
<div class="page-header">
    <div class="page-header-bar"></div>
    <div><h2>Backtesting de Estrategia</h2>
    <p>Simule sua estrategia de compra/venda com dados historicos coletados pelo BotCripto</p></div>
</div>""", unsafe_allow_html=True)

    st.info(
        "O backtesting usa os dados de score e preco registrados automaticamente pelo BotCripto "
        "a cada vez que a Visao Geral e carregada. Quanto mais dias de uso, mais rico o historico."
    )

    # ── Configuracao ─────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        bt_type = st.selectbox("Tipo", ["Criptomoeda", "Acao/ETF"], key="bt_type")
        if bt_type == "Criptomoeda":
            bt_asset = st.selectbox("Ativo", CRYPTO_IDS, format_func=lambda x: x.title(), key="bt_asset")
        else:
            bt_asset = st.selectbox("Ativo", STOCK_TICKERS, key="bt_asset")
    with col2:
        bt_capital = st.number_input("Capital inicial (USD)", value=10000.0, min_value=100.0, step=500.0, key="bt_cap")
        col_buy, col_sell = st.columns(2)
        bt_buy  = col_buy.number_input("Score de COMPRA",  value=BUY_CONFIDENCE_STRONG, min_value=1, max_value=99, key="bt_buy")
        bt_sell = col_sell.number_input("Score de VENDA",  value=SELL_CONFIDENCE,       min_value=1, max_value=99, key="bt_sell")

    if bt_buy <= bt_sell:
        st.error("O score de COMPRA deve ser maior que o score de VENDA.")
        return

    st.markdown("---")

    # ── Carrega dados ─────────────────────────────────────────────────────
    try:
        from database import load_score_history, load_price_history
        score_hist  = load_score_history(bt_asset)
        price_hist  = load_price_history(bt_asset)
    except Exception as e:
        st.error(f"Erro ao acessar banco de dados: {e}")
        return

    MIN_SCORE_SNAPS = 10
    MIN_PRICE_DAYS  = 20

    if len(score_hist) < MIN_SCORE_SNAPS or len(price_hist) < MIN_PRICE_DAYS:
        n_snaps = len(score_hist)
        n_price = len(price_hist)
        falta_snaps = max(0, MIN_SCORE_SNAPS - n_snaps)
        falta_price = max(0, MIN_PRICE_DAYS - n_price)

        st.warning(f"**Historico insuficiente para {bt_asset.upper()}**")
        col_a, col_b = st.columns(2)
        col_a.metric(
            "Snapshots de score",
            f"{n_snaps}/{MIN_SCORE_SNAPS}",
            delta=f"Faltam {falta_snaps}" if falta_snaps else "OK",
            delta_color="inverse" if falta_snaps else "normal",
        )
        col_b.metric(
            "Dias de preco",
            f"{n_price}/{MIN_PRICE_DAYS}",
            delta=f"Faltam {falta_price}" if falta_price else "OK",
            delta_color="inverse" if falta_price else "normal",
        )
        st.markdown(
            "**Como acumular historico:**\n"
            "1. Abra a pagina **Visao Geral** uma vez por dia — cada visita salva um snapshot\n"
            "2. Os precos sao coletados automaticamente junto com os scores\n"
            f"3. Em ~{max(falta_snaps, falta_price)} dias de uso o backtesting estara disponivel"
        )
        return

    # ── Executa simulacao ────────────────────────────────────────────────
    trades, final_equity, eq_df = _simulate_strategy(
        score_hist, price_hist, bt_buy, bt_sell, bt_capital
    )

    # ── Metricas resumo ──────────────────────────────────────────────────
    total_return = (final_equity / bt_capital - 1) * 100
    n_trades = len(trades)
    n_wins   = sum(1 for t in trades if t["_pnl_raw"] >= 0)
    win_rate = (n_wins / n_trades * 100) if n_trades else 0.0

    # Calcula max drawdown a partir da equity curve
    max_dd = 0.0
    if not eq_df.empty:
        roll_max = eq_df["equity"].cummax()
        drawdowns = (eq_df["equity"] - roll_max) / roll_max * 100
        max_dd = float(drawdowns.min())

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Capital Inicial", f"${bt_capital:,.0f}")
    mc2.metric("Capital Final",   f"${final_equity:,.0f}")
    mc3.metric("Retorno Total",   f"{total_return:+.1f}%")
    mc4.metric("Trades",          n_trades, help=f"{n_wins} lucrativos")
    mc5.metric("Win Rate",        f"{win_rate:.0f}%")
    st.markdown(f"**Max Drawdown:** `{max_dd:.1f}%`")
    st.markdown("---")

    # ── Grafico de equity ───────────────────────────────────────────────
    if not eq_df.empty:
        st.markdown('<span class="section-label">Curva de Capital</span>', unsafe_allow_html=True)
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=eq_df.index, y=eq_df["equity"],
            mode="lines", name="Capital",
            line=dict(color="#00E5C3", width=2),
            fill="tozeroy", fillcolor="rgba(0,229,195,0.06)",
        ))
        fig_eq.add_hline(y=bt_capital, line_dash="dash", line_color="gray",
                         opacity=0.5, annotation_text="Capital inicial")
        fig_eq.update_layout(
            height=280, template="plotly_dark",
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            yaxis=dict(title="USD"), xaxis=dict(title=""),
            margin=dict(l=50, r=20, t=20, b=20), showlegend=False,
        )
        st.plotly_chart(fig_eq, use_container_width=True)

    # ── Score ao longo do tempo ──────────────────────────────────────────
    st.markdown('<span class="section-label">Score e Sinais</span>', unsafe_allow_html=True)
    fig_sc = go.Figure()
    fig_sc.add_trace(go.Scatter(
        x=score_hist["date"], y=score_hist["score"],
        mode="lines+markers", name="Score",
        line=dict(color="#4A9EFF", width=1.5),
        marker=dict(size=4),
    ))
    fig_sc.add_hline(y=bt_buy, line_dash="dash", line_color="#00E5C3",
                     opacity=0.6, annotation_text=f"Compra >= {bt_buy}")
    fig_sc.add_hline(y=bt_sell, line_dash="dash", line_color="#FF4757",
                     opacity=0.6, annotation_text=f"Venda <= {bt_sell}")
    # Marca entradas e saidas
    for t in trades:
        fig_sc.add_vline(x=t["Entrada"], line_dash="solid", line_color="#00E5C3", opacity=0.4, line_width=1)
        fig_sc.add_vline(x=t["Saida"],   line_dash="solid", line_color="#FF4757",  opacity=0.4, line_width=1)
    fig_sc.update_layout(
        height=260, template="plotly_dark",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        yaxis=dict(range=[0, 100], title="Score"), xaxis=dict(title=""),
        margin=dict(l=50, r=80, t=20, b=20), showlegend=False,
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    # ── Tabela de trades ─────────────────────────────────────────────────
    if trades:
        st.markdown('<span class="section-label">Historico de Trades</span>', unsafe_allow_html=True)
        trades_df = pd.DataFrame([{k: v for k, v in t.items() if not k.startswith("_")} for t in trades])
        st.dataframe(trades_df, use_container_width=True, hide_index=True)
        csv_bt = trades_df.to_csv(index=False)
        st.download_button("Exportar Trades CSV", csv_bt,
                           f"backtesting_{bt_asset}.csv", "text/csv")
    else:
        st.info(
            f"Nenhum trade executado com Score Compra >= {bt_buy} / Score Venda <= {bt_sell}.\n\n"
            f"**Sugestoes:**\n"
            f"- Reduza o Score de Compra de {bt_buy} para {max(50, bt_buy - 5)}\n"
            f"- Eleve o Score de Venda de {bt_sell} para {min(bt_sell + 5, bt_buy - 10)}\n"
            f"- Aguarde mais dias de historico (atual: {len(score_hist)} snapshots)"
        )

    # ── Dados brutos ─────────────────────────────────────────────────────
    with st.expander("Ver dados brutos de score"):
        st.dataframe(score_hist, use_container_width=True, hide_index=True)


# -------------------------------------------------------
# Roteamento
# -------------------------------------------------------
if page == "Visao Geral":
    render_overview()
elif page == "Analise Detalhada":
    render_deep_dive()
elif page == "Gestao de Risco":
    render_risk()
elif page == "Alertas":
    render_alerts()
elif page == "Noticias":
    render_news()
elif page == "Portfolio":
    render_portfolio()
elif page == "Backtesting":
    render_backtesting()
