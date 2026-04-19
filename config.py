# ============================================================
# BotCripto v2 - Configuracoes
# Edite este arquivo para personalizar seus ativos e parametros
# ============================================================

# --- Criptomoedas monitoradas (IDs do CoinGecko) ---
# NOTA: API gratuita do CoinGecko permite ~8-10 req/min.
# Cada cripto aqui gera 1 requisicao de historico.
# Recomendado: ate 6-8 criptos para evitar erros 429.
CRYPTO_IDS = [
    "bitcoin", "ethereum", "solana", "ripple", "cardano",
    "dogecoin", "avalanche-2",
]

# Criptos extras (descomente para monitorar mais, mas pode demorar mais):
# CRYPTO_IDS += ["chainlink", "polkadot", "polygon-ecosystem-token"]

# --- Acoes da B3 (Bolsa do Brasil) ---
# Use .SA para acoes negociadas na B3
BR_STOCK_TICKERS = [
    "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBAS3.SA", "WEGE3.SA",
    "ABEV3.SA", "MGLU3.SA", "RENT3.SA", "BOVA11.SA", "IVVB11.SA",
]

# --- Acoes e ETFs internacionais (EUA) ---
GLOBAL_STOCK_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA",
    "VOO", "QQQ",
]

# --- Lista completa de acoes (B3 + EUA) ---
STOCK_TICKERS = BR_STOCK_TICKERS + GLOBAL_STOCK_TICKERS

# --- Limiares de analise tecnica ---
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
STOCH_RSI_OVERSOLD = 20
STOCH_RSI_OVERBOUGHT = 80

# --- ADX (forca da tendencia) ---
ADX_WEAK = 20        # Abaixo = sem tendencia clara (mercado lateral)
ADX_STRONG = 25      # Acima = tendencia confirmada
ADX_VERY_STRONG = 40 # Acima = tendencia muito forte

# --- EMA (medias moveis exponenciais) ---
EMA_PERIODS = [9, 21]

# --- ATR (volatilidade e stop-loss) ---
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0  # Stop-loss = preco - (ATR * multiplicador)

# --- Gestao de risco ---
DEFAULT_RISK_PER_TRADE = 2.0   # % do portfolio a arriscar por operacao
DEFAULT_PORTFOLIO_VALUE = 10000.0  # Valor padrao em USD (float obrigatorio)

# --- Fibonacci (retracements) ---
FIBONACCI_LEVELS = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
FIBONACCI_LABELS = ["0%", "23.6%", "38.2%", "50%", "61.8%", "78.6%", "100%"]

# --- Periodos de media movel ---
SMA_PERIODS = [20, 50, 200]

# --- Scoring v2: pontuacao maxima por indicador (soma = 100) ---
# Rebalanceado para incluir novos indicadores
SCORE_MAX_RSI = 12
SCORE_MAX_STOCH_RSI = 8
SCORE_MAX_MACD = 10
SCORE_MAX_TREND = 12        # SMA + EMA combinados
SCORE_MAX_ADX = 10
SCORE_MAX_BOLLINGER = 8
SCORE_MAX_VOLUME = 10       # Volume + OBV combinados
SCORE_MAX_FIBONACCI = 10    # Substitui suporte basico
SCORE_MAX_FEAR_GREED = 8
SCORE_MAX_DIVERGENCE = 12   # Divergencia RSI + MACD (sinal mais forte)

# --- Classificacao de confianca de compra ---
BUY_CONFIDENCE_STRONG = 72   # >= 72 = Compra Forte
BUY_CONFIDENCE_MODERATE = 55  # >= 55 = Compra Moderada
SELL_CONFIDENCE = 30          # <= 30 = Zona de Venda

# --- APIs ---
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=30"

# --- Cache (segundos) ---
CACHE_TTL = 600  # 10 minutos

# --- Periodo de analise ---
LOOKBACK_DAYS = 365


# ============================================================
# Helpers de classificacao de score (centralizados aqui para
# evitar logica duplicada em app.py, database.py, etc.)
# ============================================================

def score_label(score: int) -> tuple[str, str]:
    """
    Retorna (texto, classe_css) para um score numerico.

    Classes disponiveis:
      'buy'     -> COMPRA FORTE  (score >= BUY_CONFIDENCE_STRONG)
      'watch'   -> OBSERVACAO    (score >= BUY_CONFIDENCE_MODERATE)
      'neutral' -> NEUTRO        (score >= SELL_CONFIDENCE)
      'sell'    -> VENDA         (score <  SELL_CONFIDENCE)
    """
    if score >= BUY_CONFIDENCE_STRONG:
        return "COMPRA FORTE", "buy"
    elif score >= BUY_CONFIDENCE_MODERATE:
        return "OBSERVACAO", "watch"
    elif score >= SELL_CONFIDENCE:
        return "NEUTRO", "neutral"
    return "VENDA", "sell"
