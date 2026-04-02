# 📊 BotCripto — Monitor Inteligente de Mercado

> Painel analítico completo para monitoramento de criptomoedas e ações, com indicadores técnicos profissionais, gestão de risco automática e alertas inteligentes.

**Autor:** João Amaral · [@joaopedroBH04](https://github.com/joaopedroBH04)

---

## 🎯 Motivação

A maioria das ferramentas gratuitas de análise financeira mostra apenas se um ativo subiu ou caiu no dia. O problema é que isso não diz nada sobre o que fazer — um ativo pode estar mais barato hoje do que ontem e ainda assim ser uma péssima compra, porque a tendência é continuar caindo.

O **BotCripto** foi criado para resolver exatamente isso: analisar o **contexto completo** do mercado antes de sugerir qualquer ação, combinando 10 indicadores técnicos diferentes numa pontuação unificada com explicação em português.

---

## 🖥️ Screenshots

| Visão Geral | Análise Detalhada | Gestão de Risco |
|---|---|---|
| Ranking de ativos por score | Gráfico completo com indicadores | Calculadora ATR + DCA |

---

## ✨ Funcionalidades

### 📈 Análise Técnica Completa (10 indicadores)

| Indicador | O que mede |
|-----------|-----------|
| **RSI (14)** | Sobrecompra / Sobrevenda |
| **Stochastic RSI** | Timing de entrada mais preciso |
| **MACD** | Momentum e cruzamentos de tendência |
| **EMA 9/21** | Tendência de curto prazo (mais rápida que SMA) |
| **SMA 20/50/200** | Tendência de médio e longo prazo |
| **ADX + DI+/DI-** | Força real da tendência (evita sinais falsos em mercado lateral) |
| **Bollinger Bands** | Volatilidade e zonas de sobreextensão |
| **OBV (On Balance Volume)** | Se o "smart money" está comprando ou vendendo |
| **Fibonacci Retracement** | Níveis matemáticos de suporte e resistência |
| **Divergências RSI/MACD** | Sinal mais forte de reversão que existe |

### 🧠 Score Inteligente (0–100)

Todos os 10 indicadores são combinados numa pontuação única:

```
≥ 72  →  🟢 COMPRA FORTE
55–71 →  🟡 COMPRA
30–54 →  ⚪ NEUTRO
< 30  →  🔴 VENDA / EVITAR
```

O sistema também mostra **confluência** — quantos indicadores concordam simultaneamente. Uma pontuação alta com alta confluência (ex: 8/10 indicadores alinhados) é muito mais confiável do que uma pontuação alta com baixa confluência.

### 💡 Detecção de Divergências

Quando o preço faz uma mínima mais baixa mas o RSI ou MACD fazem uma mínima mais alta, isso é chamado de **divergência bullish** — historicamente um dos sinais mais confiáveis de reversão para alta. O sistema detecta isso automaticamente e emite alertas visíveis.

### 🛡️ Gestão de Risco

**Calculadora de Posição (ATR-based):**
- Stop-loss automático baseado na volatilidade real do ativo (2× ATR)
- Calcula exatamente quantas unidades comprar para não arriscar mais de X% do portfolio
- Mostra alvos de lucro baseados em Fibonacci
- Ratio risco/retorno calculado automaticamente

**Plano DCA (Dollar Cost Averaging):**
- Divide o investimento em parcelas nos níveis de Fibonacci
- Distribui mais dinheiro nos preços mais baixos (61.8% recebe 30%, preço atual recebe 15%)
- Se o preço cair, você compra mais barato automaticamente

**Matriz de Correlação:**
- Mostra quais ativos se movem juntos
- Identifica pares com correlação > 0.8 (exposição duplicada a evitar)
- Identifica pares com correlação < 0.3 (boa diversificação)

### 🔔 Alertas Automáticos

- Zona de Compra Forte: ativos com score ≥ 72
- Zona de Venda: ativos com score ≤ 30
- Divergência Bullish detectada (destaque em verde)
- Em Observação: ativos com score entre 55–72

### 📰 Notícias e Sentimento

- Fear & Greed Index com histórico de 30 dias
- RSS feeds de CoinDesk, CoinTelegraph e Yahoo Finance
- Análise de sentimento por palavras-chave (positivo/neutro/negativo)

### 💼 Portfolio Tracker

- Acompanhe seus ativos com P&L em tempo real
- Gráfico de alocação (pizza)
- Preço médio de compra vs. preço atual

---

## 🏗️ Arquitetura

```
BotCripto/
├── app.py              # Dashboard Streamlit (UI + roteamento)
├── analysis.py         # Motor de análise técnica e scoring
├── data_fetcher.py     # Coleta de dados com rate limiting inteligente
├── config.py           # Configurações, watchlists e parâmetros
├── requirements.txt    # Dependências Python
└── .streamlit/
    └── config.toml     # Tema escuro
```

### Fluxo de dados

```
CoinGecko API ──┐
Yahoo Finance ──┼──► data_fetcher.py ──► analysis.py ──► app.py ──► Browser
Fear&Greed  ──┘         (cache 10min)   (10 indicadores)  (Streamlit)
RSS Feeds   ──┘
```

### Rate Limiting inteligente

A API gratuita do CoinGecko tem limite de ~10 req/min. O sistema usa:
- `RateLimiter` com janela deslizante (máx. 5 req/min, conservador)
- Retry automático com backoff exponencial (20s, 40s, 60s, 80s)
- Cache agressivo de 10–30 minutos por endpoint
- Pausa de 5s entre buscas de histórico de ativos diferentes

---

## 🚀 Instalação e Uso

### Pré-requisitos

- Python 3.11+
- pip

### Passo a passo

```bash
# 1. Clone o repositório
git clone https://github.com/joaopedroBH04/botcripto.git
cd botcripto

# 2. (Opcional) Crie um ambiente virtual
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Execute o dashboard
streamlit run app.py
```

O painel abre automaticamente em `http://localhost:8501`.

### Nenhuma API key necessária

Todas as fontes de dados usadas são **100% gratuitas** e não requerem cadastro:
- [CoinGecko API](https://www.coingecko.com/en/api) — criptomoedas
- [yfinance](https://pypi.org/project/yfinance/) — ações via Yahoo Finance
- [alternative.me](https://alternative.me/crypto/fear-and-greed-index/) — Fear & Greed Index
- RSS públicos — notícias

---

## ⚙️ Configuração

Edite `config.py` para personalizar:

```python
# Criptomoedas monitoradas
CRYPTO_IDS = [
    "bitcoin", "ethereum", "solana", "ripple", "cardano", "dogecoin", "avalanche-2",
]

# Ações e ETFs (use .SA para B3)
STOCK_TICKERS = [
    "PETR4.SA", "VALE3.SA", "ITUB4.SA",   # Brasil
    "AAPL", "NVDA", "TSLA",                # EUA
    "VOO", "BOVA11.SA",                    # ETFs
]

# Thresholds do scoring
BUY_CONFIDENCE_STRONG = 72   # Score ≥ 72 = Compra Forte
BUY_CONFIDENCE_MODERATE = 55 # Score ≥ 55 = Compra
SELL_CONFIDENCE = 30         # Score ≤ 30 = Venda

# Gestão de risco padrão
DEFAULT_RISK_PER_TRADE = 2.0  # % do portfolio por operação
ATR_STOP_MULTIPLIER = 2.0     # Stop-loss = preço - (2 × ATR)
```

---

## 📐 Sistema de Scoring (v2)

O score final (0–100) é a soma das pontuações de 10 indicadores independentes:

| Indicador | Pontos Máx. | Lógica |
|-----------|:-----------:|--------|
| RSI 14 | 12 | RSI ≤ 20 = 12pts; RSI ≤ 30 = 9pts; escala progressiva |
| Stochastic RSI | 8 | Sobrevendido + cruzamento bullish = 8pts |
| MACD | 10 | Cruzamento bullish + histograma crescendo = 10pts |
| Tendência SMA+EMA | 12 | Preço abaixo da SMA50 + EMA virando = 12pts |
| ADX | 10 | ADX > 40 com DI+ > DI- = 10pts; ADX < 20 = mercado lateral |
| Bollinger Bands | 8 | Preço na banda inferior + RSI baixo + bandas estáveis = 8pts |
| Volume + OBV | 10 | Queda no volume + OBV subindo = smart money comprando = 10pts |
| Fibonacci | 10 | Preço no nível 61.8% = 10pts; nível 50% = 8pts |
| Fear & Greed | 8 | Medo Extremo (≤15) = 8pts; lógica contrarian |
| Divergências | 12 | RSI bullish = 6pts; MACD bullish = 6pts (mais alto = mais caro) |

### Por que divergências valem mais?

Divergência bullish ocorre quando o preço faz uma **mínima mais baixa** mas o indicador faz uma **mínima mais alta**. Isso significa que, mesmo com o preço caindo, o momentum de venda está enfraquecendo — um dos sinais de reversão mais estudados em análise técnica, com alta taxa de acerto histórica.

---

## 📊 Páginas do Dashboard

### 🏠 Visão Geral
- Fear & Greed Index atual
- Dominância do BTC
- Ranking completo de ativos por score
- Alerta de divergências bullish detectadas
- Top 3 recomendações com texto explicativo

### 🔍 Análise Detalhada
- Gráfico interativo com: Preço, SMA 20/50/200, EMA 9/21, Bollinger Bands, Fibonacci, RSI, Stoch RSI, MACD, ADX
- Gauge de score com classificação
- Indicador de confluência (X/10 indicadores concordam)
- Detecção de divergências com destaque visual
- Breakdown dos 10 indicadores individuais com explicação em português
- Tabela de níveis Fibonacci com distância percentual do preço atual

### 🛡️ Gestão de Risco
- **Calculadora ATR:** stop-loss automático, tamanho de posição, alvos de lucro
- **Plano DCA:** parcelas por níveis Fibonacci com distribuição inteligente de capital
- **Matriz de correlação:** heatmap interativo com alertas de exposição duplicada

### 🔔 Alertas
- Ativos em Compra Forte (score ≥ 72)
- Ativos com divergência bullish
- Ativos em observação (score 55–71)
- Ativos em zona de venda (score ≤ 30)

### 📰 Notícias
- Histórico 30 dias do Fear & Greed Index
- Feed de notícias com análise de sentimento
- Contador de notícias positivas/neutras/negativas

### 💼 Portfolio
- Registro de posições com preço médio de compra
- Valor atual e P&L em tempo real
- Gráfico de alocação por ativo

---

## ⚠️ Aviso Legal

> Este projeto é **estritamente educacional e informativo**. As análises e pontuações geradas pelo BotCripto **não constituem recomendação de investimento**. O mercado financeiro envolve riscos e perdas de capital. Sempre faça sua própria análise e, se necessário, consulte um profissional certificado antes de investir.

---

## 🤝 Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests.

1. Fork o repositório
2. Crie sua branch: `git checkout -b feat/nova-funcionalidade`
3. Commit suas mudanças: `git commit -m 'feat: adiciona nova funcionalidade'`
4. Push: `git push origin feat/nova-funcionalidade`
5. Abra um Pull Request

---

## 📦 Dependências

```
streamlit>=1.30.0     # Dashboard web
pandas>=2.0.0         # Manipulação de dados
numpy>=1.24.0         # Cálculos numéricos
plotly>=5.18.0        # Gráficos interativos
yfinance>=0.2.36      # Dados de ações (Yahoo Finance)
ta>=0.11.0            # Indicadores de análise técnica
requests>=2.31.0      # Requisições HTTP
feedparser>=6.0.0     # Parser de RSS feeds
```

---

## 📄 Licença

MIT License — veja [LICENSE](LICENSE) para detalhes.

---

<div align="center">
  <strong>Desenvolvido por João Amaral</strong><br>
  <a href="https://github.com/joaopedroBH04">@joaopedroBH04</a>
</div>
