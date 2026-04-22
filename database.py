# ============================================================
# BotCripto - Camada de persistencia (SQLAlchemy)
# ============================================================
#
# Responsavel por salvar o historico de precos puxados e os
# scores calculados do dia, para uso em backtesting futuro.
#
# String de conexao configuravel via env var BOTCRIPTO_DB_URL.
# Default: SQLite local em ./data/botcripto.db.
#
# Exemplos de configuracao:
#   SQLite (default):
#     BOTCRIPTO_DB_URL=sqlite:///data/botcripto.db
#
#   PostgreSQL (para migracao futura):
#     BOTCRIPTO_DB_URL=postgresql://user:pass@host:5432/botcripto
# ============================================================

import os
import json
import threading
from datetime import datetime, date, timedelta
from typing import Optional

import pandas as pd
from sqlalchemy import (
    create_engine, Column, Integer, Float, String, DateTime, Date,
    UniqueConstraint, Index, text, func,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError


# ============================================================
# Configuracao
# ============================================================

_DEFAULT_DB_PATH = os.path.join("data", "botcripto.db")
DB_URL = os.getenv("BOTCRIPTO_DB_URL", f"sqlite:///{_DEFAULT_DB_PATH}")

# Garantir que a pasta data/ exista quando estivermos usando SQLite local
if DB_URL.startswith("sqlite:///") and not DB_URL.startswith("sqlite:////"):
    _db_file = DB_URL.replace("sqlite:///", "", 1)
    _db_dir = os.path.dirname(_db_file)
    if _db_dir:
        os.makedirs(_db_dir, exist_ok=True)

# SQLite precisa de check_same_thread=False para uso em Streamlit multi-thread
_engine_kwargs = {}
if DB_URL.startswith("sqlite:"):
    _engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(DB_URL, future=True, **_engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()

_init_lock = threading.Lock()
_initialized = False


# ============================================================
# Modelos
# ============================================================

class PriceHistory(Base):
    """OHLCV diario por ativo — base do backtesting."""
    __tablename__ = "price_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(String(64), nullable=False, index=True)
    asset_type = Column(String(16), nullable=False)   # "crypto" | "stock"
    date = Column(Date, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float, nullable=False)
    volume = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("asset_id", "date", name="uq_price_asset_date"),
        Index("ix_price_asset_date", "asset_id", "date"),
    )


class ScoreSnapshot(Base):
    """Snapshot diario do score e sinais de um ativo."""
    __tablename__ = "score_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(String(64), nullable=False, index=True)
    asset_type = Column(String(16), nullable=False)
    snapshot_date = Column(Date, nullable=False)
    snapshot_ts = Column(DateTime, default=datetime.utcnow)

    score = Column(Integer, nullable=False)
    label = Column(String(32))                        # COMPRA FORTE / COMPRA / ...
    trend = Column(String(32))
    trend_strength = Column(Integer)
    confluence_agree = Column(Integer)
    confluence_total = Column(Integer)

    rsi_divergence = Column(String(16))               # none | bullish | bearish
    macd_divergence = Column(String(16))

    # Campo JSON com o dict completo de sinais — util para debugging
    signals_json = Column(String)

    __table_args__ = (
        UniqueConstraint("asset_id", "snapshot_date", name="uq_score_asset_date"),
        Index("ix_score_asset_date", "asset_id", "snapshot_date"),
    )


class AlertLog(Base):
    """Registro de alertas disparados — evita duplicar notificacao."""
    __tablename__ = "alert_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(String(64), nullable=False, index=True)
    alert_type = Column(String(32), nullable=False)   # strong_buy | bullish_divergence
    snapshot_date = Column(Date, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("asset_id", "alert_type", "snapshot_date",
                         name="uq_alert_asset_type_date"),
    )


class PortfolioEntry(Base):
    """Posicao do portfolio — persiste entre sessoes do Streamlit."""
    __tablename__ = "portfolio"

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(String(64), nullable=False, unique=True)
    asset_type = Column(String(16), nullable=False)   # "crypto" | "stock"
    quantity = Column(Float, nullable=False)
    buy_price = Column(Float, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ============================================================
# Inicializacao (lazy — cria tabelas na 1a chamada)
# ============================================================

def init_db() -> None:
    """Cria todas as tabelas no banco configurado. Idempotente."""
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        Base.metadata.create_all(bind=engine)
        _initialized = True


def get_session() -> Session:
    """Factory de sessao — sempre chama init_db() antes."""
    init_db()
    return SessionLocal()


# ============================================================
# Operacoes de escrita
# ============================================================

def save_price_history(
    asset_id: str,
    asset_type: str,
    df: pd.DataFrame,
) -> int:
    """
    Persiste (upsert) historico OHLCV de um ativo.

    Retorna o numero de linhas novas inseridas.
    Erros sao silenciados para nao quebrar o fluxo da UI.
    """
    if df is None or df.empty or "Close" not in df.columns:
        return 0

    try:
        with get_session() as session:
            existing = {
                r[0] for r in session.query(PriceHistory.date)
                .filter(PriceHistory.asset_id == asset_id).all()
            }
            inserted = 0
            for idx, row in df.iterrows():
                try:
                    d = pd.to_datetime(idx).date()
                except Exception:
                    continue
                if d in existing:
                    continue
                session.add(PriceHistory(
                    asset_id=asset_id,
                    asset_type=asset_type,
                    date=d,
                    open=float(row.get("Open", row["Close"])),
                    high=float(row.get("High", row["Close"])),
                    low=float(row.get("Low", row["Close"])),
                    close=float(row["Close"]),
                    volume=float(row.get("Volume", 0) or 0),
                ))
                inserted += 1
            session.commit()
            return inserted
    except SQLAlchemyError:
        return 0


def save_scores_snapshot(scores: list[dict], snapshot_date: Optional[date] = None) -> int:
    """
    Persiste o score do dia para cada ativo (upsert por (asset_id, date)).

    scores: lista produzida por analyze_all_assets() em app.py.
    Retorna o numero de linhas novas inseridas.
    """
    if not scores:
        return 0

    snapshot_date = snapshot_date or date.today()
    inserted = 0

    try:
        with get_session() as session:
            existing = {
                r[0] for r in session.query(ScoreSnapshot.asset_id)
                .filter(ScoreSnapshot.snapshot_date == snapshot_date).all()
            }

            new_rows: list[ScoreSnapshot] = []
            for item in scores:
                asset_id = item.get("id")
                if not asset_id or asset_id in existing:
                    continue

                sr = item.get("_score_result") or {}
                conf = sr.get("confluence", {}) or {}
                divs = sr.get("divergences", {}) or {}

                # Tambem persiste historico de precos se presente
                df_asset = item.get("_df")
                if isinstance(df_asset, pd.DataFrame):
                    save_price_history(asset_id, item.get("type", "crypto"), df_asset)

                new_rows.append(ScoreSnapshot(
                    asset_id=asset_id,
                    asset_type=item.get("type", "crypto"),
                    snapshot_date=snapshot_date,
                    score=int(sr.get("score", item.get("Score", 0)) or 0),
                    label=sr.get("label", item.get("Sinal")),
                    trend=sr.get("trend"),
                    trend_strength=int(sr.get("trend_strength", 0) or 0),
                    confluence_agree=int(conf.get("agree_buy", 0) or 0),
                    confluence_total=int(conf.get("total", 10) or 10),
                    rsi_divergence=(divs.get("rsi") or {}).get("type", "none"),
                    macd_divergence=(divs.get("macd") or {}).get("type", "none"),
                    signals_json=json.dumps(sr.get("signals", {}), default=str),
                ))

            # Insercao em bloco — mais eficiente que add() individual em loops grandes
            session.add_all(new_rows)
            session.commit()
            inserted = len(new_rows)
    except SQLAlchemyError:
        return 0

    return inserted


def record_alert(asset_id: str, alert_type: str,
                 snapshot_date: Optional[date] = None) -> bool:
    """
    Registra um alerta disparado. Retorna True se foi registrado pela
    primeira vez hoje (ou seja, a notificacao DEVE ser enviada),
    False se ja existia (evita spam de notificacoes).
    """
    snapshot_date = snapshot_date or date.today()
    try:
        with get_session() as session:
            already = session.query(AlertLog).filter(
                AlertLog.asset_id == asset_id,
                AlertLog.alert_type == alert_type,
                AlertLog.snapshot_date == snapshot_date,
            ).first()
            if already:
                return False
            session.add(AlertLog(
                asset_id=asset_id,
                alert_type=alert_type,
                snapshot_date=snapshot_date,
            ))
            session.commit()
            return True
    except SQLAlchemyError:
        return False


# ============================================================
# Operacoes de leitura (uteis para backtesting)
# ============================================================

def load_price_history(asset_id: str) -> pd.DataFrame:
    """Retorna o historico OHLCV salvo de um ativo como DataFrame."""
    try:
        with get_session() as session:
            rows = session.query(PriceHistory).filter(
                PriceHistory.asset_id == asset_id
            ).order_by(PriceHistory.date.asc()).all()
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame([{
                "Date": r.date,
                "Open": r.open, "High": r.high,
                "Low": r.low, "Close": r.close, "Volume": r.volume,
            } for r in rows])
            df.set_index("Date", inplace=True)
            return df
    except SQLAlchemyError:
        return pd.DataFrame()


def load_score_history(asset_id: str) -> pd.DataFrame:
    """Retorna o historico de scores salvo de um ativo como DataFrame."""
    try:
        with get_session() as session:
            rows = session.query(ScoreSnapshot).filter(
                ScoreSnapshot.asset_id == asset_id
            ).order_by(ScoreSnapshot.snapshot_date.asc()).all()
            if not rows:
                return pd.DataFrame()
            return pd.DataFrame([{
                "date": r.snapshot_date,
                "score": r.score,
                "label": r.label,
                "trend": r.trend,
                "rsi_divergence": r.rsi_divergence,
                "macd_divergence": r.macd_divergence,
            } for r in rows])
    except SQLAlchemyError:
        return pd.DataFrame()


def get_score_trend(asset_id: str) -> dict:
    """
    Compara os dois ultimos snapshots de score de um ativo.

    Retorna um dict com:
      latest_score   : score do ultimo registro
      previous_score : score do penultimo registro (ou None)
      delta          : diferenca (latest - previous), ou None
      direction      : 'up' | 'down' | 'flat' | None
    """
    try:
        with get_session() as session:
            rows = (
                session.query(ScoreSnapshot.score, ScoreSnapshot.snapshot_date)
                .filter(ScoreSnapshot.asset_id == asset_id)
                .order_by(ScoreSnapshot.snapshot_date.desc())
                .limit(2)
                .all()
            )
        if not rows:
            return {"latest_score": None, "previous_score": None, "delta": None, "direction": None}
        latest = rows[0].score
        if len(rows) < 2:
            return {"latest_score": latest, "previous_score": None, "delta": None, "direction": None}
        previous = rows[1].score
        delta = latest - previous
        direction = "up" if delta > 0 else ("down" if delta < 0 else "flat")
        return {"latest_score": latest, "previous_score": previous, "delta": delta, "direction": direction}
    except SQLAlchemyError:
        return {"latest_score": None, "previous_score": None, "delta": None, "direction": None}


def load_alert_history(days: int = 30) -> pd.DataFrame:
    """
    Retorna os alertas disparados nos ultimos N dias como DataFrame.

    Colunas: asset_id, alert_type, snapshot_date, created_at
    """
    try:
        with get_session() as session:
            cutoff = date.today() - timedelta(days=days)
            rows = (
                session.query(AlertLog)
                .filter(AlertLog.snapshot_date >= cutoff)
                .order_by(AlertLog.created_at.desc())
                .all()
            )
            if not rows:
                return pd.DataFrame()
            return pd.DataFrame([{
                "Ativo": r.asset_id,
                "Tipo": r.alert_type,
                "Data": r.snapshot_date,
                "Hora": r.created_at,
            } for r in rows])
    except SQLAlchemyError:
        return pd.DataFrame()


# ============================================================
# Portfolio — persistencia entre sessoes
# ============================================================

def save_portfolio_entry(
    asset_id: str,
    asset_type: str,
    quantity: float,
    buy_price: float,
) -> bool:
    """
    Salva (upsert) uma posicao no portfolio.

    Retorna True se bem-sucedido, False em caso de erro.
    """
    try:
        with get_session() as session:
            entry = session.query(PortfolioEntry).filter(
                PortfolioEntry.asset_id == asset_id
            ).first()
            if entry:
                entry.asset_type = asset_type
                entry.quantity = quantity
                entry.buy_price = buy_price
                entry.updated_at = datetime.utcnow()
            else:
                session.add(PortfolioEntry(
                    asset_id=asset_id,
                    asset_type=asset_type,
                    quantity=quantity,
                    buy_price=buy_price,
                ))
            session.commit()
            return True
    except SQLAlchemyError:
        return False


def delete_portfolio_entry(asset_id: str) -> bool:
    """Remove uma posicao do portfolio pelo asset_id."""
    try:
        with get_session() as session:
            session.query(PortfolioEntry).filter(
                PortfolioEntry.asset_id == asset_id
            ).delete()
            session.commit()
            return True
    except SQLAlchemyError:
        return False


def clear_portfolio() -> bool:
    """Remove todas as posicoes do portfolio."""
    try:
        with get_session() as session:
            session.query(PortfolioEntry).delete()
            session.commit()
            return True
    except SQLAlchemyError:
        return False


def load_portfolio() -> dict[str, dict]:
    """
    Carrega o portfolio salvo no banco.

    Retorna dict no mesmo formato que st.session_state.portfolio:
      { asset_id: {"type": ..., "quantity": ..., "buy_price": ...} }
    """
    try:
        with get_session() as session:
            rows = session.query(PortfolioEntry).all()
            return {
                r.asset_id: {
                    "type": r.asset_type,
                    "quantity": r.quantity,
                    "buy_price": r.buy_price,
                }
                for r in rows
            }
    except SQLAlchemyError:
        return {}
