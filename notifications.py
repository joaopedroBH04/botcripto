# ============================================================
# BotCripto - Modulo de Alertas via Webhook
# ============================================================
#
# Suporta Telegram e Discord via webhook. A escolha do canal
# e automatica a partir das variaveis de ambiente:
#
#   TELEGRAM_BOT_TOKEN  + TELEGRAM_CHAT_ID   -> Telegram
#   DISCORD_WEBHOOK_URL                       -> Discord
#   BOTCRIPTO_WEBHOOK_URL                     -> Webhook generico (POST JSON)
#
# Se nenhuma var estiver configurada, a funcao apenas faz log
# e retorna False sem erro — assim o dashboard nunca quebra.
#
# Gatilhos de alerta (dispatch_strong_signals):
#   1. Score >= 72  (COMPRA FORTE)
#   2. Divergencia Bullish detectada (RSI ou MACD)
#
# Deduplicacao: cada (asset, tipo) so dispara uma notificacao
# por dia, via a tabela alert_log (modulo database.py).
# ============================================================

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Optional

import requests


log = logging.getLogger("botcripto.notifications")

STRONG_BUY_THRESHOLD = 72   # alinhado com config.BUY_CONFIDENCE_STRONG

# Importacao lazy do modulo de banco — evita dependencia circular e falha
# silenciosa quando database nao esta disponivel (ex.: testes sem SQLAlchemy).
try:
    from database import record_alert as _record_alert
except Exception:  # pragma: no cover
    _record_alert = None  # type: ignore[assignment]


# ============================================================
# Abstracao de canais
# ============================================================

class NotificationChannel(ABC):
    """Canal generico de notificacao."""

    @abstractmethod
    def send(self, title: str, message: str) -> bool:
        """Envia uma mensagem. Retorna True se bem-sucedido."""
        raise NotImplementedError


class TelegramChannel(NotificationChannel):
    """Envia mensagens via Telegram Bot API."""

    def __init__(self, bot_token: str, chat_id: str, timeout: int = 10):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.timeout = timeout
        self.url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    def send(self, title: str, message: str) -> bool:
        try:
            body = f"*{title}*\n\n{message}"
            resp = requests.post(
                self.url,
                json={"chat_id": self.chat_id, "text": body, "parse_mode": "Markdown"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return True
        except Exception as e:
            log.warning("Falha ao enviar alerta Telegram: %s", e)
            return False


class DiscordChannel(NotificationChannel):
    """Envia mensagens via webhook do Discord."""

    def __init__(self, webhook_url: str, timeout: int = 10):
        self.webhook_url = webhook_url
        self.timeout = timeout

    def send(self, title: str, message: str) -> bool:
        try:
            payload = {
                "embeds": [{
                    "title": title,
                    "description": message,
                    "color": 0x16a34a,   # verde — sinal positivo
                }]
            }
            resp = requests.post(self.webhook_url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return True
        except Exception as e:
            log.warning("Falha ao enviar alerta Discord: %s", e)
            return False


class GenericWebhookChannel(NotificationChannel):
    """
    Webhook generico que faz POST com JSON {title, message, timestamp}.
    Util para integracao customizada (Slack, n8n, Zapier, etc.).
    """

    def __init__(self, url: str, timeout: int = 10):
        self.url = url
        self.timeout = timeout

    def send(self, title: str, message: str) -> bool:
        try:
            resp = requests.post(
                self.url,
                json={"title": title, "message": message},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return True
        except Exception as e:
            log.warning("Falha ao enviar alerta Webhook: %s", e)
            return False


class NullChannel(NotificationChannel):
    """Canal 'no-op' usado quando nenhuma credencial esta configurada."""

    def send(self, title: str, message: str) -> bool:
        log.info("[Notificacao local] %s - %s", title, message)
        return False


# ============================================================
# Factory a partir das env vars
# ============================================================

def build_channel_from_env() -> NotificationChannel:
    """Escolhe o canal baseado nas variaveis de ambiente disponiveis."""
    tg_token = os.getenv("TELEGRAM_BOT_TOKEN")
    tg_chat = os.getenv("TELEGRAM_CHAT_ID")
    if tg_token and tg_chat:
        return TelegramChannel(tg_token, tg_chat)

    discord_url = os.getenv("DISCORD_WEBHOOK_URL")
    if discord_url:
        return DiscordChannel(discord_url)

    generic = os.getenv("BOTCRIPTO_WEBHOOK_URL")
    if generic:
        return GenericWebhookChannel(generic)

    return NullChannel()


# Canal global — construido lazy na primeira chamada
_channel: Optional[NotificationChannel] = None


def get_channel() -> NotificationChannel:
    global _channel
    if _channel is None:
        _channel = build_channel_from_env()
    return _channel


def reset_channel() -> None:
    """Limpa o cache do canal — util em testes."""
    global _channel
    _channel = None


# ============================================================
# API publica
# ============================================================

def send_alert(title: str, message: str) -> bool:
    """Envia um alerta pelo canal configurado."""
    return get_channel().send(title, message)


def notify_strong_buy(asset_name: str, score: int, signals_summary: str = "") -> bool:
    title = f"COMPRA FORTE: {asset_name} (Score {score}/100)"
    msg = (
        f"O ativo *{asset_name}* atingiu score {score}/100, "
        f"indicando COMPRA FORTE.\n"
        f"{signals_summary}".strip()
    )
    return send_alert(title, msg)


def notify_bullish_divergence(asset_name: str, source: str, description: str = "") -> bool:
    """
    source: 'RSI' ou 'MACD'.
    """
    title = f"Divergencia Bullish ({source}): {asset_name}"
    msg = (
        f"Divergencia bullish detectada em *{asset_name}* via {source}.\n"
        f"Sinal classico de possivel reversao para alta.\n\n"
        f"{description}".strip()
    )
    return send_alert(title, msg)


# ============================================================
# Integracao com o fluxo do dashboard
# ============================================================

def dispatch_strong_signals(scores: list[dict]) -> int:
    """
    Dispara notificacoes para todos os ativos que cumprirem os criterios:
      - score >= 72 (COMPRA FORTE), ou
      - divergencia bullish detectada em RSI ou MACD

    Usa database.record_alert para evitar notificar o mesmo ativo mais
    de uma vez por dia. Retorna o numero de notificacoes enviadas.
    """
    if not scores:
        return 0

    # Usa a referencia ao modulo importada no nivel do modulo.
    # Pode ser substituida em testes via patch("notifications._record_alert").
    record_alert = _record_alert

    sent = 0
    for item in scores:
        asset_id = item.get("id") or item.get("Ativo") or ""
        asset_name = item.get("Ativo") or asset_id
        sr = item.get("_score_result") or {}
        score = int(sr.get("score", item.get("Score", 0)) or 0)
        divs = sr.get("divergences", {}) or {}
        rsi_type = (divs.get("rsi") or {}).get("type", "none")
        macd_type = (divs.get("macd") or {}).get("type", "none")

        # 1. COMPRA FORTE (score >= 72)
        if score >= STRONG_BUY_THRESHOLD:
            should_send = True
            if record_alert is not None and asset_id:
                should_send = record_alert(asset_id, "strong_buy")
            if should_send:
                signals = sr.get("signals", {}) or {}
                top_signal = max(
                    signals.items(),
                    key=lambda kv: (kv[1].get("points", 0) / max(kv[1].get("max", 1), 1)),
                    default=(None, None),
                )
                signals_summary = ""
                if top_signal[0]:
                    signals_summary = f"Indicador mais forte: {top_signal[0]} — {top_signal[1].get('signal', '')}"
                if notify_strong_buy(asset_name, score, signals_summary):
                    sent += 1

        # 2. Divergencia Bullish (RSI ou MACD)
        if rsi_type == "bullish":
            should_send = True
            if record_alert is not None and asset_id:
                should_send = record_alert(asset_id, "bullish_divergence_rsi")
            if should_send:
                desc = (divs.get("rsi") or {}).get("description", "")
                if notify_bullish_divergence(asset_name, "RSI", desc):
                    sent += 1

        if macd_type == "bullish":
            should_send = True
            if record_alert is not None and asset_id:
                should_send = record_alert(asset_id, "bullish_divergence_macd")
            if should_send:
                desc = (divs.get("macd") or {}).get("description", "")
                if notify_bullish_divergence(asset_name, "MACD", desc):
                    sent += 1

    return sent
