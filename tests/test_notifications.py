# ============================================================
# tests/test_notifications.py — Testes do modulo notifications.py
# ============================================================

import os
import sys
import pytest
from unittest.mock import patch, MagicMock, call


# ============================================================
# Imports do modulo
# ============================================================

import notifications as notif


# ============================================================
# Fixture: limpa o canal global entre testes
# ============================================================

@pytest.fixture(autouse=True)
def reset_channel():
    """Garante que o canal global seja recriado a cada teste."""
    notif.reset_channel()
    yield
    notif.reset_channel()


# ============================================================
# NullChannel
# ============================================================

class TestNullChannel:
    def test_send_returns_false(self):
        ch = notif.NullChannel()
        assert ch.send("titulo", "mensagem") is False

    def test_send_does_not_raise(self):
        ch = notif.NullChannel()
        ch.send("titulo", "mensagem")   # sem excecao


# ============================================================
# TelegramChannel
# ============================================================

class TestTelegramChannel:
    def test_send_success(self):
        ch = notif.TelegramChannel("TOKEN123", "CHAT456")
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        with patch("notifications.requests.post", return_value=mock_resp) as mock_post:
            result = ch.send("Titulo", "Mensagem")
        assert result is True
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert "TOKEN123" in call_kwargs.args[0]

    def test_send_includes_chat_id(self):
        ch = notif.TelegramChannel("TOKEN123", "CHAT456")
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        with patch("notifications.requests.post", return_value=mock_resp) as mock_post:
            ch.send("T", "M")
        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert payload["chat_id"] == "CHAT456"

    def test_send_failure_returns_false(self):
        ch = notif.TelegramChannel("TOKEN", "CHAT")
        with patch("notifications.requests.post", side_effect=Exception("timeout")):
            result = ch.send("Titulo", "Mensagem")
        assert result is False

    def test_send_http_error_returns_false(self):
        import requests as req
        ch = notif.TelegramChannel("TOKEN", "CHAT")
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = req.HTTPError("403 Forbidden")
        with patch("notifications.requests.post", return_value=mock_resp):
            result = ch.send("T", "M")
        assert result is False

    def test_message_uses_markdown_format(self):
        ch = notif.TelegramChannel("TOKEN", "CHAT")
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        with patch("notifications.requests.post", return_value=mock_resp) as mock_post:
            ch.send("MeuTitulo", "MinhaMensagem")
        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert payload["parse_mode"] == "Markdown"
        assert "MeuTitulo" in payload["text"]
        assert "MinhaMensagem" in payload["text"]


# ============================================================
# DiscordChannel
# ============================================================

class TestDiscordChannel:
    def test_send_success(self):
        ch = notif.DiscordChannel("https://discord.com/api/webhooks/123/TOKEN")
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        with patch("notifications.requests.post", return_value=mock_resp) as mock_post:
            result = ch.send("Titulo", "Mensagem")
        assert result is True
        mock_post.assert_called_once()

    def test_send_uses_embed_format(self):
        ch = notif.DiscordChannel("https://discord.com/webhook")
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        with patch("notifications.requests.post", return_value=mock_resp) as mock_post:
            ch.send("MeuTitulo", "MinhaMensagem")
        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert "embeds" in payload
        embed = payload["embeds"][0]
        assert embed["title"] == "MeuTitulo"
        assert embed["description"] == "MinhaMensagem"
        assert embed["color"] == 0x16a34a   # verde

    def test_send_failure_returns_false(self):
        ch = notif.DiscordChannel("https://discord.com/webhook")
        with patch("notifications.requests.post", side_effect=ConnectionError("refused")):
            assert ch.send("T", "M") is False


# ============================================================
# GenericWebhookChannel
# ============================================================

class TestGenericWebhookChannel:
    def test_send_success(self):
        ch = notif.GenericWebhookChannel("https://example.com/hook")
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        with patch("notifications.requests.post", return_value=mock_resp) as mock_post:
            result = ch.send("Titulo", "Mensagem")
        assert result is True

    def test_send_payload_structure(self):
        ch = notif.GenericWebhookChannel("https://example.com/hook")
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        with patch("notifications.requests.post", return_value=mock_resp) as mock_post:
            ch.send("T", "M")
        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert payload["title"] == "T"
        assert payload["message"] == "M"

    def test_send_failure_returns_false(self):
        ch = notif.GenericWebhookChannel("https://example.com/hook")
        with patch("notifications.requests.post", side_effect=Exception("error")):
            assert ch.send("T", "M") is False


# ============================================================
# build_channel_from_env — prioridade dos canais
# ============================================================

class TestBuildChannelFromEnv:
    def test_telegram_takes_priority(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "TOKEN")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "CHAT")
        monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/hook")
        monkeypatch.setenv("BOTCRIPTO_WEBHOOK_URL", "https://example.com/hook")
        ch = notif.build_channel_from_env()
        assert isinstance(ch, notif.TelegramChannel)

    def test_discord_without_telegram(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/hook")
        ch = notif.build_channel_from_env()
        assert isinstance(ch, notif.DiscordChannel)

    def test_generic_without_telegram_or_discord(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        monkeypatch.setenv("BOTCRIPTO_WEBHOOK_URL", "https://example.com/hook")
        ch = notif.build_channel_from_env()
        assert isinstance(ch, notif.GenericWebhookChannel)

    def test_null_channel_when_no_env_vars(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        monkeypatch.delenv("BOTCRIPTO_WEBHOOK_URL", raising=False)
        ch = notif.build_channel_from_env()
        assert isinstance(ch, notif.NullChannel)

    def test_requires_both_telegram_fields(self, monkeypatch):
        """Apenas token sem chat_id nao deve usar Telegram."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "TOKEN")
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        monkeypatch.delenv("BOTCRIPTO_WEBHOOK_URL", raising=False)
        ch = notif.build_channel_from_env()
        assert isinstance(ch, notif.NullChannel)


# ============================================================
# notify_strong_buy / notify_bullish_divergence
# ============================================================

class TestNotifyHelpers:
    def test_notify_strong_buy_calls_channel(self):
        mock_ch = MagicMock()
        mock_ch.send.return_value = True
        with patch.object(notif, "get_channel", return_value=mock_ch):
            result = notif.notify_strong_buy("Bitcoin", 80)
        assert result is True
        mock_ch.send.assert_called_once()
        title, msg = mock_ch.send.call_args.args
        assert "Bitcoin" in title
        assert "80" in title

    def test_notify_strong_buy_includes_signals_summary(self):
        mock_ch = MagicMock()
        mock_ch.send.return_value = True
        with patch.object(notif, "get_channel", return_value=mock_ch):
            notif.notify_strong_buy("Bitcoin", 80, "RSI sobrevenda")
        _, msg = mock_ch.send.call_args.args
        assert "RSI sobrevenda" in msg

    def test_notify_bullish_divergence_rsi(self):
        mock_ch = MagicMock()
        mock_ch.send.return_value = True
        with patch.object(notif, "get_channel", return_value=mock_ch):
            result = notif.notify_bullish_divergence("Ethereum", "RSI")
        assert result is True
        title, _ = mock_ch.send.call_args.args
        assert "Ethereum" in title
        assert "RSI" in title

    def test_notify_bullish_divergence_macd(self):
        mock_ch = MagicMock()
        mock_ch.send.return_value = True
        with patch.object(notif, "get_channel", return_value=mock_ch):
            result = notif.notify_bullish_divergence("Solana", "MACD", "sinal detectado")
        assert result is True


# ============================================================
# dispatch_strong_signals
# ============================================================

class TestDispatchStrongSignals:
    def _item(self, asset_id, score, rsi_div="none", macd_div="none"):
        return {
            "id": asset_id,
            "Ativo": asset_id.title(),
            "Score": score,
            "_score_result": {
                "score": score,
                "divergences": {
                    "rsi": {"type": rsi_div, "description": "desc rsi"},
                    "macd": {"type": macd_div, "description": "desc macd"},
                },
                "signals": {},
            },
        }

    def test_returns_zero_for_empty_list(self):
        assert notif.dispatch_strong_signals([]) == 0

    def test_strong_buy_triggers_notification(self):
        mock_ch = MagicMock()
        mock_ch.send.return_value = True
        with patch.object(notif, "get_channel", return_value=mock_ch):
            with patch("notifications._record_alert", return_value=True):
                sent = notif.dispatch_strong_signals([self._item("bitcoin", 75)])
        assert sent == 1

    def test_below_threshold_does_not_trigger(self):
        mock_ch = MagicMock()
        mock_ch.send.return_value = True
        with patch.object(notif, "get_channel", return_value=mock_ch):
            with patch("notifications._record_alert", return_value=True):
                sent = notif.dispatch_strong_signals([self._item("bitcoin", 71)])
        assert sent == 0

    def test_exactly_at_threshold_triggers(self):
        mock_ch = MagicMock()
        mock_ch.send.return_value = True
        with patch.object(notif, "get_channel", return_value=mock_ch):
            with patch("notifications._record_alert", return_value=True):
                sent = notif.dispatch_strong_signals([self._item("bitcoin", 72)])
        assert sent == 1

    def test_bullish_rsi_divergence_triggers(self):
        mock_ch = MagicMock()
        mock_ch.send.return_value = True
        with patch.object(notif, "get_channel", return_value=mock_ch):
            with patch("notifications._record_alert", return_value=True):
                sent = notif.dispatch_strong_signals([self._item("bitcoin", 50, rsi_div="bullish")])
        assert sent == 1

    def test_bullish_macd_divergence_triggers(self):
        mock_ch = MagicMock()
        mock_ch.send.return_value = True
        with patch.object(notif, "get_channel", return_value=mock_ch):
            with patch("notifications._record_alert", return_value=True):
                sent = notif.dispatch_strong_signals([self._item("bitcoin", 50, macd_div="bullish")])
        assert sent == 1

    def test_both_rsi_and_macd_divergence_trigger_two(self):
        mock_ch = MagicMock()
        mock_ch.send.return_value = True
        with patch.object(notif, "get_channel", return_value=mock_ch):
            with patch("notifications._record_alert", return_value=True):
                item = self._item("bitcoin", 50, rsi_div="bullish", macd_div="bullish")
                sent = notif.dispatch_strong_signals([item])
        assert sent == 2

    def test_deduplication_suppresses_already_sent(self):
        mock_ch = MagicMock()
        mock_ch.send.return_value = True
        with patch.object(notif, "get_channel", return_value=mock_ch):
            with patch("notifications._record_alert", return_value=False):
                sent = notif.dispatch_strong_signals([self._item("bitcoin", 80)])
        assert sent == 0

    def test_multiple_assets_counted_correctly(self):
        mock_ch = MagicMock()
        mock_ch.send.return_value = True
        items = [
            self._item("bitcoin", 80),
            self._item("ethereum", 75),
            self._item("solana", 40),    # abaixo do threshold
        ]
        with patch.object(notif, "get_channel", return_value=mock_ch):
            with patch("notifications._record_alert", return_value=True):
                sent = notif.dispatch_strong_signals(items)
        assert sent == 2

    def test_bearish_divergence_does_not_trigger(self):
        mock_ch = MagicMock()
        mock_ch.send.return_value = True
        with patch.object(notif, "get_channel", return_value=mock_ch):
            with patch("notifications._record_alert", return_value=True):
                sent = notif.dispatch_strong_signals([self._item("bitcoin", 50, rsi_div="bearish")])
        assert sent == 0

    def test_works_when_record_alert_is_none(self):
        """Se _record_alert for None (database indisponivel), deve disparar sem dedup."""
        mock_ch = MagicMock()
        mock_ch.send.return_value = True
        with patch.object(notif, "get_channel", return_value=mock_ch):
            with patch("notifications._record_alert", None):
                sent = notif.dispatch_strong_signals([self._item("bitcoin", 80)])
        # Sem dedup: deve enviar normalmente
        assert sent == 1
