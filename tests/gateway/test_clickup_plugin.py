"""Tests for the ClickUp Chat platform-plugin adapter.

Loaded via the ``load_plugin_adapter`` helper so this lives under
``plugin_adapter_clickup`` in ``sys.modules`` and cannot collide with
sibling platform-plugin tests on the same xdist worker.

Covers: config validation, env-enablement, dedup, the shared-secret webhook
gate, the bot-loop guard, the inbound→read→dispatch path, channel gating,
reply formatting, and the plugin-shape (``register`` / ``_standalone_send``)
contracts. The ClickUp adapter modifies no core Hermes files — everything
routes through the ``platform_registry``.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_clickup = load_plugin_adapter("clickup")

ClickUpAdapter = _clickup.ClickUpAdapter
check_requirements = _clickup.check_requirements
validate_config = _clickup.validate_config
is_connected = _clickup.is_connected
register = _clickup.register
_env_enablement = _clickup._env_enablement
_apply_yaml_config = _clickup._apply_yaml_config
_standalone_send = _clickup._standalone_send
_webhook_ids = _clickup._webhook_ids
_extract_text = _clickup._extract_text
_extract_author_id = _clickup._extract_author_id
MAX_MESSAGE_LENGTH = _clickup.MAX_MESSAGE_LENGTH
DEFAULT_API_BASE = _clickup.DEFAULT_API_BASE
DEFAULT_SECRET_HEADER = _clickup.DEFAULT_SECRET_HEADER


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _cfg(**extra):
    extra.setdefault("api_key", "pk_test")
    extra.setdefault("team_id", "9001")
    return PlatformConfig(enabled=True, extra=extra)


# ---------------------------------------------------------------------------
# 1. Platform enum (plugin-discovered, not bundled)
# ---------------------------------------------------------------------------


def test_platform_enum_resolves_via_plugin_scan():
    from gateway.config import Platform

    p = Platform("clickup")
    assert p.value == "clickup"
    # Identity stability — repeated lookups return the same pseudo-member.
    assert Platform("clickup") is p


# ---------------------------------------------------------------------------
# 2. check_requirements / validate_config / is_connected
# ---------------------------------------------------------------------------


class TestRequirements:
    def test_false_when_aiohttp_unavailable(self, monkeypatch):
        monkeypatch.setenv("CLICKUP_API_KEY", "pk_x")
        monkeypatch.setenv("CLICKUP_TEAM_ID", "9001")
        monkeypatch.setattr(_clickup, "AIOHTTP_AVAILABLE", False)
        assert check_requirements() is False

    def test_false_without_api_key(self, monkeypatch):
        monkeypatch.setattr(_clickup, "AIOHTTP_AVAILABLE", True)
        monkeypatch.delenv("CLICKUP_API_KEY", raising=False)
        monkeypatch.setenv("CLICKUP_TEAM_ID", "9001")
        assert check_requirements() is False

    def test_false_without_team_id(self, monkeypatch):
        monkeypatch.setattr(_clickup, "AIOHTTP_AVAILABLE", True)
        monkeypatch.setenv("CLICKUP_API_KEY", "pk_x")
        monkeypatch.delenv("CLICKUP_TEAM_ID", raising=False)
        assert check_requirements() is False

    def test_true_when_configured(self, monkeypatch):
        monkeypatch.setattr(_clickup, "AIOHTTP_AVAILABLE", True)
        monkeypatch.setenv("CLICKUP_API_KEY", "pk_x")
        monkeypatch.setenv("CLICKUP_TEAM_ID", "9001")
        assert check_requirements() is True

    def test_validate_config_from_extra(self, monkeypatch):
        monkeypatch.delenv("CLICKUP_API_KEY", raising=False)
        monkeypatch.delenv("CLICKUP_TEAM_ID", raising=False)
        assert validate_config(PlatformConfig(enabled=True, extra={})) is False
        assert validate_config(
            PlatformConfig(enabled=True, extra={"api_key": "pk", "team_id": "1"})
        ) is True

    def test_validate_config_from_env(self, monkeypatch):
        monkeypatch.setenv("CLICKUP_API_KEY", "pk_x")
        monkeypatch.setenv("CLICKUP_TEAM_ID", "9001")
        assert validate_config(PlatformConfig(enabled=True, extra={})) is True

    def test_is_connected_mirrors_validate(self, monkeypatch):
        monkeypatch.delenv("CLICKUP_API_KEY", raising=False)
        monkeypatch.delenv("CLICKUP_TEAM_ID", raising=False)
        assert is_connected(_cfg()) is True
        assert is_connected(PlatformConfig(enabled=True, extra={})) is False


# ---------------------------------------------------------------------------
# 3. Adapter init
# ---------------------------------------------------------------------------


class TestInit:
    def test_credentials_from_extra(self, monkeypatch):
        monkeypatch.delenv("CLICKUP_API_KEY", raising=False)
        monkeypatch.delenv("CLICKUP_TEAM_ID", raising=False)
        a = ClickUpAdapter(_cfg(api_key="pk_abc", team_id="42"))
        assert a._api_key == "pk_abc"
        assert a._team_id == "42"

    def test_credentials_env_wins(self, monkeypatch):
        monkeypatch.setenv("CLICKUP_API_KEY", "pk_env")
        monkeypatch.setenv("CLICKUP_TEAM_ID", "777")
        a = ClickUpAdapter(_cfg(api_key="pk_extra", team_id="42"))
        assert a._api_key == "pk_env"
        assert a._team_id == "777"

    def test_default_port_and_host(self, monkeypatch):
        monkeypatch.delenv("CLICKUP_PORT", raising=False)
        monkeypatch.delenv("CLICKUP_HOST", raising=False)
        a = ClickUpAdapter(_cfg())
        assert a._port == _clickup.DEFAULT_PORT
        assert a._host == _clickup.DEFAULT_HOST

    def test_secret_header_default(self):
        a = ClickUpAdapter(_cfg(webhook_secret="s"))
        assert a._secret_header == DEFAULT_SECRET_HEADER

    def test_channel_gating_parsed(self):
        a = ClickUpAdapter(
            _cfg(
                require_mention=True,
                allowed_channels=["c1", "c2"],
                free_response_channels="c3,c4",
            )
        )
        assert a._require_mention is True
        assert a._allowed_channels == {"c1", "c2"}
        assert a._free_response_channels == {"c3", "c4"}

    def test_initial_state(self):
        a = ClickUpAdapter(_cfg())
        assert a._client is None
        assert a._site is None
        assert a._bot_user_id is None


# ---------------------------------------------------------------------------
# 4. Webhook id extraction (query + body, nested envelopes)
# ---------------------------------------------------------------------------


class TestWebhookIds:
    def _req(self, query=None):
        req = MagicMock()
        req.query = query or {}
        return req

    def test_ids_from_body_flat(self):
        ch, msg = _webhook_ids(self._req(), {"channel_id": "C1", "message_id": "M1"})
        assert (ch, msg) == ("C1", "M1")

    def test_ids_from_query(self):
        ch, msg = _webhook_ids(self._req({"channel_id": "C9", "message_id": "M9"}), {})
        assert (ch, msg) == ("C9", "M9")

    def test_ids_from_nested_payload(self):
        body = {"payload": {"channelId": "C2", "id": "M2"}}
        ch, msg = _webhook_ids(self._req(), body)
        assert (ch, msg) == ("C2", "M2")

    def test_comment_id_alias(self):
        ch, msg = _webhook_ids(self._req(), {"channel": "C3", "comment_id": "M3"})
        assert (ch, msg) == ("C3", "M3")

    def test_missing_returns_empty(self):
        ch, msg = _webhook_ids(self._req(), {})
        assert ch == "" and msg == ""


# ---------------------------------------------------------------------------
# 5. Message field extraction
# ---------------------------------------------------------------------------


class TestExtraction:
    def test_text_from_content(self):
        assert _extract_text({"content": "hello"}) == "hello"

    def test_text_from_comment_fragments(self):
        msg = {"comment": [{"text": "foo "}, {"text": "bar"}]}
        assert _extract_text(msg) == "foo bar"

    def test_text_empty(self):
        assert _extract_text({}) == ""

    def test_author_from_user_id(self):
        assert _extract_author_id({"user_id": 123}) == "123"

    def test_author_from_user_object(self):
        assert _extract_author_id({"user": {"id": 55}}) == "55"

    def test_author_missing(self):
        assert _extract_author_id({}) == ""


# ---------------------------------------------------------------------------
# 6. Deduplication
# ---------------------------------------------------------------------------


class TestDedup:
    def test_first_not_duplicate(self):
        a = ClickUpAdapter(_cfg())
        assert a._dedup.is_duplicate("m1") is False

    def test_second_is_duplicate(self):
        a = ClickUpAdapter(_cfg())
        a._dedup.is_duplicate("m1")
        assert a._dedup.is_duplicate("m1") is True

    def test_distinct_ids(self):
        a = ClickUpAdapter(_cfg())
        a._dedup.is_duplicate("m1")
        assert a._dedup.is_duplicate("m2") is False


# ---------------------------------------------------------------------------
# 7. Webhook handler — shared-secret gate
# ---------------------------------------------------------------------------


def _fake_request(*, headers=None, body=b"", query=None):
    req = MagicMock()
    req.headers = headers or {}
    req.query = query or {}
    req.read = AsyncMock(return_value=body)
    return req


class TestSecretGate:
    def test_rejects_missing_secret_header(self):
        a = ClickUpAdapter(_cfg(webhook_secret="topsecret"))
        a._process_webhook = AsyncMock()  # type: ignore[assignment]
        resp = _run(a._handle_webhook(_fake_request(headers={})))
        assert resp.status == 401
        a._process_webhook.assert_not_called()

    def test_rejects_wrong_secret(self):
        a = ClickUpAdapter(_cfg(webhook_secret="topsecret"))
        a._process_webhook = AsyncMock()  # type: ignore[assignment]
        req = _fake_request(headers={DEFAULT_SECRET_HEADER: "nope"})
        resp = _run(a._handle_webhook(req))
        assert resp.status == 401
        a._process_webhook.assert_not_called()

    def test_accepts_correct_secret(self):
        a = ClickUpAdapter(_cfg(webhook_secret="topsecret"))
        a._process_webhook = AsyncMock()  # type: ignore[assignment]
        body = json.dumps({"channel_id": "C1", "message_id": "M1"}).encode()
        req = _fake_request(headers={DEFAULT_SECRET_HEADER: "topsecret"}, body=body)
        resp = _run(a._handle_webhook(req))
        assert resp.status == 200
        a._process_webhook.assert_awaited_once()

    def test_custom_secret_header(self):
        a = ClickUpAdapter(_cfg(webhook_secret="s", webhook_secret_header="X-Custom"))
        a._process_webhook = AsyncMock()  # type: ignore[assignment]
        req = _fake_request(headers={"X-Custom": "s"}, body=b"{}")
        resp = _run(a._handle_webhook(req))
        assert resp.status == 200
        a._process_webhook.assert_awaited_once()


# ---------------------------------------------------------------------------
# 8. Inbound → read → dispatch path
# ---------------------------------------------------------------------------


class TestInboundDispatch:
    def _adapter(self, **extra):
        a = ClickUpAdapter(_cfg(**extra))
        a._client = MagicMock()
        a.handle_message = AsyncMock()  # type: ignore[assignment]
        return a

    def test_reads_body_and_dispatches(self):
        a = self._adapter()
        a._bot_user_id = "999"
        a._client.get_message = AsyncMock(
            return_value={"id": "M1", "content": "hi bot", "user_id": "555"}
        )
        _run(a._process_webhook(MagicMock(query={}), {"channel_id": "C1", "message_id": "M1"}))
        a.handle_message.assert_awaited_once()
        event = a.handle_message.call_args[0][0]
        assert event.text == "hi bot"
        assert event.source.chat_id == "C1"
        assert event.source.user_id == "555"
        assert event.message_id == "M1"

    def test_skips_when_ids_missing(self):
        a = self._adapter()
        a._client.get_message = AsyncMock()
        _run(a._process_webhook(MagicMock(query={}), {}))
        a._client.get_message.assert_not_called()
        a.handle_message.assert_not_called()

    def test_dedup_blocks_retry(self):
        a = self._adapter()
        a._client.get_message = AsyncMock(
            return_value={"id": "M1", "content": "hi", "user_id": "5"}
        )
        body = {"channel_id": "C1", "message_id": "M1"}
        _run(a._process_webhook(MagicMock(query={}), body))
        _run(a._process_webhook(MagicMock(query={}), body))
        assert a.handle_message.await_count == 1

    def test_bot_loop_guard_skips_own_message(self):
        a = self._adapter()
        a._bot_user_id = "555"
        a._client.get_message = AsyncMock(
            return_value={"id": "M1", "content": "my own reply", "user_id": "555"}
        )
        _run(a._process_webhook(MagicMock(query={}), {"channel_id": "C1", "message_id": "M1"}))
        a.handle_message.assert_not_called()

    def test_allowed_channels_whitelist(self):
        a = self._adapter(allowed_channels=["GOOD"])
        a._client.get_message = AsyncMock(
            return_value={"id": "M1", "content": "hi", "user_id": "5"}
        )
        _run(a._process_webhook(MagicMock(query={}), {"channel_id": "BAD", "message_id": "M1"}))
        a._client.get_message.assert_not_called()
        a.handle_message.assert_not_called()

    def test_missing_message_not_dispatched(self):
        a = self._adapter()
        a._client.get_message = AsyncMock(return_value=None)
        _run(a._process_webhook(MagicMock(query={}), {"channel_id": "C1", "message_id": "M1"}))
        a.handle_message.assert_not_called()

    def test_require_mention_skips_unmentioned(self):
        a = self._adapter(require_mention=True)
        a._bot_user_id = "999"
        a._client.get_message = AsyncMock(
            return_value={"id": "M1", "content": "no mention here", "user_id": "5"}
        )
        _run(a._process_webhook(MagicMock(query={}), {"channel_id": "C1", "message_id": "M1"}))
        a.handle_message.assert_not_called()

    def test_require_mention_dispatches_when_mentioned(self):
        a = self._adapter(require_mention=True)
        a._bot_user_id = "999"
        a._client.get_message = AsyncMock(
            return_value={"id": "M1", "content": "hey @999 do this", "user_id": "5"}
        )
        _run(a._process_webhook(MagicMock(query={}), {"channel_id": "C1", "message_id": "M1"}))
        a.handle_message.assert_awaited_once()
        # @mention stripped from the text the agent sees.
        assert "@999" not in a.handle_message.call_args[0][0].text

    def test_free_response_channel_bypasses_mention(self):
        a = self._adapter(require_mention=True, free_response_channels=["C1"])
        a._bot_user_id = "999"
        a._client.get_message = AsyncMock(
            return_value={"id": "M1", "content": "no mention", "user_id": "5"}
        )
        _run(a._process_webhook(MagicMock(query={}), {"channel_id": "C1", "message_id": "M1"}))
        a.handle_message.assert_awaited_once()

    def test_channel_prompt_attached(self):
        a = self._adapter(channel_prompts={"C1": "Be terse."})
        a._client.get_message = AsyncMock(
            return_value={"id": "M1", "content": "hi", "user_id": "5"}
        )
        _run(a._process_webhook(MagicMock(query={}), {"channel_id": "C1", "message_id": "M1"}))
        assert a.handle_message.call_args[0][0].channel_prompt == "Be terse."


# ---------------------------------------------------------------------------
# 9. Outbound reply formatting
# ---------------------------------------------------------------------------


class TestSend:
    def _adapter(self):
        a = ClickUpAdapter(_cfg())
        a._client = MagicMock()
        a._client.post_reply = AsyncMock(return_value={"id": "R1"})
        a._client.post_channel_message = AsyncMock(return_value={"id": "T1"})
        return a

    def test_send_without_client_fails(self):
        a = ClickUpAdapter(_cfg())
        result = _run(a.send("C1", "hi"))
        assert result.success is False
        assert "not initialized" in result.error.lower()

    def test_reply_uses_replies_endpoint(self):
        a = self._adapter()
        result = _run(a.send("C1", "hello", reply_to="M1"))
        assert result.success is True
        assert result.message_id == "R1"
        a._client.post_reply.assert_awaited_once_with("M1", "hello")
        a._client.post_channel_message.assert_not_called()

    def test_top_level_when_no_reply_to(self):
        a = self._adapter()
        result = _run(a.send("C1", "hello"))
        assert result.success is True
        assert result.message_id == "T1"
        a._client.post_channel_message.assert_awaited_once_with("C1", "hello")
        a._client.post_reply.assert_not_called()

    def test_send_error_is_retryable(self):
        a = self._adapter()
        a._client.post_channel_message = AsyncMock(side_effect=RuntimeError("boom"))
        result = _run(a.send("C1", "hello"))
        assert result.success is False
        assert result.retryable is True

    def test_long_message_chunked(self):
        a = self._adapter()
        long = "x" * (MAX_MESSAGE_LENGTH + 100)
        _run(a.send("C1", long))
        # First chunk top-level, remaining chunk(s) also top-level (no reply_to).
        assert a._client.post_channel_message.await_count >= 2

    def test_send_typing_noop(self):
        a = ClickUpAdapter(_cfg())
        _run(a.send_typing("C1"))  # must not raise

    def test_get_chat_info(self):
        a = ClickUpAdapter(_cfg())
        info = _run(a.get_chat_info("C1"))
        assert info["chat_id"] == "C1"
        assert info["type"] == "channel"


# ---------------------------------------------------------------------------
# 10. REST client — auth header & rate limit
# ---------------------------------------------------------------------------


class TestClient:
    def test_auth_header_has_no_bearer_prefix(self):
        client = _clickup._ClickUpClient("pk_token", "9001")
        headers = client._headers()
        assert headers["Authorization"] == "pk_token"
        assert "Bearer" not in headers["Authorization"]

    def test_workspace_base_url(self):
        client = _clickup._ClickUpClient("pk", "9001")
        assert client._workspace_base == f"{DEFAULT_API_BASE}/workspaces/9001"

    def test_get_message_matches_by_id(self):
        client = _clickup._ClickUpClient("pk", "9001")
        client.list_channel_messages = AsyncMock(
            return_value=[{"id": "A"}, {"id": "B", "content": "found"}]
        )
        msg = _run(client.get_message("C1", "B"))
        assert msg["content"] == "found"

    def test_get_message_none_when_absent(self):
        client = _clickup._ClickUpClient("pk", "9001")
        client.list_channel_messages = AsyncMock(return_value=[{"id": "A"}])
        assert _run(client.get_message("C1", "Z")) is None


# ---------------------------------------------------------------------------
# 11. connect() / disconnect()
# ---------------------------------------------------------------------------


class TestConnect:
    def test_connect_fails_without_aiohttp(self, monkeypatch):
        monkeypatch.setattr(_clickup, "AIOHTTP_AVAILABLE", False)
        a = ClickUpAdapter(_cfg(webhook_secret="s"))
        assert _run(a.connect()) is False
        assert a.has_fatal_error is True

    def test_connect_fails_without_credentials(self, monkeypatch):
        monkeypatch.setattr(_clickup, "AIOHTTP_AVAILABLE", True)
        monkeypatch.delenv("CLICKUP_API_KEY", raising=False)
        monkeypatch.delenv("CLICKUP_TEAM_ID", raising=False)
        a = ClickUpAdapter(PlatformConfig(enabled=True, extra={"webhook_secret": "s"}))
        assert _run(a.connect()) is False
        assert a.has_fatal_error is True

    def test_connect_refuses_without_secret(self, monkeypatch):
        monkeypatch.setattr(_clickup, "AIOHTTP_AVAILABLE", True)
        monkeypatch.delenv("CLICKUP_WEBHOOK_SECRET", raising=False)
        a = ClickUpAdapter(_cfg())  # no webhook_secret
        assert _run(a.connect()) is False
        assert a.has_fatal_error is True
        assert a._fatal_error_code == "config_missing"


# ---------------------------------------------------------------------------
# 12. _env_enablement()
# ---------------------------------------------------------------------------


class TestEnvEnablement:
    def test_none_without_credentials(self, monkeypatch):
        monkeypatch.delenv("CLICKUP_API_KEY", raising=False)
        monkeypatch.delenv("CLICKUP_TEAM_ID", raising=False)
        assert _env_enablement() is None

    def test_seeds_credentials(self, monkeypatch):
        monkeypatch.setenv("CLICKUP_API_KEY", "pk_env")
        monkeypatch.setenv("CLICKUP_TEAM_ID", "777")
        seed = _env_enablement()
        assert seed["api_key"] == "pk_env"
        assert seed["team_id"] == "777"

    def test_seeds_secret_and_gating(self, monkeypatch):
        monkeypatch.setenv("CLICKUP_API_KEY", "pk")
        monkeypatch.setenv("CLICKUP_TEAM_ID", "1")
        monkeypatch.setenv("CLICKUP_WEBHOOK_SECRET", "sec")
        monkeypatch.setenv("CLICKUP_REQUIRE_MENTION", "true")
        monkeypatch.setenv("CLICKUP_ALLOWED_CHANNELS", "c1,c2")
        seed = _env_enablement()
        assert seed["webhook_secret"] == "sec"
        assert seed["require_mention"] is True
        assert set(seed["allowed_channels"]) == {"c1", "c2"}

    def test_home_channel(self, monkeypatch):
        monkeypatch.setenv("CLICKUP_API_KEY", "pk")
        monkeypatch.setenv("CLICKUP_TEAM_ID", "1")
        monkeypatch.setenv("CLICKUP_HOME_CHANNEL", "C-home")
        monkeypatch.setenv("CLICKUP_HOME_CHANNEL_NAME", "Ops")
        seed = _env_enablement()
        assert seed["home_channel"]["chat_id"] == "C-home"
        assert seed["home_channel"]["name"] == "Ops"


# ---------------------------------------------------------------------------
# 12b. _apply_yaml_config() — YAML→extra bridge
# ---------------------------------------------------------------------------


class TestApplyYamlConfig:
    def test_bridges_allowed_channels(self):
        seed = _apply_yaml_config({}, {"allowed_channels": ["c1", "c2"]})
        assert seed["allowed_channels"] == ["c1", "c2"]

    def test_bridges_secret_and_port(self):
        seed = _apply_yaml_config({}, {"webhook_secret": "s", "port": 9000})
        assert seed["webhook_secret"] == "s"
        assert seed["port"] == 9000

    def test_returns_none_when_nothing_to_bridge(self):
        assert _apply_yaml_config({}, {"require_mention": True}) is None


# ---------------------------------------------------------------------------
# 13. _standalone_send()
# ---------------------------------------------------------------------------


class TestStandaloneSend:
    def _pconfig(self):
        p = MagicMock()
        p.extra = {"api_key": "pk", "team_id": "9001"}
        return p

    def test_errors_without_credentials(self, monkeypatch):
        monkeypatch.delenv("CLICKUP_API_KEY", raising=False)
        monkeypatch.delenv("CLICKUP_TEAM_ID", raising=False)
        p = MagicMock()
        p.extra = {}
        result = _run(_standalone_send(p, "C1", "hi"))
        assert "error" in result

    def test_errors_without_channel(self):
        result = _run(_standalone_send(self._pconfig(), "", "hi"))
        assert "error" in result

    def test_posts_top_level(self):
        with patch.object(_clickup, "_ClickUpClient") as MockClient:
            inst = MockClient.return_value
            inst.post_channel_message = AsyncMock(return_value={"id": "T9"})
            inst.post_reply = AsyncMock()
            result = _run(_standalone_send(self._pconfig(), "C1", "hello"))
        assert result["success"] is True
        assert result["platform"] == "clickup"
        assert result["message_id"] == "T9"
        inst.post_channel_message.assert_awaited_once()

    def test_posts_reply_when_thread_id(self):
        with patch.object(_clickup, "_ClickUpClient") as MockClient:
            inst = MockClient.return_value
            inst.post_reply = AsyncMock(return_value={"id": "R9"})
            inst.post_channel_message = AsyncMock()
            result = _run(_standalone_send(self._pconfig(), "C1", "hello", thread_id="M1"))
        assert result["success"] is True
        inst.post_reply.assert_awaited_once_with("M1", "hello")

    def test_error_on_failure(self):
        with patch.object(_clickup, "_ClickUpClient") as MockClient:
            inst = MockClient.return_value
            inst.post_channel_message = AsyncMock(side_effect=RuntimeError("403"))
            result = _run(_standalone_send(self._pconfig(), "C1", "hi"))
        assert "error" in result


# ---------------------------------------------------------------------------
# 14. register() — plugin-side metadata
# ---------------------------------------------------------------------------


def test_register_calls_register_platform():
    ctx = MagicMock()
    register(ctx)
    ctx.register_platform.assert_called_once()
    kwargs = ctx.register_platform.call_args.kwargs
    assert kwargs["name"] == "clickup"
    assert kwargs["label"] == "ClickUp Chat"
    assert kwargs["required_env"] == ["CLICKUP_API_KEY", "CLICKUP_TEAM_ID"]
    assert kwargs["allowed_users_env"] == "CLICKUP_ALLOWED_USERS"
    assert kwargs["allow_all_env"] == "CLICKUP_ALLOW_ALL_USERS"
    assert kwargs["cron_deliver_env_var"] == "CLICKUP_HOME_CHANNEL"
    assert kwargs["max_message_length"] == MAX_MESSAGE_LENGTH
    assert callable(kwargs["check_fn"])
    assert callable(kwargs["validate_config"])
    assert callable(kwargs["is_connected"])
    assert callable(kwargs["env_enablement_fn"])
    assert callable(kwargs["standalone_sender_fn"])
    assert callable(kwargs["adapter_factory"])
    assert callable(kwargs["apply_yaml_config_fn"])
    assert "clickup" in kwargs["platform_hint"].lower()


def test_adapter_factory_returns_clickup_adapter():
    ctx = MagicMock()
    register(ctx)
    factory = ctx.register_platform.call_args.kwargs["adapter_factory"]
    adapter = factory(_cfg())
    assert isinstance(adapter, ClickUpAdapter)
