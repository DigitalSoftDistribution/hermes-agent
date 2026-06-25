"""ClickUp Chat platform adapter (Hermes plugin).

Runs an aiohttp webhook server that receives ClickUp "Chat webhook
Automation" POSTs and relays messages between ClickUp Chat channels and the
Hermes agent. Outbound replies use the ClickUp Chat v3 REST API.

This adapter ships as a Hermes platform plugin under
``plugins/platforms/clickup/``. The Hermes plugin loader scans the
directory at startup, calls :func:`register`, and the platform becomes
available to ``gateway/run.py`` and ``tools/send_message_tool`` through the
platform registry — no edits to core Hermes files required.

ClickUp Chat specifics
-----------------------
* **Auth.** ClickUp personal tokens (``pk_...``) authenticate with a bare
  ``Authorization: <token>`` header — **no** ``Bearer`` prefix.
* **Base URL.** ``https://api.clickup.com/api/v3/workspaces/{team_id}``.
* **Thin webhooks.** A Chat webhook Automation POST carries only the
  message/comment id and the channel id (in the URL query and/or the JSON
  body — both are supported). It does NOT carry the message text, so the
  adapter reads the body back via
  ``GET /chat/channels/{channel_id}/messages`` before dispatching.
* **No HMAC.** ClickUp chat automations cannot sign requests, so the
  adapter verifies a shared secret from a configurable header
  (``X-Hermes-Secret`` by default) against ``CLICKUP_WEBHOOK_SECRET``.
* **Retries.** ClickUp re-delivers failed webhook deliveries for ~75
  minutes, so inbound messages are deduplicated on the message id with a
  TTL cache sized to outlast that window.
* **Bot-loop guard.** The token's own user id is fetched once via
  ``GET /user``; messages authored by that user are ignored.
* **Rate limits.** Standard plans allow ~100 requests/minute. On HTTP 429
  the adapter honors ``X-RateLimit-Reset`` (epoch seconds) before retrying.

Configuration in config.yaml::

    platforms:
      clickup:
        enabled: true
        extra:
          api_key: "pk_..."             # or CLICKUP_API_KEY env var
          team_id: "9008..."            # or CLICKUP_TEAM_ID env var
          port: 8650                    # or CLICKUP_PORT env var
          host: "127.0.0.1"             # or CLICKUP_HOST env var
          webhook_secret: "s3cr3t"      # or CLICKUP_WEBHOOK_SECRET env var
          require_mention: false        # or CLICKUP_REQUIRE_MENTION env var
          allowed_channels: ["6-901..."]    # whitelist (empty = all)
          free_response_channels: ["6-901..."]
          channel_prompts:
            "6-901...": "Be terse in this channel."

Environment variables (env wins over config.yaml ``extra``):

    CLICKUP_API_KEY                Personal API token, pk_... (required)
    CLICKUP_TEAM_ID                Workspace (team) id (required)
    CLICKUP_PORT                   Webhook listen port (default: 8650)
    CLICKUP_HOST                   Webhook bind host (default: 127.0.0.1)
    CLICKUP_WEBHOOK_SECRET         Shared secret for the webhook header
    CLICKUP_WEBHOOK_SECRET_HEADER  Header carrying the secret (default: X-Hermes-Secret)
    CLICKUP_REQUIRE_MENTION        Require @mention in channels (default: false)
    CLICKUP_ALLOWED_CHANNELS       Channel-id whitelist (comma-separated)
    CLICKUP_FREE_RESPONSE_CHANNELS Channels that respond without @mention
    CLICKUP_ALLOWED_USERS          Allowed user ids (comma-separated)
    CLICKUP_ALLOW_ALL_USERS        Allow any user (dev only)
    CLICKUP_HOME_CHANNEL           Default channel id for cron delivery
    CLICKUP_HOME_CHANNEL_NAME      Human label for the home channel
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

try:
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    resolve_channel_prompt,
)
from gateway.platforms.helpers import MessageDeduplicator

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────

DEFAULT_API_BASE = "https://api.clickup.com/api/v3"
DEFAULT_PORT = 8650
DEFAULT_HOST = "127.0.0.1"
DEFAULT_WEBHOOK_PATH = "/clickup/webhook"
DEFAULT_SECRET_HEADER = "X-Hermes-Secret"
WEBHOOK_BODY_MAX_BYTES = 1_048_576  # 1 MiB — chat webhooks are tiny
# ClickUp re-delivers failed webhooks for ~75 min. Keep the dedup window
# comfortably past that so a late retry of an already-handled message is
# still recognised as a duplicate.
DEDUP_TTL_SECONDS = 5400  # 90 min
DEDUP_MAX_SIZE = 5000
# ClickUp chat message content cap (generous; the API itself accepts large
# bodies but very long replies are unreadable in chat). The base
# smart-chunker splits anything longer across multiple messages.
MAX_MESSAGE_LENGTH = 30000
# Max attempts when ClickUp returns 429 before giving up on a request.
RATE_LIMIT_MAX_RETRIES = 2
# Cap any single rate-limit sleep so a hostile/garbage Reset header can't
# stall the gateway indefinitely.
RATE_LIMIT_MAX_SLEEP_SECONDS = 30.0


# ── Config helpers ────────────────────────────────────────────────────────────

def _csv_set(value: Any) -> set[str]:
    """Normalize a comma-separated string or a list into a set of strings."""
    if not value:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(v).strip() for v in value if str(v).strip()}
    return {part.strip() for part in str(value).split(",") if part.strip()}


def _truthy(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _coerce_port(value: Any, *, default: int = DEFAULT_PORT) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ── ClickUp Chat REST client ──────────────────────────────────────────────────

class _ClickUpClient:
    """Thin async wrapper around the ClickUp Chat v3 REST API.

    Uses ``aiohttp`` directly to avoid an SDK dependency. Personal tokens
    authenticate with a bare ``Authorization: <token>`` header (no Bearer
    prefix). Handles HTTP 429 by honoring ``X-RateLimit-Reset``.
    """

    def __init__(
        self,
        api_key: str,
        team_id: str,
        *,
        api_base: str = DEFAULT_API_BASE,
        timeout: float = 20.0,
    ) -> None:
        self._api_key = (api_key or "").strip()
        self._team_id = str(team_id or "").strip()
        self._api_base = api_base.rstrip("/")
        self._timeout = timeout

    @property
    def _workspace_base(self) -> str:
        return f"{self._api_base}/workspaces/{self._team_id}"

    def _headers(self) -> Dict[str, str]:
        # ClickUp personal tokens (pk_...) use a bare Authorization header.
        return {
            "Authorization": self._api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def _request(
        self,
        method: str,
        url: str,
        *,
        json_body: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> Any:
        """Perform a request, retrying on 429 with X-RateLimit-Reset backoff."""
        import aiohttp

        timeout = aiohttp.ClientTimeout(total=self._timeout)
        attempt = 0
        while True:
            async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
                async with session.request(
                    method,
                    url,
                    headers=self._headers(),
                    json=json_body,
                    params=params,
                ) as resp:
                    if resp.status == 429 and attempt < RATE_LIMIT_MAX_RETRIES:
                        await self._sleep_for_rate_limit(resp)
                        attempt += 1
                        continue
                    if resp.status >= 400:
                        body = await resp.text()
                        raise RuntimeError(
                            f"ClickUp {method} {url} -> {resp.status}: {body[:300]}"
                        )
                    if resp.status == 204 or resp.content_length == 0:
                        return None
                    try:
                        return await resp.json()
                    except Exception:
                        return None

    @staticmethod
    async def _sleep_for_rate_limit(resp: Any) -> None:
        """Sleep until the rate-limit window resets (clamped)."""
        import time

        reset_raw = resp.headers.get("X-RateLimit-Reset", "")
        delay = 1.0
        try:
            reset_epoch = float(reset_raw)
            # ClickUp returns an absolute epoch (seconds). Convert to a
            # relative delay; tolerate a header that is already relative.
            delta = reset_epoch - time.time()
            delay = delta if 0 < delta <= RATE_LIMIT_MAX_SLEEP_SECONDS else (
                reset_epoch if 0 < reset_epoch <= RATE_LIMIT_MAX_SLEEP_SECONDS else 1.0
            )
        except (TypeError, ValueError):
            delay = 1.0
        delay = max(0.5, min(delay, RATE_LIMIT_MAX_SLEEP_SECONDS))
        logger.warning("[clickup] rate limited (429) — sleeping %.1fs", delay)
        await asyncio.sleep(delay)

    async def get_authorized_user_id(self) -> Optional[str]:
        """Fetch the token owner's user id (for the bot-loop guard).

        ``GET /user`` lives at the v2 base; we derive it from the configured
        v3 base by swapping the version segment so a custom api_base still
        works.
        """
        user_url = f"{self._api_base.rsplit('/api/', 1)[0]}/api/v2/user"
        try:
            data = await self._request("GET", user_url)
        except Exception as exc:
            logger.debug("[clickup] get_authorized_user_id failed: %s", exc)
            return None
        if not isinstance(data, dict):
            return None
        user = data.get("user") or {}
        uid = user.get("id")
        return str(uid) if uid is not None else None

    async def list_channel_messages(
        self, channel_id: str, *, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Return recent messages for a channel (newest first)."""
        url = f"{self._workspace_base}/chat/channels/{channel_id}/messages"
        data = await self._request("GET", url, params={"limit": str(limit)})
        if isinstance(data, dict):
            items = data.get("data")
            if isinstance(items, list):
                return items
        if isinstance(data, list):
            return data
        return []

    async def get_message(
        self, channel_id: str, message_id: str
    ) -> Optional[Dict[str, Any]]:
        """Find a single message in a channel by id.

        ClickUp has no get-single-chat-message endpoint, so we page the
        channel's recent messages and match on id. The webhook fires
        promptly, so the target is among the most recent messages.
        """
        messages = await self.list_channel_messages(channel_id)
        target = str(message_id)
        for msg in messages:
            if str(msg.get("id")) == target:
                return msg
        return None

    async def post_channel_message(
        self, channel_id: str, content: str
    ) -> Dict[str, Any]:
        """Create a top-level message in a channel."""
        url = f"{self._workspace_base}/chat/channels/{channel_id}/messages"
        body = {"type": "message", "content_format": "text/md", "content": content}
        data = await self._request("POST", url, json_body=body)
        return data if isinstance(data, dict) else {}

    async def post_reply(self, message_id: str, content: str) -> Dict[str, Any]:
        """Create an in-thread reply to a message."""
        url = f"{self._workspace_base}/chat/messages/{message_id}/replies"
        body = {"type": "message", "content_format": "text/md", "content": content}
        data = await self._request("POST", url, json_body=body)
        return data if isinstance(data, dict) else {}


# ── Message field extraction ──────────────────────────────────────────────────

def _extract_text(message: Dict[str, Any]) -> str:
    """Pull the human-readable text out of a ClickUp chat message object."""
    for key in ("content", "comment_text", "text_content", "text"):
        val = message.get(key)
        if isinstance(val, str) and val.strip():
            return val
    # ClickUp comment-style payloads carry a list of comment fragments.
    comment = message.get("comment")
    if isinstance(comment, list):
        parts = [str(frag.get("text", "")) for frag in comment if isinstance(frag, dict)]
        joined = "".join(parts).strip()
        if joined:
            return joined
    return ""


def _extract_author_id(message: Dict[str, Any]) -> str:
    """Pull the author user id out of a ClickUp chat message object."""
    for key in ("user_id", "userid"):
        val = message.get(key)
        if val is not None and str(val).strip():
            return str(val)
    user = message.get("user")
    if isinstance(user, dict) and user.get("id") is not None:
        return str(user.get("id"))
    return ""


def _extract_author_name(message: Dict[str, Any]) -> str:
    user = message.get("user")
    if isinstance(user, dict):
        return str(user.get("username") or user.get("email") or "")
    return ""


def _webhook_ids(request: Any, body: Dict[str, Any]) -> Tuple[str, str]:
    """Resolve ``(channel_id, message_id)`` from the webhook URL query and body.

    ClickUp Chat webhook Automations are configurable; the message/comment id
    and channel id may arrive in the query string and/or the JSON body. We
    support both, preferring the body and falling back to the query.
    """
    query = dict(request.query) if request is not None else {}

    def _first(*keys: str) -> str:
        for src in (body, query):
            for key in keys:
                val = src.get(key)
                if val is not None and str(val).strip():
                    return str(val).strip()
        return ""

    # ClickUp payloads nest the relevant ids under a few shapes; check the
    # common flat keys plus a nested ``payload``/``data`` envelope.
    nested = {}
    for env_key in ("payload", "data", "message", "comment"):
        candidate = body.get(env_key)
        if isinstance(candidate, dict):
            nested.update(candidate)

    def _first_nested(*keys: str) -> str:
        for src in (body, nested, query):
            for key in keys:
                val = src.get(key)
                if val is not None and str(val).strip():
                    return str(val).strip()
        return ""

    channel_id = _first_nested("channel_id", "channelId", "channel")
    message_id = _first_nested("message_id", "messageId", "comment_id", "commentId", "id")
    return channel_id, message_id


# ── Plugin gates ──────────────────────────────────────────────────────────────

def check_requirements() -> bool:
    """Plugin gate: require credentials AND aiohttp at runtime.

    Reads env vars directly to avoid the cost of a full
    ``load_gateway_config()`` on every pre-flight check.
    """
    if not AIOHTTP_AVAILABLE:
        return False
    if not os.getenv("CLICKUP_API_KEY", "").strip():
        return False
    if not os.getenv("CLICKUP_TEAM_ID", "").strip():
        return False
    return True


def validate_config(config) -> bool:
    """Return True when the config has both an API key and a team id."""
    extra = getattr(config, "extra", {}) or {}
    has_key = bool(os.getenv("CLICKUP_API_KEY") or extra.get("api_key"))
    has_team = bool(os.getenv("CLICKUP_TEAM_ID") or extra.get("team_id"))
    return has_key and has_team


def is_connected(config) -> bool:
    """Surface in ``hermes status`` even before the adapter is instantiated."""
    return validate_config(config)


# ── Adapter ─────────────────────────────────────────────────────────────────

class ClickUpAdapter(BasePlatformAdapter):
    """ClickUp Chat gateway adapter (webhook inbound + REST outbound)."""

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("clickup"))
        extra = config.extra or {}

        # Credentials
        self._api_key: str = (
            os.getenv("CLICKUP_API_KEY") or extra.get("api_key", "")
        ).strip()
        self._team_id: str = str(
            os.getenv("CLICKUP_TEAM_ID") or extra.get("team_id", "")
        ).strip()
        self._api_base: str = (
            os.getenv("CLICKUP_API_BASE") or extra.get("api_base", DEFAULT_API_BASE)
        ).rstrip("/")

        # Webhook server — bind to localhost by default; operators put a
        # reverse proxy / tunnel in front for ClickUp to reach.
        self._host: str = os.getenv("CLICKUP_HOST") or extra.get("host", DEFAULT_HOST)
        self._port: int = _coerce_port(
            os.getenv("CLICKUP_PORT") or extra.get("port", DEFAULT_PORT)
        )
        self._webhook_path: str = extra.get("webhook_path", DEFAULT_WEBHOOK_PATH)

        # Shared-secret verification (ClickUp chat automations have no HMAC).
        self._webhook_secret: str = (
            os.getenv("CLICKUP_WEBHOOK_SECRET") or extra.get("webhook_secret", "")
        ).strip()
        self._secret_header: str = (
            os.getenv("CLICKUP_WEBHOOK_SECRET_HEADER")
            or extra.get("webhook_secret_header", DEFAULT_SECRET_HEADER)
        )

        # Channel gating (mirrors the Mattermost/Discord channel semantics).
        self._require_mention: bool = _truthy(
            os.getenv("CLICKUP_REQUIRE_MENTION")
            if os.getenv("CLICKUP_REQUIRE_MENTION") is not None
            else extra.get("require_mention"),
            default=False,
        )
        self._allowed_channels: set[str] = _csv_set(
            os.getenv("CLICKUP_ALLOWED_CHANNELS")
        ) | _csv_set(extra.get("allowed_channels"))
        self._free_response_channels: set[str] = _csv_set(
            os.getenv("CLICKUP_FREE_RESPONSE_CHANNELS")
        ) | _csv_set(extra.get("free_response_channels"))

        # Runtime state
        self._client: Optional[_ClickUpClient] = None
        self._app = None  # aiohttp.web.Application
        self._runner = None  # aiohttp.web.AppRunner
        self._site = None  # aiohttp.web.TCPSite
        self._bot_user_id: Optional[str] = None
        self._dedup = MessageDeduplicator(max_size=DEDUP_MAX_SIZE, ttl_seconds=DEDUP_TTL_SECONDS)

    # -- Connection lifecycle -----------------------------------------------

    async def connect(self) -> bool:
        if not AIOHTTP_AVAILABLE:
            self._set_fatal_error(
                "missing_dep",
                "aiohttp is required for the ClickUp adapter — install with `pip install aiohttp`",
                retryable=False,
            )
            return False

        if not self._api_key or not self._team_id:
            self._set_fatal_error(
                "config_missing",
                "CLICKUP_API_KEY and CLICKUP_TEAM_ID must be set",
                retryable=False,
            )
            return False

        if not self._webhook_secret:
            # A missing secret would accept any unauthenticated POST. Refuse
            # to start rather than expose an open webhook.
            self._set_fatal_error(
                "config_missing",
                "CLICKUP_WEBHOOK_SECRET must be set — the ClickUp webhook is "
                "otherwise unauthenticated (chat automations have no HMAC)",
                retryable=False,
            )
            return False

        self._client = _ClickUpClient(self._api_key, self._team_id, api_base=self._api_base)

        # Best-effort: fetch our own user id so we can filter self-authored
        # messages. If it fails (offline tests, transient 5xx) we fall back
        # to not filtering; ClickUp does not echo our replies back as new
        # inbound webhooks in normal operation.
        try:
            self._bot_user_id = await self._client.get_authorized_user_id()
        except Exception as exc:
            logger.debug("[clickup] get_authorized_user_id failed: %s", exc)
            self._bot_user_id = None

        self._app = web.Application(client_max_size=WEBHOOK_BODY_MAX_BYTES)
        self._app.router.add_post(self._webhook_path, self._handle_webhook)
        self._app.router.add_get(f"{self._webhook_path}/health", self._handle_health)

        self._runner = web.AppRunner(self._app)
        try:
            await self._runner.setup()
            self._site = web.TCPSite(self._runner, self._host, self._port)
            await self._site.start()
        except OSError as exc:
            self._set_fatal_error(
                "bind_failed",
                f"Could not bind ClickUp webhook on {self._host}:{self._port}: {exc}",
                retryable=True,
            )
            return False

        self._mark_connected()
        logger.info(
            "[clickup] Webhook listening on %s:%s%s",
            self._host,
            self._port,
            self._webhook_path,
        )
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()
        if self._site is not None:
            try:
                await self._site.stop()
            except Exception:
                pass
            self._site = None
        if self._runner is not None:
            try:
                await self._runner.cleanup()
            except Exception:
                pass
            self._runner = None
        self._app = None
        self._client = None
        self._dedup.clear()
        logger.info("[clickup] Disconnected")

    # -- Webhook handlers ----------------------------------------------------

    async def _handle_health(self, request) -> Any:
        return web.json_response({"status": "ok", "platform": "clickup"})

    async def _handle_webhook(self, request) -> Any:
        # Shared-secret verification — ClickUp chat automations can attach a
        # custom header but cannot sign the body, so this is our only
        # authenticity gate. Constant-time compare avoids a timing oracle.
        import hmac

        provided = request.headers.get(self._secret_header, "")
        if not self._webhook_secret or not hmac.compare_digest(
            provided, self._webhook_secret
        ):
            return web.Response(status=401, text="invalid secret")

        try:
            raw = await request.read()
        except Exception as exc:
            logger.debug("[clickup] read failed: %s", exc)
            return web.Response(status=400, text="bad request")
        if len(raw) > WEBHOOK_BODY_MAX_BYTES:
            return web.Response(status=413, text="payload too large")

        body: Dict[str, Any] = {}
        if raw:
            try:
                import json

                parsed = json.loads(raw.decode("utf-8"))
                if isinstance(parsed, dict):
                    body = parsed
            except (UnicodeDecodeError, ValueError):
                # ClickUp sometimes fires with an empty/non-JSON body and the
                # ids in the query string — tolerate that rather than 400.
                body = {}

        try:
            await self._process_webhook(request, body)
        except Exception:
            logger.exception("[clickup] process_webhook failed")
        # Always ACK 200 so ClickUp doesn't enter its 75-min retry storm for
        # a message we've already accepted (dedup protects against genuine
        # retries of un-acked deliveries).
        return web.Response(status=200, text="ok")

    async def _process_webhook(self, request, body: Dict[str, Any]) -> None:
        channel_id, message_id = _webhook_ids(request, body)
        if not channel_id or not message_id:
            logger.debug(
                "[clickup] webhook missing channel_id/message_id (channel=%r message=%r)",
                channel_id,
                message_id,
            )
            return

        # Dedup — ClickUp retries failed deliveries for ~75 min.
        if self._dedup.is_duplicate(message_id):
            logger.debug("[clickup] ignoring duplicate message %s", message_id)
            return

        # allowed_channels whitelist (must pass before any read/dispatch).
        if self._allowed_channels and channel_id not in self._allowed_channels:
            logger.debug("[clickup] ignoring message in non-allowed channel %s", channel_id)
            return

        if self._client is None:
            return

        # The webhook is thin — read the message body back from the API.
        try:
            message = await self._client.get_message(channel_id, message_id)
        except Exception as exc:
            logger.warning("[clickup] failed to read message %s: %s", message_id, exc)
            return
        if not message:
            logger.debug("[clickup] message %s not found in channel %s", message_id, channel_id)
            return

        # Bot-loop guard — skip our own messages.
        author_id = _extract_author_id(message)
        if self._bot_user_id and author_id and author_id == self._bot_user_id:
            logger.debug("[clickup] skipping own message %s", message_id)
            return

        text = _extract_text(message)

        # Mention gating for channels (DMs/self-loop already excluded).
        if self._require_mention and channel_id not in self._free_response_channels:
            has_mention = self._has_bot_mention(text, message)
            if not has_mention:
                logger.debug(
                    "[clickup] skipping message without @mention (channel=%s)", channel_id
                )
                return
            text = self._strip_bot_mention(text)

        source = self.build_source(
            chat_id=channel_id,
            chat_name=channel_id,
            chat_type="channel",
            user_id=author_id or channel_id,
            user_name=_extract_author_name(message) or author_id or channel_id,
            guild_id=self._team_id,
            message_id=message_id,
        )

        channel_prompt = resolve_channel_prompt(self.config.extra or {}, channel_id, None)

        event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=message,
            message_id=message_id,
            channel_prompt=channel_prompt,
        )

        logger.debug("[clickup] message in channel %s: %s", channel_id, text[:80])
        await self.handle_message(event)

    def _has_bot_mention(self, text: str, message: Dict[str, Any]) -> bool:
        """Return True if our bot user is @mentioned in the message."""
        if self._bot_user_id and f"@{self._bot_user_id}" in (text or ""):
            return True
        # ClickUp carries structured mentions; match our user id if present.
        mentions = message.get("mentions") or message.get("mentioned_users") or []
        if isinstance(mentions, list):
            for m in mentions:
                if isinstance(m, dict):
                    mid = m.get("id") or m.get("user_id")
                    if mid is not None and str(mid) == str(self._bot_user_id):
                        return True
                elif str(m) == str(self._bot_user_id):
                    return True
        return False

    def _strip_bot_mention(self, text: str) -> str:
        if self._bot_user_id:
            return (text or "").replace(f"@{self._bot_user_id}", "").strip()
        return (text or "").strip()

    # -- Outbound messaging -------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if self._client is None:
            return SendResult(success=False, error="ClickUp client not initialized")

        formatted = self.format_message(content)
        chunks = self.truncate_message(formatted, max_length=self.MAX_MESSAGE_LENGTH)
        last_message_id: Optional[str] = None

        for index, chunk in enumerate(chunks):
            try:
                # Thread the first chunk under the triggering message when we
                # have one; subsequent chunks go to the channel top-level so a
                # long reply isn't a deep nested chain.
                if reply_to and index == 0:
                    data = await self._client.post_reply(reply_to, chunk)
                else:
                    data = await self._client.post_channel_message(chat_id, chunk)
                last_message_id = str((data or {}).get("id") or "") or last_message_id
            except Exception as exc:
                logger.error("[clickup] send failed: %s", exc)
                return SendResult(success=False, error=str(exc), retryable=True)

        return SendResult(success=True, message_id=last_message_id)

    async def send_typing(self, chat_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        # ClickUp Chat has no typing-indicator API.
        return None

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "channel", "chat_id": chat_id}


# ── Env-driven auto-configuration ─────────────────────────────────────────────

def _env_enablement() -> Optional[Dict[str, Any]]:
    """Seed ``PlatformConfig.extra`` from env vars during gateway config load.

    Called by the platform registry's env-enablement hook BEFORE adapter
    construction, so ``gateway status`` and ``get_connected_platforms()``
    reflect env-only configuration without binding the webhook server.
    Returns ``None`` when ClickUp isn't minimally configured.

    The special ``home_channel`` key in the returned dict becomes a proper
    ``HomeChannel`` dataclass on the ``PlatformConfig`` via the core hook.
    """
    api_key = os.getenv("CLICKUP_API_KEY", "").strip()
    team_id = os.getenv("CLICKUP_TEAM_ID", "").strip()
    if not (api_key and team_id):
        return None

    seed: Dict[str, Any] = {"api_key": api_key, "team_id": team_id}
    port = os.getenv("CLICKUP_PORT", "").strip()
    if port:
        try:
            seed["port"] = int(port)
        except ValueError:
            pass
    host = os.getenv("CLICKUP_HOST", "").strip()
    if host:
        seed["host"] = host
    secret = os.getenv("CLICKUP_WEBHOOK_SECRET", "").strip()
    if secret:
        seed["webhook_secret"] = secret
    secret_header = os.getenv("CLICKUP_WEBHOOK_SECRET_HEADER", "").strip()
    if secret_header:
        seed["webhook_secret_header"] = secret_header
    if os.getenv("CLICKUP_REQUIRE_MENTION") is not None:
        seed["require_mention"] = _truthy(os.getenv("CLICKUP_REQUIRE_MENTION"))
    allowed = _csv_set(os.getenv("CLICKUP_ALLOWED_CHANNELS"))
    if allowed:
        seed["allowed_channels"] = sorted(allowed)
    free = _csv_set(os.getenv("CLICKUP_FREE_RESPONSE_CHANNELS"))
    if free:
        seed["free_response_channels"] = sorted(free)

    home = os.getenv("CLICKUP_HOME_CHANNEL", "").strip()
    if home:
        seed["home_channel"] = {
            "chat_id": home,
            "name": os.getenv("CLICKUP_HOME_CHANNEL_NAME", "Home"),
        }
    return seed


# ── YAML→extra config bridge ───────────────────────────────────────────────────

def _apply_yaml_config(yaml_cfg: dict, clickup_cfg: dict) -> Optional[Dict[str, Any]]:
    """Translate ``config.yaml`` ``clickup:`` keys into ``PlatformConfig.extra``.

    Implements the ``apply_yaml_config_fn`` contract. The core shared-key
    loop already bridges ``require_mention``, ``free_response_channels``, and
    ``channel_prompts`` into ``extra``; this hook additionally surfaces the
    keys that loop does not know about — ``allowed_channels`` (whitelist),
    ``webhook_secret``, ``port``/``host``, and ``api_base`` — so a top-level
    ``clickup:`` block is fully honored without forcing the user to nest them
    under ``extra:``.

    Returned values are merged into ``PlatformConfig.extra``. ``token`` /
    ``api_key`` from env still win at adapter-construct time (the adapter
    reads ``os.getenv`` first), so this never overrides an explicit env var.
    """
    seed: Dict[str, Any] = {}
    if "allowed_channels" in clickup_cfg:
        seed["allowed_channels"] = clickup_cfg["allowed_channels"]
    for key in ("webhook_secret", "webhook_secret_header", "host", "port", "api_base", "api_key", "team_id"):
        if key in clickup_cfg:
            seed[key] = clickup_cfg[key]
    return seed or None


# ── Out-of-process cron delivery ───────────────────────────────────────────────

async def _standalone_send(
    pconfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[List[str]] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    """Out-of-process delivery for cron jobs / send_message_tool fallbacks.

    Used by ``tools/send_message_tool._send_via_adapter`` and the cron
    scheduler when the gateway runner is not in this process (e.g.
    ``hermes cron`` running standalone). Without this hook,
    ``deliver=clickup`` cron jobs fail with ``No live adapter for platform``.

    ``thread_id`` routes the message as an in-thread reply when set; media
    files are accepted for signature parity but not delivered (a text note is
    appended instead — cron-side media needs a publicly reachable upload flow
    the standalone path can't construct).
    """
    if not AIOHTTP_AVAILABLE:
        return {"error": "ClickUp standalone send: aiohttp not installed"}

    extra = getattr(pconfig, "extra", {}) or {}
    api_key = (os.getenv("CLICKUP_API_KEY") or extra.get("api_key", "")).strip()
    team_id = str(os.getenv("CLICKUP_TEAM_ID") or extra.get("team_id", "")).strip()
    api_base = (os.getenv("CLICKUP_API_BASE") or extra.get("api_base", DEFAULT_API_BASE)).rstrip("/")
    if not api_key or not team_id:
        return {"error": "ClickUp standalone send: CLICKUP_API_KEY and CLICKUP_TEAM_ID required"}
    if not chat_id:
        return {"error": "ClickUp standalone send: chat_id (channel id) required"}

    text = message or ""
    if media_files:
        text = f"{text}\n[{len(media_files)} attachment(s) generated; not deliverable from cron]".strip()

    client = _ClickUpClient(api_key, team_id, api_base=api_base)
    try:
        if thread_id:
            data = await client.post_reply(thread_id, text)
        else:
            data = await client.post_channel_message(chat_id, text)
        return {
            "success": True,
            "platform": "clickup",
            "chat_id": chat_id,
            "message_id": str((data or {}).get("id") or "") or None,
        }
    except Exception as exc:
        return {"error": f"ClickUp standalone send failed: {exc}"}


# ── Interactive setup ──────────────────────────────────────────────────────────

def interactive_setup() -> None:
    """Minimal stdin wizard for ``hermes setup clickup``."""
    print()
    print("ClickUp Chat setup")
    print("------------------")
    print("Create a personal API token at https://app.clickup.com/settings/apps")
    print("and copy your Workspace (team) ID from the ClickUp URL.")
    print()

    try:
        from hermes_cli.config import get_env_value, save_env_value
    except ImportError:
        print("hermes_cli.config not available; set CLICKUP_* vars manually in ~/.hermes/.env")
        return

    from hermes_cli.cli_output import (
        print_info,
        print_success,
        print_warning,
        prompt,
        prompt_yes_no,
    )

    existing = get_env_value("CLICKUP_API_KEY")
    if existing and not prompt_yes_no("ClickUp already configured. Reconfigure?", False):
        return

    api_key = prompt("ClickUp API token (pk_...)", default=existing or "", password=True)
    if not api_key:
        print_warning("API token is required — skipping ClickUp setup")
        return
    save_env_value("CLICKUP_API_KEY", api_key.strip())

    team_id = prompt("Workspace (team) ID", default=get_env_value("CLICKUP_TEAM_ID") or "")
    if not team_id:
        print_warning("Workspace ID is required — skipping ClickUp setup")
        return
    save_env_value("CLICKUP_TEAM_ID", team_id.strip())

    secret = prompt(
        "Webhook shared secret (sent in the X-Hermes-Secret header)",
        default=get_env_value("CLICKUP_WEBHOOK_SECRET") or "",
        password=True,
    )
    if not secret:
        print_warning(
            "No webhook secret — the gateway will refuse to start an "
            "unauthenticated ClickUp webhook. Set CLICKUP_WEBHOOK_SECRET."
        )
    else:
        save_env_value("CLICKUP_WEBHOOK_SECRET", secret.strip())

    print()
    print_info("Create a ClickUp Chat webhook Automation per channel pointing at")
    print_info("  <your-public-url>/clickup/webhook")
    print_info(f"with header  {DEFAULT_SECRET_HEADER}: <your-secret>")
    print_success("ClickUp configuration saved to ~/.hermes/.env")


# ── Plugin registration ─────────────────────────────────────────────────────────

def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system at startup."""
    ctx.register_platform(
        name="clickup",
        label="ClickUp Chat",
        adapter_factory=lambda cfg: ClickUpAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["CLICKUP_API_KEY", "CLICKUP_TEAM_ID"],
        install_hint="pip install aiohttp",
        setup_fn=interactive_setup,
        # Env-driven auto-configuration: seeds PlatformConfig.extra so
        # env-only setups show up in `hermes gateway status` without binding
        # the webhook server.
        env_enablement_fn=_env_enablement,
        # YAML→extra config bridge — surfaces clickup: top-level keys the core
        # shared-key loop doesn't know about (allowed_channels, webhook_secret,
        # port/host) into PlatformConfig.extra.
        apply_yaml_config_fn=_apply_yaml_config,
        # Cron home-channel delivery support — `deliver=clickup` cron jobs
        # route to CLICKUP_HOME_CHANNEL when set.
        cron_deliver_env_var="CLICKUP_HOME_CHANNEL",
        # Out-of-process cron delivery via the ClickUp Chat REST API. Without
        # this hook, deliver=clickup cron jobs fail with "No live adapter"
        # when cron runs separately from the gateway.
        standalone_sender_fn=_standalone_send,
        # Auth env vars for _is_user_authorized() integration.
        allowed_users_env="CLICKUP_ALLOWED_USERS",
        allow_all_env="CLICKUP_ALLOW_ALL_USERS",
        max_message_length=MAX_MESSAGE_LENGTH,
        emoji="🗂️",
        allow_update_command=True,
        platform_hint=(
            "You are chatting via ClickUp Chat. ClickUp renders Markdown — "
            "bold (**text**), italic (*text*), inline code (`code`), and "
            "fenced code blocks all work. Keep responses clear and "
            "professional; replies are posted in-thread under the triggering "
            "message."
        ),
    )
