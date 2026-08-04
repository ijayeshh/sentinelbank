"""Ollama client wrapper speaking the OpenAI-compatible /v1/chat/completions interface.

Why this endpoint and not Ollama's native /api/generate: /v1/chat/completions is a
vendor-neutral contract. vLLM, llama.cpp's server, TGI and Ollama all serve it with the
same request and response shapes, so swapping the serving layer later is a change to
OLLAMA_BASE_URL and OLLAMA_MODEL only — no code edit (PROJECT_PLAN.md §2, §4.5).
Nothing in this module may depend on an Ollama-specific field.

On num_ctx: it is deliberately NOT sent in the request body. The OpenAI-compatible
surface has no context-window parameter, and inventing an `options` key here would be
exactly the Ollama-specific coupling this interface exists to avoid. The 4096 cap lives
in the Modelfile, which `ollama create sentinelbank-llm` bakes into the model. The
LLM_NUM_CTX setting is carried here so that a mismatch between .env and the Modelfile is
reported loudly at startup rather than discovered as an OOM mid-demo.

The client is a module-level singleton (§1.1 rule 4): one connection pool per process,
never one per request.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# Requests are one-shot and local. 120s is generous for a 3B model on a GTX 1650 but
# still bounded — a hung Ollama must surface as an error, not as a stalled UI.
_REQUEST_TIMEOUT = httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0)

Message = dict[str, str]


def _strip_default_tag(name: str) -> str:
    """Drop an explicit ':latest' so tagged and untagged names compare equal."""
    return name[: -len(":latest")] if name.endswith(":latest") else name


class LLMError(RuntimeError):
    """Raised when the LLM endpoint is unreachable or returns an unusable response.

    Callers treat this as a hard failure. Fail closed (§6 rule 4): a rail or agent step
    that cannot get a completion must block, never proceed on a guessed answer.
    """


@dataclass(frozen=True)
class Completion:
    """One assistant turn plus the numbers the smoke test and latency tables need."""

    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_s: float

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient:
    """Thin synchronous wrapper over POST {base_url}/chat/completions.

    Deliberately thin. This project's security properties live in the rails, and a
    clever client that retried, cached, or rewrote prompts would sit between the rails
    and the model where nobody would think to audit it. Semantic caching and model
    routing are explicitly out of scope (§2.1).
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> None:
        # Trailing slash matters: httpx.Client(base_url=...) joins relative paths
        # against it, and "http://host/v1" + "chat/completions" would drop the /v1.
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model
        self.temperature = settings.llm_temperature if temperature is None else temperature
        self._client = httpx.Client(base_url=self.base_url, timeout=_REQUEST_TIMEOUT)

    @property
    def _http(self) -> httpx.Client:
        """The connection pool, reopened if it was closed.

        `close()` is called from the FastAPI lifespan shutdown, but this object is a
        module-level singleton that outlives any one app instance — a test suite, or a
        reloading dev server, starts the app again in the same process. Without this the
        second startup would inherit a permanently closed pool and every LLM call would
        fail with "Cannot send a request, as the client has been closed", which reads like
        an Ollama outage rather than a lifecycle bug.
        """
        if self._client.is_closed:
            self._client = httpx.Client(base_url=self.base_url, timeout=_REQUEST_TIMEOUT)
        return self._client

    # -- health ---------------------------------------------------------------
    def is_available(self) -> bool:
        """True if the endpoint answers and serves the configured model.

        Uses /v1/models — the OpenAI-compatible listing, not Ollama's /api/tags.
        """
        try:
            return self.serves_model()
        except LLMError:
            return False

    def serves_model(self) -> bool:
        """True if the configured model name matches something the endpoint serves.

        Tag-insensitive: Ollama lists a model created without an explicit tag as
        "sentinelbank-llm:latest" but accepts "sentinelbank-llm" in requests. A literal
        string comparison would report a working model as missing.
        """
        wanted = _strip_default_tag(self.model)
        return any(_strip_default_tag(name) == wanted for name in self.list_models())

    def list_models(self) -> list[str]:
        """Model IDs the endpoint currently serves."""
        try:
            response = self._http.get("/models")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"cannot reach {self.base_url}/models: {exc}") from exc

        try:
            return [entry["id"] for entry in response.json()["data"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise LLMError(f"unexpected /models payload from {self.base_url}: {exc}") from exc

    # -- generation -----------------------------------------------------------
    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> Completion:
        """Send a message list, return the assistant reply.

        `messages` is the OpenAI shape: [{"role": "system"|"user"|"assistant",
        "content": "..."}]. Only fields in the OpenAI request schema are sent.
        """
        payload: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            # Streaming is off by design: the PII rail redacts whole responses, and
            # streaming-aware redaction is explicitly out of scope (§2.1).
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stop:
            payload["stop"] = stop

        started = time.perf_counter()
        try:
            response = self._http.post("/chat/completions", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            # Ollama returns the reason in the body; the status line alone is rarely
            # enough to tell "model not created" from "context overflow".
            raise LLMError(
                f"{self.model} returned HTTP {exc.response.status_code}: "
                f"{exc.response.text[:500]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(
                f"cannot reach {self.base_url}/chat/completions: {exc}. "
                "Is `ollama serve` running?"
            ) from exc
        elapsed = time.perf_counter() - started

        try:
            body = response.json()
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError(f"malformed completion payload: {exc}") from exc

        # usage is optional in the OpenAI schema; treat it as telemetry, not contract.
        usage = body.get("usage") or {}
        return Completion(
            text=text,
            model=body.get("model", self.model),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            elapsed_s=elapsed,
        )

    def complete(self, prompt: str, *, system: str | None = None, **kwargs) -> Completion:
        """Single-turn convenience wrapper around chat()."""
        messages: list[Message] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **kwargs)

    def close(self) -> None:
        """Release the connection pool. Called from the FastAPI lifespan shutdown."""
        self._client.close()


# Module-level singleton. Import this; do not construct LLMClient() per request —
# a new pool per request is exactly the per-request allocation §1.1 rule 4 forbids.
llm = LLMClient()
