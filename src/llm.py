"""OpenAI-compatible chat completions with provider fallback (stdlib urllib)."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from .providers import ProviderConfig, provider_chain

_RETRYABLE_SIGNALS = (
    "429", "402", "rate_limit", "rate limit",
    "500", "502", "503", "504", "overloaded",
    "timed out", "timeout", "time_out", "read timeout",
    "404", "not found", "model not found",
    "does not exist", "does not support response format",
    "json_schema", "decommissioned", "deprecated", "no longer supported",
)


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    # Try several locations
    for candidate in [
        Path(__file__).resolve().parent.parent / ".env",
        Path.cwd() / ".env",
    ]:
        if candidate.is_file():
            load_dotenv(candidate)


_load_dotenv()


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(sig in msg for sig in _RETRYABLE_SIGNALS)


def _call_openai_compatible(
    conf: ProviderConfig,
    *,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    timeout_sec: float = 120.0,
) -> str:
    api_key = os.getenv(conf.api_key_env, "")
    if not api_key:
        raise ValueError(f"{conf.name}: {conf.api_key_env} not set")
    url = f"{conf.base_url.rstrip('/')}/chat/completions"
    payload: dict = {
        "model": conf.default_model,
        "temperature": temperature,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM HTTP {e.code}: {body}") from e

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected LLM response: {data!r}") from e


def chat_completion(
    system: str,
    user: str,
    *,
    llm_task: str = "formal",
    temperature: float = 0.2,
    max_tokens: int = 4096,
    timeout_sec: float = 120.0,
) -> str:
    """Try providers in chain order; each uses its default_model."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    ordered: list[ProviderConfig] = []
    dedup: set[tuple[str, str]] = set()
    for conf in provider_chain(llm_task):
        if not conf.enabled or not os.getenv(conf.api_key_env, ""):
            continue
        tup = (conf.api_key_env, conf.base_url)
        if tup in dedup:
            continue
        dedup.add(tup)
        ordered.append(conf)

    if not ordered:
        hint = ", ".join(p.api_key_env for p in provider_chain(llm_task) if p.enabled) or "(none)"
        raise RuntimeError(
            "No LLM API keys available. Set keys in .env file. "
            f"Expected env vars: {hint}"
        )

    last_error: Exception | None = None
    for conf in ordered:
        try:
            return _call_openai_compatible(
                conf,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_sec=timeout_sec,
            )
        except Exception as exc:
            last_error = exc
            if _is_retryable(exc):
                print(
                    f"  ⚠ {conf.name} unavailable ({str(exc)[:80]}), trying next...",
                    file=sys.stderr,
                )
                continue
            raise

    raise RuntimeError(
        f"All LLM providers exhausted. Last error: {last_error}\n"
        f"Tried: {[p.name for p in ordered]}"
    )
