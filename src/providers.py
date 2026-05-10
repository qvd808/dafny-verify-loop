"""Provider registry — OpenAI-compatible API endpoints with fallback chain.

The list ORDER is fallback priority: first usable key wins; subsequent providers
are tried when the previous returns a retryable error (429, 5xx, unknown model, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    api_key_env: str
    base_url: str
    default_model: str
    enabled: bool = True
    preferred_for_formal: bool = False


PROVIDERS: list[ProviderConfig] = [
    ProviderConfig(
        name="groq",
        api_key_env="GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1",
        default_model="llama-3.3-70b-versatile",
    ),
    ProviderConfig(
        name="nvidia",
        api_key_env="NVIDIA_API_KEY",
        base_url="https://integrate.api.nvidia.com/v1",
        default_model="meta/llama-3.3-70b-instruct",
        preferred_for_formal=True,
    ),
    ProviderConfig(
        name="sambanova",
        api_key_env="SAMBANOVA_API_KEY",
        base_url="https://api.sambanova.ai/v1",
        default_model="Meta-Llama-3.3-70B-Instruct",
        preferred_for_formal=True,
    ),
    ProviderConfig(
        name="mistral",
        api_key_env="MISTRAL_API_KEY",
        base_url="https://api.mistral.ai/v1",
        default_model="mistral-small-latest",
        preferred_for_formal=True,
    ),
    ProviderConfig(
        name="openrouter",
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        default_model="meta-llama/llama-3.3-70b-instruct:free",
    ),
    ProviderConfig(
        name="cerebras",
        api_key_env="CEREBRAS_API_KEY",
        base_url="https://api.cerebras.ai/v1",
        default_model="llama3.1-8b",
    ),
]


def enabled_in_order() -> list[ProviderConfig]:
    return [p for p in PROVIDERS if p.enabled]


def formal_ordered() -> list[ProviderConfig]:
    enabled = enabled_in_order()
    formal = [p for p in enabled if p.preferred_for_formal]
    others = [p for p in enabled if not p.preferred_for_formal]
    return formal + others


def provider_chain(task: str = "formal") -> list[ProviderConfig]:
    """Hoist preferred_for_formal providers first when task=='formal'."""
    if task == "formal":
        return formal_ordered()
    return enabled_in_order()
