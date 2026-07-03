"""Provider-agnostic LLM + embedding client for the bytical-talk brain.

Defaults to Azure OpenAI (chat = gpt-4o-mini, embeddings = text-embedding-3-small)
but works with any OpenAI-compatible endpoint (OpenAI, Azure AI Foundry v1, local
vLLM, etc.) by setting BYTICAL_LLM_PROVIDER=openai. All credentials are read from
the environment / .env — nothing is hard-coded, nothing is committed.

Public surface:
    client = LLMClient()          # reads env
    client.chat(messages) -> str
    client.chat_json(messages) -> dict     # strict JSON out (structured brain output)
    client.embed(list_of_text) -> list[list[float]]
    client.available -> bool               # False if no creds (brain degrades gracefully)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

try:  # optional: load .env if python-dotenv is present
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


@dataclass
class LLMConfig:
    provider: str = field(default_factory=lambda: os.getenv("BYTICAL_LLM_PROVIDER", "azure").lower())
    # azure
    azure_endpoint: str | None = field(default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT"))
    azure_key: str | None = field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_KEY"))
    azure_api_version: str = field(default_factory=lambda: os.getenv("AZURE_API_VERSION", "2025-01-01-preview"))
    azure_chat_deployment: str = field(default_factory=lambda: os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o-mini"))
    azure_embed_deployment: str = field(default_factory=lambda: os.getenv("AZURE_EMBEDDING_MODEL_NAME", "text-embedding-3-small"))
    # generic openai-compatible
    openai_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_base_url: str | None = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL"))
    openai_chat_model: str = field(default_factory=lambda: os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    openai_embed_model: str = field(default_factory=lambda: os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"))


class LLMClient:
    """Thin wrapper over the OpenAI SDK that hides the Azure-vs-OpenAI difference."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        self._client = None
        self._chat_model = None
        self._embed_model = None
        self._init_client()

    # -- setup ------------------------------------------------------------
    def _init_client(self) -> None:
        cfg = self.config
        try:
            if cfg.provider == "azure":
                if not (cfg.azure_endpoint and cfg.azure_key):
                    return
                from openai import AzureOpenAI

                self._client = AzureOpenAI(
                    azure_endpoint=cfg.azure_endpoint,
                    api_key=cfg.azure_key,
                    api_version=cfg.azure_api_version,
                )
                self._chat_model = cfg.azure_chat_deployment
                self._embed_model = cfg.azure_embed_deployment
            else:
                if not cfg.openai_key:
                    return
                from openai import OpenAI

                self._client = OpenAI(api_key=cfg.openai_key, base_url=cfg.openai_base_url)
                self._chat_model = cfg.openai_chat_model
                self._embed_model = cfg.openai_embed_model
        except Exception:
            self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def _require(self) -> None:
        if not self.available:
            raise RuntimeError(
                "LLMClient has no credentials. Set AZURE_OPENAI_* (provider=azure) or "
                "OPENAI_API_KEY (provider=openai) in your environment / .env."
            )

    # -- chat -------------------------------------------------------------
    def chat(self, messages: list[dict[str, str]], temperature: float = 0.3,
             max_tokens: int = 1024) -> str:
        self._require()
        resp = self._client.chat.completions.create(
            model=self._chat_model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    def chat_json(self, messages: list[dict[str, str]], temperature: float = 0.2,
                  max_tokens: int = 2048) -> dict[str, Any]:
        """Chat constrained to a JSON object. Falls back to best-effort extraction
        if the model/endpoint does not support response_format."""
        self._require()
        try:
            resp = self._client.chat.completions.create(
                model=self._chat_model, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or "{}"
        except Exception:
            content = self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        return _loads_lenient(content)

    # -- embeddings -------------------------------------------------------
    def embed(self, texts: list[str]) -> list[list[float]]:
        self._require()
        resp = self._client.embeddings.create(model=self._embed_model, input=texts)
        return [d.embedding for d in resp.data]


def _loads_lenient(s: str) -> dict[str, Any]:
    """Parse JSON, tolerating markdown fences or surrounding prose."""
    s = s.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1] if "```" in s[3:] else s[3:]
        if s.lstrip().startswith("json"):
            s = s.lstrip()[4:]
    try:
        return json.loads(s)
    except Exception:
        start, end = s.find("{"), s.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(s[start:end + 1])
            except Exception:
                pass
    return {}
