"""LLM clients with one tiny common interface:

    client.status()                       -> dict (reachable, model, ...)
    client.chat_json(system, user, schema) -> (parsed_json, meta)

Providers:
  * ollama       - local, uses Ollama structured outputs (JSON schema)
  * groq / gemini / openrouter / openai - any OpenAI-compatible chat API with JSON mode
  * anthropic    - Messages API
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

import requests

from .config import Settings


class LLMError(Exception):
    pass


class LLMUnavailable(LLMError):
    """The provider cannot be reached / is not configured -> rule-based fallback."""


def parse_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    raise LLMError("Model returned text that is not valid JSON: " + text[:200])


def _schema_hint(schema: dict[str, Any]) -> str:
    """Compact description of the expected JSON for providers without schema enforcement."""
    props = schema.get("properties", {})
    if "actions" in props:
        enum = props["actions"]["items"]["properties"]["action"].get("enum", [])
        return ("Return ONLY a JSON object of the form "
                '{"thinking": "<your reasoning>", "actions": [{"action": "<one of: ' + ", ".join(enum) + '>", '
                '"column": "<column name>", "params": {<parameters for that action>}, "reason": "<short reason>", '
                '"confidence": <0-1>}]}. "actions" may be an empty list. No markdown, no text outside the JSON.')
    if "summary" in props:
        return 'Return ONLY a JSON object: {"summary": "<text>"}. No markdown.'
    return "Return ONLY a JSON object matching the requested keys."


# --------------------------------------------------------------------------------------
# Ollama (local)
# --------------------------------------------------------------------------------------

class OllamaClient:
    def __init__(self, settings: Settings):
        self.s = settings
        self._supports_schema = True

    def base(self) -> str:
        return self.s.resolved_base_url()

    def status(self) -> dict[str, Any]:
        model = self.s.resolved_model()
        try:
            r = requests.get(f"{self.base()}/api/tags", timeout=4)
            r.raise_for_status()
            models = [m.get("name") for m in r.json().get("models", [])]
        except requests.RequestException as exc:
            return {"provider": "ollama", "reachable": False, "error": str(exc), "models": [], "model": model, "model_available": False}
        version = None
        try:
            version = requests.get(f"{self.base()}/api/version", timeout=3).json().get("version")
        except Exception:  # noqa: BLE001
            pass
        available = model in models or f"{model}:latest" in models or any(m.split(":")[0] == model for m in models)
        return {"provider": "ollama", "reachable": True, "version": version, "models": models, "model": model, "model_available": available}

    def chat_json(self, system: str, user: str, schema: dict[str, Any], num_predict: int = 1800):
        payload = {
            "model": self.s.resolved_model(),
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": float(self.s.temperature), "num_ctx": int(self.s.num_ctx), "num_predict": num_predict},
            "keep_alive": "15m",
            "format": schema if self._supports_schema else "json",
        }
        t0 = time.time()
        try:
            r = requests.post(f"{self.base()}/api/chat", json=payload, timeout=self.s.request_timeout)
        except requests.ConnectionError as exc:
            raise LLMUnavailable(f"Cannot reach Ollama at {self.base()} - is `ollama serve` running?") from exc
        except requests.Timeout as exc:
            raise LLMError(f"Ollama timed out after {self.s.request_timeout}s") from exc
        if r.status_code == 404:
            raise LLMError(f"Model '{self.s.resolved_model()}' not found. Run:  ollama pull {self.s.resolved_model()}")
        if r.status_code == 400 and self._supports_schema and "format" in r.text.lower():
            self._supports_schema = False
            return self.chat_json(system, user, schema, num_predict)
        if r.status_code != 200:
            raise LLMError(f"Ollama error {r.status_code}: {r.text[:300]}")
        data = r.json()
        content = (data.get("message") or {}).get("content", "")
        meta = {"seconds": round(time.time() - t0, 1), "raw": content, "eval_count": data.get("eval_count"),
                "prompt_eval_count": data.get("prompt_eval_count")}
        return parse_json(content), meta


# --------------------------------------------------------------------------------------
# OpenAI-compatible (Groq, Gemini, OpenRouter, OpenAI, Together, ...)
# --------------------------------------------------------------------------------------

class OpenAICompatClient:
    def __init__(self, settings: Settings):
        self.s = settings

    def _headers(self) -> dict[str, str]:
        h = {"Authorization": f"Bearer {self.s.api_key}", "Content-Type": "application/json"}
        if self.s.provider == "openrouter":
            h["HTTP-Referer"] = "https://github.com/ai-dataset-cleaner"
            h["X-Title"] = "AI Dataset Cleaner"
        return h

    def status(self) -> dict[str, Any]:
        model = self.s.resolved_model()
        base = {"provider": self.s.provider, "model": model, "models": [], "base_url": self.s.resolved_base_url()}
        if not self.s.api_key:
            return base | {"reachable": False, "model_available": False, "error": "LLM_API_KEY is not set"}
        try:
            r = requests.get(f"{self.s.resolved_base_url()}/models", headers=self._headers(), timeout=8)
            if r.status_code == 401:
                return base | {"reachable": True, "model_available": False, "error": "API key rejected (401)"}
            r.raise_for_status()
            data = r.json()
            items = data.get("data", data.get("models", []))
            models = []
            for m in items:
                mid = m.get("id") or m.get("name", "")
                models.append(mid.replace("models/", ""))
            available = (not models) or model in models or any(m.endswith(model) for m in models)
            return base | {"reachable": True, "models": sorted(models)[:200], "model_available": available}
        except requests.RequestException as exc:
            return base | {"reachable": False, "model_available": False, "error": str(exc)[:200]}

    def chat_json(self, system: str, user: str, schema: dict[str, Any], num_predict: int = 1800):
        if not self.s.api_key:
            raise LLMUnavailable(f"LLM_API_KEY is not set for provider '{self.s.provider}'")
        system_full = system + "\n\n" + _schema_hint(schema)
        payload = {
            "model": self.s.resolved_model(),
            "messages": [{"role": "system", "content": system_full}, {"role": "user", "content": user}],
            "temperature": float(self.s.temperature),
            "max_tokens": num_predict,
            "response_format": {"type": "json_object"},
        }
        t0 = time.time()
        last_err = None
        for attempt in range(3):
            try:
                r = requests.post(f"{self.s.resolved_base_url()}/chat/completions", headers=self._headers(),
                                  json=payload, timeout=self.s.request_timeout)
            except requests.ConnectionError as exc:
                raise LLMUnavailable(f"Cannot reach {self.s.provider} API: {exc}") from exc
            except requests.Timeout as exc:
                raise LLMError(f"{self.s.provider} timed out after {self.s.request_timeout}s") from exc
            if r.status_code == 429:
                wait = min(20, 3 * (attempt + 1))
                try:
                    wait = min(30, float(r.headers.get("retry-after", wait)))
                except ValueError:
                    pass
                last_err = f"rate limited (429): {r.text[:160]}"
                time.sleep(wait)
                continue
            if r.status_code == 400 and "response_format" in r.text and "response_format" in payload:
                payload.pop("response_format")          # provider/model without JSON mode
                continue
            if r.status_code in (401, 403):
                raise LLMUnavailable(f"{self.s.provider} rejected the API key ({r.status_code})")
            if r.status_code == 404:
                raise LLMError(f"Model '{self.s.resolved_model()}' not found at {self.s.provider} ({r.text[:160]})")
            if r.status_code >= 500:
                last_err = f"{self.s.provider} server error {r.status_code}"
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code != 200:
                raise LLMError(f"{self.s.provider} error {r.status_code}: {r.text[:300]}")
            data = r.json()
            content = data["choices"][0]["message"].get("content", "")
            usage = data.get("usage", {})
            meta = {"seconds": round(time.time() - t0, 1), "raw": content,
                    "eval_count": usage.get("completion_tokens"), "prompt_eval_count": usage.get("prompt_tokens")}
            return parse_json(content), meta
        raise LLMError(last_err or "request failed")


# --------------------------------------------------------------------------------------
# Anthropic Messages API
# --------------------------------------------------------------------------------------

class AnthropicClient:
    def __init__(self, settings: Settings):
        self.s = settings

    def _headers(self):
        return {"x-api-key": self.s.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}

    def status(self):
        model = self.s.resolved_model()
        base = {"provider": "anthropic", "model": model, "models": []}
        if not self.s.api_key:
            return base | {"reachable": False, "model_available": False, "error": "LLM_API_KEY is not set"}
        try:
            r = requests.get(f"{self.s.resolved_base_url()}/models", headers=self._headers(), timeout=8)
            if r.status_code == 401:
                return base | {"reachable": True, "model_available": False, "error": "API key rejected (401)"}
            r.raise_for_status()
            models = [m["id"] for m in r.json().get("data", [])]
            return base | {"reachable": True, "models": models, "model_available": (model in models) or not models}
        except requests.RequestException as exc:
            return base | {"reachable": False, "model_available": False, "error": str(exc)[:200]}

    def chat_json(self, system: str, user: str, schema: dict[str, Any], num_predict: int = 1800):
        if not self.s.api_key:
            raise LLMUnavailable("LLM_API_KEY is not set for provider 'anthropic'")
        payload = {"model": self.s.resolved_model(), "max_tokens": num_predict, "temperature": float(self.s.temperature),
                   "system": system + "\n\n" + _schema_hint(schema),
                   "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": "{"}]}
        t0 = time.time()
        try:
            r = requests.post(f"{self.s.resolved_base_url()}/messages", headers=self._headers(), json=payload,
                              timeout=self.s.request_timeout)
        except requests.ConnectionError as exc:
            raise LLMUnavailable(f"Cannot reach Anthropic API: {exc}") from exc
        if r.status_code in (401, 403):
            raise LLMUnavailable(f"Anthropic rejected the API key ({r.status_code})")
        if r.status_code != 200:
            raise LLMError(f"Anthropic error {r.status_code}: {r.text[:300]}")
        data = r.json()
        content = "{" + "".join(b.get("text", "") for b in data.get("content", []))
        usage = data.get("usage", {})
        meta = {"seconds": round(time.time() - t0, 1), "raw": content,
                "eval_count": usage.get("output_tokens"), "prompt_eval_count": usage.get("input_tokens")}
        return parse_json(content), meta


def make_client(settings: Settings):
    p = settings.provider
    if p == "ollama":
        return OllamaClient(settings)
    if p == "anthropic":
        return AnthropicClient(settings)
    return OpenAICompatClient(settings)
