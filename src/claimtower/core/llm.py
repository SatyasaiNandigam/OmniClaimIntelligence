"""The ONLY module allowed to talk to an LLM provider SDK (T0.4).

    llm = LLMClient.from_settings(settings)
    text = llm.chat("Summarise ...")
    obj  = llm.structured(MyModel, prompt)            # validated Pydantic object
    obj  = llm.vision(prompt, [png_bytes], model_cls=ImageFinding)

Providers (selected by `llm.provider` in configs/app.yaml): `openai` (also any OpenAI-compatible
endpoint via LLM_BASE_URL) and `azure_openai`. Both use the `openai` SDK, imported lazily.

Cassettes make tests deterministic and offline:
    mode "replay" -> answer only from the cassette (flags.llm_replay_mode: true); a miss raises CassetteMiss
    mode "record" -> call the provider and save every response (env LLM_RECORD=1)
    mode "auto"   -> use the recorded answer if there is one (no API call, no cost), otherwise call the provider and record it
    mode "off"    -> call the provider (default)
Every call is logged in `client.calls` with provider, model, prompt name/version and token usage
so the audit trail can store model_version / prompt_version.

`settlement/` must never import this module (import-linter rule in pyproject.toml).
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Callable, Literal, Protocol, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from claimtower.core.config import Pack, Settings
from claimtower.core.errors import ConfigError
from claimtower.core.registry import Registry

log = logging.getLogger("claimtower.llm")
M = TypeVar("M", bound=BaseModel)
REQUEST_TIMEOUT_S = 120.0
EMBED_BATCH = 64
EMBED_DECIMALS = 5                  # vectors are rounded so a live call and its replay give identical numbers
Mode = Literal["off", "record", "replay", "auto"]


# ------------------------------------------------------------------ errors
class LLMError(Exception): ...
class TransientLLMError(LLMError):
    """Rate limit / timeout / 5xx - retried with backoff."""
class CassetteMiss(LLMError):
    """Replay mode and no recorded response for this request."""
class StructuredOutputError(LLMError):
    """Model output could not be parsed/validated into the requested model."""


# ------------------------------------------------------------------ data types
@dataclass(frozen=True)
class Image:
    data: bytes
    mime: str = "image/png"


@dataclass
class LLMRequest:
    model: str
    messages: list[dict[str, str]]            # [{"role": "user"|"assistant", "content": str}]
    system: str | None = None
    images: list[Image] = field(default_factory=list)   # attached to the last user message
    temperature: float | None = None          # None = not sent (reasoning models such as gpt-5-mini only accept their default)
    max_tokens: int | None = None
    reasoning_effort: str | None = None       # minimal | low | medium | high (reasoning models); None = not sent

    def cache_key(self) -> str:
        payload = {"model": self.model, "system": self.system, "messages": self.messages,
                   "images": [hashlib.sha256(i.data).hexdigest() + i.mime for i in self.images],
                   "temperature": self.temperature, "max_tokens": self.max_tokens, "reasoning_effort": self.reasoning_effort}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class CallRecord:
    kind: str                       # chat | structured | vision | embed
    provider: str
    model: str
    prompt_name: str | None
    prompt_version: str | None
    input_tokens: int
    output_tokens: int
    latency_s: float
    replayed: bool


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    text: str


class Provider(Protocol):
    name: str
    def complete(self, req: LLMRequest) -> LLMResponse: ...


PROVIDERS: Registry[Callable[..., Provider]] = Registry("llm provider")


# ------------------------------------------------------------------ prompts
_FRONT = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)
_VAR = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def load_prompt(name: str, pack: Pack, **variables: Any) -> Prompt:
    """Read packs/<pack>/prompts/<name>.md. Needs `version:` front-matter; `{{var}}` placeholders are filled."""
    path = pack.prompts_dir / f"{name}.md"
    if not path.is_file():
        raise ConfigError(f"prompt '{name}' not found at {path}")
    m = _FRONT.match(path.read_text(encoding="utf-8"))
    meta = yaml.safe_load(m.group(1)) if m else None
    if not isinstance(meta, dict) or "version" not in meta:
        raise ConfigError(f"{path.name}: prompts need '---\\nversion: <x>\\n---' front-matter (audit records the version)")
    body = m.group(2)
    missing = sorted({v for v in _VAR.findall(body)} - set(variables))
    if missing:
        raise ConfigError(f"{path.name}: missing prompt variables {missing}")
    return Prompt(name=name, version=str(meta["version"]), text=_VAR.sub(lambda g: str(variables[g.group(1)]), body))


# ------------------------------------------------------------------ cassette
class Cassette:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._data: dict[str, dict[str, Any]] = {}
        if self.path.is_file():
            self._data = json.loads(self.path.read_text(encoding="utf-8"))

    def get(self, key: str) -> LLMResponse | None:
        d = self._data.get(key)
        return LLMResponse(**d) if d else None

    def put(self, key: str, resp: LLMResponse, *, save: bool = True) -> None:
        self._data[key] = {"text": resp.text, "model": resp.model,
                           "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens}
        if save:
            self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ providers
_TRANSIENT = {"RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError"}


class OpenAIProvider:
    """OpenAI, any OpenAI-compatible endpoint (LLM_BASE_URL), or Azure OpenAI (azure=True)."""

    def __init__(self, name: str = "openai", client: Any = None, azure: bool = False) -> None:
        self.name = name
        self._client = client
        self._azure = azure

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import openai  # the only SDK import in the codebase
            except ImportError as e:
                raise LLMError("openai SDK not installed: pip install -e '.[llm]'") from e
            key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise LLMError("set LLM_API_KEY in .env (or use replay mode)")
            if self._azure:
                self._client = openai.AzureOpenAI(
                    api_key=key, azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"), timeout=REQUEST_TIMEOUT_S, max_retries=0)
            else:
                # our own bounded retry (LLMClient) governs retries, so the SDK's hidden ones are off
                self._client = openai.OpenAI(api_key=key, base_url=os.environ.get("LLM_BASE_URL") or None, timeout=REQUEST_TIMEOUT_S,
                                             max_retries=0)
        return self._client

    @staticmethod
    def _to_messages(req: LLMRequest) -> list[dict[str, Any]]:
        msgs: list[dict[str, Any]] = [{"role": "system", "content": req.system}] if req.system else []
        msgs += [dict(m) for m in req.messages]
        if req.images:
            parts: list[dict[str, Any]] = [{"type": "text", "text": msgs[-1]["content"]}]
            for img in req.images:
                b64 = base64.b64encode(img.data).decode()
                parts.append({"type": "image_url", "image_url": {"url": f"data:{img.mime};base64,{b64}"}})
            msgs[-1] = {"role": msgs[-1]["role"], "content": parts}
        return msgs

    def complete(self, req: LLMRequest) -> LLMResponse:
        kwargs: dict[str, Any] = {"model": req.model, "messages": self._to_messages(req)}
        if req.temperature is not None:                # omitted when unset: reasoning models reject anything but their default
            kwargs["temperature"] = req.temperature
        if req.reasoning_effort:
            kwargs["reasoning_effort"] = req.reasoning_effort
        if req.max_tokens:                             # includes hidden reasoning tokens on reasoning models
            kwargs["max_completion_tokens"] = req.max_tokens
        try:
            r = self._get_client().chat.completions.create(**kwargs)
        except LLMError:
            raise
        except Exception as e:  # map SDK errors without importing SDK types
            if type(e).__name__ in _TRANSIENT:
                raise TransientLLMError(str(e)) from e
            raise LLMError(f"{type(e).__name__}: {e}") from e
        usage = getattr(r, "usage", None)
        return LLMResponse(text=r.choices[0].message.content or "", model=getattr(r, "model", req.model),
                           input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                           output_tokens=getattr(usage, "completion_tokens", 0) or 0)


    def embed(self, model: str, texts: Sequence[str], dimensions: int | None = None) -> tuple[list[list[float]], int]:
        kwargs: dict[str, Any] = {"model": model, "input": list(texts)}
        if dimensions:
            kwargs["dimensions"] = dimensions
        try:
            r = self._get_client().embeddings.create(**kwargs)
        except LLMError:
            raise
        except Exception as e:
            if type(e).__name__ in _TRANSIENT:
                raise TransientLLMError(str(e)) from e
            raise LLMError(f"{type(e).__name__}: {e}") from e
        usage = getattr(r, "usage", None)
        return [list(d.embedding) for d in sorted(r.data, key=lambda d: d.index)], getattr(usage, "prompt_tokens", 0) or 0


PROVIDERS.register("openai", lambda: OpenAIProvider("openai"))
PROVIDERS.register("azure_openai", lambda: OpenAIProvider("azure_openai", azure=True))


# ------------------------------------------------------------------ client
class LLMClient:
    def __init__(self, provider: Provider, *, text_model: str, vision_model: str, temperature: float | None = None,
                 reasoning_effort: str | None = None, embedding_model: str | None = None, embedding_dimensions: int | None = None,
                 mode: Mode = "off", cassette: Cassette | None = None,
                 max_retries: int = 3, backoff_s: float = 1.0, sleep: Callable[[float], None] = time.sleep) -> None:
        if mode in ("record", "replay", "auto") and cassette is None:
            raise ValueError(f"mode '{mode}' needs a cassette")
        self.provider, self.text_model, self.vision_model = provider, text_model, vision_model
        self.temperature, self.reasoning_effort, self.mode, self.cassette = temperature, reasoning_effort, mode, cassette
        self.embedding_model, self.embedding_dimensions = embedding_model, embedding_dimensions
        self.max_retries, self.backoff_s, self._sleep = max_retries, backoff_s, sleep
        self.calls: list[CallRecord] = []

    @classmethod
    def from_settings(cls, settings: Settings, cassette_path: str | Path | None = None, **kw: Any) -> "LLMClient":
        cfg = settings.app.llm
        mode: Mode = "replay" if settings.app.flags.llm_replay_mode else ("record" if os.environ.get("LLM_RECORD") else "off")
        path = cassette_path or os.environ.get("LLM_CASSETTE") or settings.root / "data" / "cassettes" / "default.json"
        return cls(PROVIDERS.get(cfg.provider)(), text_model=cfg.text_model, vision_model=cfg.vision_model,
                   temperature=cfg.temperature, reasoning_effort=cfg.reasoning_effort, embedding_model=cfg.embedding_model,
                   embedding_dimensions=cfg.embedding_dimensions, mode=mode,
                   cassette=Cassette(Path(path)) if mode != "off" else None, **kw)

    def derive(self, **overrides: Any) -> "LLMClient":
        """A copy with some settings changed (e.g. a higher reasoning_effort for one hard task). It shares the cassette and the call log."""
        allowed = {"temperature", "reasoning_effort", "text_model", "vision_model", "embedding_model", "embedding_dimensions",
                   "max_retries"}
        bad = set(overrides) - allowed
        if bad:
            raise ValueError(f"cannot override {sorted(bad)}")
        clone = copy.copy(self)
        for k, v in overrides.items():
            setattr(clone, k, v)
        return clone

    # ---- public API
    def chat(self, prompt: str | Prompt, *, system: str | None = None, history: list[dict[str, str]] | None = None,
             max_tokens: int | None = None) -> str:
        return self._call("chat", self.text_model, prompt, system, history, [], max_tokens).text

    def structured(self, model_cls: type[M], prompt: str | Prompt, *, system: str | None = None,
                   images: list[Image] | None = None, max_repairs: int = 1, max_tokens: int | None = None) -> M:
        """Ask for JSON matching `model_cls`, validate it; on failure re-ask with the error (max_repairs times)."""
        schema = json.dumps(model_cls.model_json_schema())
        sys_msg = (system + "\n\n" if system else "") + (
            "Reply with ONE JSON object only (no prose, no code fences) that validates against this JSON Schema:\n" + schema)
        model = self.vision_model if images else self.text_model
        history: list[dict[str, str]] = []
        for attempt in range(max_repairs + 1):
            resp = self._call("vision" if images else "structured", model, prompt, sys_msg, history,
                              images or [], max_tokens)
            try:
                return model_cls.model_validate_json(_extract_json(resp.text))
            except (ValidationError, ValueError) as e:
                if attempt == max_repairs:
                    raise StructuredOutputError(f"{model_cls.__name__}: {e}") from e
                history = [*history, {"role": "user", "content": _text(prompt)},
                           {"role": "assistant", "content": resp.text}]
                fix = f"That reply was invalid: {e}\nReturn corrected JSON only."
                prompt = Prompt(prompt.name, prompt.version, fix) if isinstance(prompt, Prompt) else fix

    def vision(self, prompt: str | Prompt, images: list[Image | bytes], *, model_cls: type[M] | None = None,
               system: str | None = None, max_tokens: int | None = None) -> str | M:
        imgs = [i if isinstance(i, Image) else Image(i) for i in images]
        if model_cls is not None:
            return self.structured(model_cls, prompt, system=system, images=imgs, max_tokens=max_tokens)
        return self._call("vision", self.vision_model, prompt, system, None, imgs, max_tokens).text

    def embed(self, texts: Sequence[str], *, model: str | None = None, dimensions: int | None = None) -> list[list[float]]:
        """One vector per text. Cached per (model, dimensions, text) in the cassette, so a text is embedded - and paid for - once."""
        model, dims = model or self.embedding_model, dimensions or self.embedding_dimensions
        if not model or model.upper() == "TBD":
            raise ConfigError("no embedding model: set llm.embedding_model in configs/app.yaml")
        t0 = time.perf_counter()
        out: list[list[float] | None] = [None] * len(texts)
        keys = [hashlib.sha256(json.dumps(["embed", model, dims, t]).encode()).hexdigest() for t in texts]
        todo: list[int] = []
        for i, key in enumerate(keys):
            hit = self.cassette.get(key) if self.mode in ("replay", "auto") and self.cassette else None
            if hit is not None:
                out[i] = json.loads(hit.text)
            elif self.mode == "replay":
                raise CassetteMiss(f"no recorded embedding for text {i} ({key[:12]}) in {self.cassette.path}")
            else:
                todo.append(i)
        tokens = 0
        for start in range(0, len(todo), EMBED_BATCH):
            batch = todo[start:start + EMBED_BATCH]
            vectors, used = self._embed_with_retry(model, [texts[i] for i in batch], dims)
            tokens += used
            for i, vec in zip(batch, vectors):
                out[i] = [round(x, EMBED_DECIMALS) for x in vec]
                if self.mode in ("record", "auto"):
                    self.cassette.put(keys[i], LLMResponse(json.dumps(out[i]), model, used // max(len(batch), 1)), save=False)  # type: ignore[union-attr]
        if todo and self.mode in ("record", "auto"):
            self.cassette.save()  # type: ignore[union-attr]
        self.calls.append(CallRecord("embed", self.provider.name, model, None, None, tokens, 0, time.perf_counter() - t0, not todo))
        return [v for v in out if v is not None]

    def _embed_with_retry(self, model: str, texts: list[str], dims: int | None) -> tuple[list[list[float]], int]:
        fn = getattr(self.provider, "embed", None)
        if fn is None:
            raise LLMError(f"provider '{self.provider.name}' does not support embeddings")
        for attempt in range(self.max_retries + 1):
            try:
                return fn(model, texts, dims)
            except TransientLLMError:
                if attempt == self.max_retries:
                    raise
                self._sleep(self.backoff_s * 2 ** attempt)
        raise AssertionError("unreachable")

    @property
    def last_call(self) -> CallRecord | None:
        return self.calls[-1] if self.calls else None

    # ---- internals
    def _call(self, kind: str, model: str, prompt: str | Prompt, system: str | None,
              history: list[dict[str, str]] | None, images: list[Image], max_tokens: int | None) -> LLMResponse:
        req = LLMRequest(model=model, system=system, images=list(images), temperature=self.temperature,
                         reasoning_effort=self.reasoning_effort,
                         messages=[*(history or []), {"role": "user", "content": _text(prompt)}], max_tokens=max_tokens)
        key, t0, replayed = req.cache_key(), time.perf_counter(), False
        resp = self.cassette.get(key) if self.mode in ("replay", "auto") and self.cassette else None
        if self.mode == "replay" and resp is None:
            raise CassetteMiss(f"no recorded response for {kind} request {key[:12]} in {self.cassette.path}")
        if resp is not None:
            replayed = True                                    # replay, or 'auto' with a recorded answer: no API call, no cost
        else:
            if model.upper() == "TBD":
                raise ConfigError("llm model is 'TBD' in configs/app.yaml - set text_model / vision_model")
            resp = self._with_retry(req)
            if self.mode in ("record", "auto"):
                self.cassette.put(key, resp)  # type: ignore[union-attr]
        self.calls.append(CallRecord(kind, self.provider.name, resp.model, getattr(prompt, "name", None),
                                     getattr(prompt, "version", None), resp.input_tokens, resp.output_tokens,
                                     time.perf_counter() - t0, replayed))
        log.info("llm %s %s/%s in=%d out=%d replayed=%s", kind, self.provider.name, resp.model,
                 resp.input_tokens, resp.output_tokens, replayed)
        return resp

    def _with_retry(self, req: LLMRequest) -> LLMResponse:
        for attempt in range(self.max_retries + 1):
            try:
                return self.provider.complete(req)
            except TransientLLMError:
                if attempt == self.max_retries:
                    raise
                self._sleep(self.backoff_s * 2 ** attempt)
        raise AssertionError("unreachable")


def _text(prompt: str | Prompt) -> str:
    return prompt.text if isinstance(prompt, Prompt) else prompt


def _extract_json(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found in reply")
    return t[start:end + 1]
