import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pydantic import BaseModel

from claimtower.core.config import load_settings
from claimtower.core.errors import ConfigError
from claimtower.core.llm import (
    PROVIDERS, Cassette, CassetteMiss, Image, LLMClient, LLMError, LLMRequest, LLMResponse, OpenAIProvider,
    StructuredOutputError, TransientLLMError, load_prompt,
)

REPO = Path(__file__).resolve().parents[2]


class Answer(BaseModel):
    verdict: str
    score: int


class FakeProvider:
    name = "fake"

    def __init__(self, *replies):
        self.replies, self.requests = list(replies), []

    def complete(self, req):
        self.requests.append(req)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return LLMResponse(text=r, model="fake-1", input_tokens=10, output_tokens=5)


def client(provider, **kw):
    return LLMClient(provider, text_model="t", vision_model="v", sleep=lambda s: None, **kw)


# ---------------------------------------------------------------- structured / chat / vision
def test_chat_and_call_log():
    c = client(FakeProvider("hello"))
    assert c.chat("hi") == "hello"
    rec = c.last_call
    assert (rec.kind, rec.provider, rec.model, rec.input_tokens, rec.replayed) == ("chat", "fake", "fake-1", 10, False)


def test_structured_parses_fenced_json():
    c = client(FakeProvider('```json\n{"verdict": "ok", "score": 3}\n```'))
    assert c.structured(Answer, "q") == Answer(verdict="ok", score=3)


def test_structured_repairs_once_then_fails():
    p = FakeProvider("not json", '{"verdict": "ok", "score": "x"}')
    with pytest.raises(StructuredOutputError):
        client(p).structured(Answer, "q")
    assert len(p.requests) == 2
    assert "invalid" in p.requests[1].messages[-1]["content"]
    p = FakeProvider("garbage", '{"verdict": "ok", "score": 1}')
    assert client(p).structured(Answer, "q").score == 1


def test_vision_uses_vision_model_and_attaches_images():
    p = FakeProvider('{"verdict": "dry", "score": 1}')
    out = client(p).vision("describe", [b"\x89PNG"], model_cls=Answer)
    assert out.verdict == "dry"
    assert p.requests[0].model == "v" and p.requests[0].images == [Image(b"\x89PNG")]


def test_transient_errors_retried_with_backoff_then_raised():
    sleeps = []
    c = LLMClient(FakeProvider(TransientLLMError("429"), TransientLLMError("429"), "fine"),
                  text_model="t", vision_model="v", sleep=sleeps.append, backoff_s=1.0)
    assert c.chat("x") == "fine" and sleeps == [1.0, 2.0]
    c = LLMClient(FakeProvider(*[TransientLLMError("x")] * 4), text_model="t", vision_model="v",
                  sleep=lambda s: None, max_retries=3)
    with pytest.raises(TransientLLMError):
        c.chat("x")


def test_tbd_model_refused_for_live_calls():
    c = LLMClient(FakeProvider("x"), text_model="TBD", vision_model="TBD")
    with pytest.raises(ConfigError, match="TBD"):
        c.chat("hi")


# ---------------------------------------------------------------- cassettes
def test_record_then_replay_is_deterministic_and_offline(tmp_path):
    path = tmp_path / "c.json"
    rec = client(FakeProvider("answer-1"), mode="record", cassette=Cassette(path))
    assert rec.chat("same question") == "answer-1"
    assert path.is_file()

    offline = FakeProvider()                      # would crash if called
    rep = client(offline, mode="replay", cassette=Cassette(path))
    assert rep.chat("same question") == "answer-1"
    assert rep.last_call.replayed and offline.requests == []
    with pytest.raises(CassetteMiss):
        rep.chat("a different question")


def test_auto_mode_replays_what_is_recorded_and_records_what_is_new(tmp_path):
    path = tmp_path / "auto.json"
    first = FakeProvider("answer-A", "answer-B")
    c1 = client(first, mode="auto", cassette=Cassette(path))
    assert c1.chat("q1") == "answer-A" and c1.chat("q2") == "answer-B" and len(first.requests) == 2
    second = FakeProvider("answer-C")
    c2 = client(second, mode="auto", cassette=Cassette(path))
    assert c2.chat("q1") == "answer-A" and second.requests == [] and c2.last_call.replayed      # recorded: no call
    assert c2.chat("q3") == "answer-C" and len(second.requests) == 1 and not c2.last_call.replayed   # new: one call, then recorded
    assert client(FakeProvider(), mode="replay", cassette=Cassette(path)).chat("q3") == "answer-C"


def test_replay_requires_cassette():
    with pytest.raises(ValueError):
        client(FakeProvider(), mode="replay")


# ---------------------------------------------------------------- prompts
def make_pack(tmp_path):
    d = tmp_path / "prompts"
    d.mkdir()
    return SimpleNamespace(prompts_dir=d), d


def test_prompt_loader_version_and_variables(tmp_path):
    pack, d = make_pack(tmp_path)
    (d / "gap.md").write_text("---\nversion: 1.2\n---\nHello {{ who }}!\n", encoding="utf-8")
    p = load_prompt("gap", pack, who="adjuster")
    assert (p.name, p.version, p.text.strip()) == ("gap", "1.2", "Hello adjuster!")
    with pytest.raises(ConfigError, match="missing prompt variables"):
        load_prompt("gap", pack)
    (d / "bad.md").write_text("no front matter", encoding="utf-8")
    with pytest.raises(ConfigError, match="version"):
        load_prompt("bad", pack)
    with pytest.raises(ConfigError, match="not found"):
        load_prompt("nope", pack)


def test_prompt_version_recorded_in_call_log(tmp_path):
    pack, d = make_pack(tmp_path)
    (d / "p.md").write_text("---\nversion: 7\n---\nq", encoding="utf-8")
    c = client(FakeProvider("a"))
    c.chat(load_prompt("p", pack))
    assert (c.last_call.prompt_name, c.last_call.prompt_version) == ("p", "7")


# ---------------------------------------------------------------- OpenAI provider (SDK faked)
class FakeOpenAI:
    def __init__(self, exc=None):
        self.exc, self.kwargs = exc, None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        if self.exc:
            raise self.exc
        self.kwargs = kw
        return SimpleNamespace(model="gpt-test", choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                               usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3))


def test_openai_provider_request_and_usage_mapping():
    fake = FakeOpenAI()
    c = client(OpenAIProvider(client=fake))
    assert c.vision("look", [Image(b"abc", "image/jpeg")], system="sys") == "ok"
    msgs = fake.kwargs["messages"]
    assert msgs[0] == {"role": "system", "content": "sys"}
    assert msgs[1]["content"][0] == {"type": "text", "text": "look"}
    assert msgs[1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert fake.kwargs["model"] == "v" and (c.last_call.input_tokens, c.last_call.output_tokens) == (7, 3)


def test_openai_errors_mapped():
    class RateLimitError(Exception): ...
    req = LLMRequest(model="m", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(TransientLLMError):
        OpenAIProvider(client=FakeOpenAI(RateLimitError("slow"))).complete(req)
    with pytest.raises(LLMError):
        OpenAIProvider(client=FakeOpenAI(ValueError("bad"))).complete(req)


def test_temperature_is_not_sent_unless_set_and_reasoning_settings_are_forwarded():
    """gpt-5-mini accepts only its default temperature: the wrapper must omit it, send max_completion_tokens, and pass reasoning_effort."""
    fake = FakeOpenAI()
    c = LLMClient(OpenAIProvider(client=fake), text_model="t", vision_model="v", reasoning_effort="minimal", sleep=lambda s: None)
    c.chat("hi", max_tokens=500)
    assert "temperature" not in fake.kwargs and "max_tokens" not in fake.kwargs
    assert fake.kwargs["reasoning_effort"] == "minimal" and fake.kwargs["max_completion_tokens"] == 500
    c2 = LLMClient(OpenAIProvider(client=fake), text_model="t", vision_model="v", temperature=0.2, sleep=lambda s: None)
    c2.chat("hi")
    assert fake.kwargs["temperature"] == 0.2 and "reasoning_effort" not in fake.kwargs and "max_completion_tokens" not in fake.kwargs


def test_reasoning_settings_are_part_of_the_cassette_key():
    base = LLMRequest(model="m", messages=[{"role": "user", "content": "x"}])
    assert base.cache_key() != LLMRequest(model="m", messages=base.messages, reasoning_effort="low").cache_key()
    assert base.cache_key() != LLMRequest(model="m", messages=base.messages, temperature=1.0).cache_key()


def test_the_shipped_config_uses_gpt_5_mini_without_a_temperature():
    cfg = load_settings().app.llm
    assert (cfg.provider, cfg.text_model, cfg.vision_model, cfg.temperature) == ("openai", "gpt-5-mini", "gpt-5-mini", None)
    assert cfg.reasoning_effort in ("minimal", "low", "medium", "high")


def test_missing_api_key_fails_clearly(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMError, match="LLM_API_KEY|openai SDK"):
        OpenAIProvider()._get_client()


# ---------------------------------------------------------------- embeddings (T3.4)
class EmbedProvider:
    name = "fake"

    def __init__(self, fail_first=0):
        self.calls, self.fail_first = [], fail_first

    def embed(self, model, texts, dimensions=None):
        self.calls.append((model, list(texts), dimensions))
        if self.fail_first:
            self.fail_first -= 1
            raise TransientLLMError("429")
        return [[len(t) / 7.0, 1 / 3, 0.123456789] for t in texts], 11


def embed_client(provider, **kw):
    return LLMClient(provider, text_model="t", vision_model="v", embedding_model="emb-1", embedding_dimensions=3,
                     sleep=lambda s: None, **kw)


def test_embed_returns_one_vector_per_text_rounded_and_logs_a_call():
    c = embed_client(EmbedProvider())
    out = c.embed(["a", "bbbb"])
    assert out == [[0.14286, 0.33333, 0.12346], [0.57143, 0.33333, 0.12346]]
    assert (c.last_call.kind, c.last_call.model, c.last_call.input_tokens, c.last_call.replayed) == ("embed", "emb-1", 11, False)


def test_embed_is_cached_per_text_in_the_cassette_and_replays_offline(tmp_path):
    path = tmp_path / "e.json"
    p1 = EmbedProvider()
    c1 = embed_client(p1, mode="auto", cassette=Cassette(path))
    first = c1.embed(["a", "bb"])
    assert len(p1.calls) == 1
    p2 = EmbedProvider()
    c2 = embed_client(p2, mode="auto", cassette=Cassette(path))
    assert c2.embed(["bb", "a"]) == [first[1], first[0]] and p2.calls == [] and c2.last_call.replayed
    assert c2.embed(["a", "new text"])[0] == first[0] and p2.calls == [("emb-1", ["new text"], 3)]        # only the new text is sent
    rep = embed_client(EmbedProvider(), mode="replay", cassette=Cassette(path))
    assert rep.embed(["new text"]) and rep.last_call.replayed
    with pytest.raises(CassetteMiss):
        rep.embed(["never recorded"])


def test_the_embedding_model_and_dimensions_are_part_of_the_cache_key(tmp_path):
    path = tmp_path / "e.json"
    p = EmbedProvider()
    c = embed_client(p, mode="auto", cassette=Cassette(path))
    c.embed(["a"])
    c.embed(["a"], model="emb-2")
    c.embed(["a"], dimensions=8)
    assert [(m, d) for m, _, d in p.calls] == [("emb-1", 3), ("emb-2", 3), ("emb-1", 8)]


def test_embed_batches_large_requests(monkeypatch):
    import claimtower.core.llm as mod

    monkeypatch.setattr(mod, "EMBED_BATCH", 2)
    p = EmbedProvider()
    assert len(embed_client(p).embed(["a", "b", "c", "d", "e"])) == 5 and [len(t) for _, t, _ in p.calls] == [2, 2, 1]


def test_embed_retries_transient_errors_then_raises():
    assert len(embed_client(EmbedProvider(fail_first=2)).embed(["a"])) == 1
    with pytest.raises(TransientLLMError):
        embed_client(EmbedProvider(fail_first=9), max_retries=2).embed(["a"])


def test_embed_needs_a_model_and_a_provider_that_supports_it():
    with pytest.raises(ConfigError, match="embedding model"):
        LLMClient(EmbedProvider(), text_model="t", vision_model="v").embed(["a"])
    with pytest.raises(LLMError, match="does not support embeddings"):
        embed_client(FakeProvider()).embed(["a"])


def test_openai_provider_maps_the_embedding_request_and_orders_the_vectors():
    class Emb:
        def create(self, **kw):
            self.kw = kw
            data = [SimpleNamespace(index=1, embedding=[0.2]), SimpleNamespace(index=0, embedding=[0.1])]
            return SimpleNamespace(data=data, usage=SimpleNamespace(prompt_tokens=4))

    fake = SimpleNamespace(embeddings=Emb())
    vectors, tokens = OpenAIProvider(client=fake).embed("m", ["x", "y"], 256)
    assert vectors == [[0.1], [0.2]] and tokens == 4 and fake.embeddings.kw == {"model": "m", "input": ["x", "y"], "dimensions": 256}
    OpenAIProvider(client=fake).embed("m", ["x"])
    assert "dimensions" not in fake.embeddings.kw


def test_the_shipped_config_names_an_embedding_model():
    cfg = load_settings().app.llm
    assert cfg.embedding_model and cfg.embedding_dimensions
    assert LLMClient.from_settings(load_settings()).embedding_model == cfg.embedding_model


def test_derive_changes_one_setting_for_one_task_and_shares_the_cassette_and_call_log(tmp_path):
    base = LLMClient(FakeProvider("a", "b"), text_model="t", vision_model="v", reasoning_effort="minimal", mode="auto",
                     cassette=Cassette(tmp_path / "c.json"), sleep=lambda s: None)
    careful = base.derive(reasoning_effort="low")
    assert (base.reasoning_effort, careful.reasoning_effort) == ("minimal", "low")
    base.chat("q1")
    careful.chat("q1")                                   # a different setting is a different request: not served from the other's recording
    assert len(base.calls) == 2 and base.cassette is careful.cassette and base.provider.requests[-1].reasoning_effort == "low"
    with pytest.raises(ValueError, match="cannot override"):
        base.derive(mode="off")


# ---------------------------------------------------------------- provider selection via config
def test_provider_switch_is_config_only(tmp_path):
    assert {"openai", "azure_openai"} <= set(PROVIDERS.names())
    assert LLMClient.from_settings(load_settings()).provider.name == "openai"
    shutil.copytree(REPO / "configs", tmp_path / "configs")
    shutil.copytree(REPO / "packs", tmp_path / "packs")
    f = tmp_path / "configs/app.yaml"
    d = yaml.safe_load(f.read_text(encoding="utf-8"))
    d["llm"]["provider"] = "azure_openai"
    d["flags"]["llm_replay_mode"] = True
    f.write_text(yaml.safe_dump(d), encoding="utf-8")
    c = LLMClient.from_settings(load_settings(tmp_path))
    assert (c.provider.name, c.mode) == ("azure_openai", "replay")
