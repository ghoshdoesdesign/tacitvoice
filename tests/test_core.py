import json
from types import SimpleNamespace

import numpy as np
import pytest

from server import cleanup, config
from server.lips import LipResult, decode_nbest, resample_indices, to_model_input
from server.phrases import Phrasebook
from server.stt import SttResult, pcm16_to_float, peak_dbfs, resample
from server.text import corpus_wer, normalise, word_error_rate


def test_wer():
    assert word_error_rate("Summarise this thread.", "summarise this thread") == 0
    assert word_error_rate("a b c d", "a x c") == 0.5
    assert word_error_rate("", "") == 0
    assert corpus_wer([("a b", "a b"), ("c d", "c x")]) == 0.25
    assert normalise("Don’t  STOP!") == "don't stop"


def test_resample_30fps_to_25():
    idx = resample_indices([i / 30 for i in range(300)], 300, None)
    assert abs(len(idx) - 250) <= 1
    assert idx[0] == 0 and idx[-1] <= 299 and (np.diff(idx) >= 1).all()
    assert list(resample_indices(None, 25, 25.0)) == list(range(25))
    assert len(resample_indices(None, 0, 30)) == 0
    assert list(resample_indices(None, 1, 30)) == [0]


def test_resample_uses_timestamps_not_count():
    # A dropped-frame gap must not shift later frames in time.
    ts = [0.0, 0.04, 0.08, 0.48, 0.52]
    idx = resample_indices(ts, 5, None)
    assert len(idx) == 14 and idx[-1] == 4 and idx[5] in (2, 3)


def test_model_input_shape_and_range():
    x = to_model_input(np.full((7, 96, 96), 128, dtype=np.uint8))
    assert x.shape == (7, 1, 88, 88) and x.dtype == np.float32
    assert x.mean() == pytest.approx((128 / 255 - 0.421) / 0.165, abs=1e-4)


def test_decode_nbest_dedupes_and_cleans():
    tokens = ["<blank>", "▁HELLO", "▁WORLD", "S", "<eos>"]
    hyps = [
        {"yseq": [4, 1, 2, 4], "score": -1.0},
        {"yseq": [4, 1, 2, 4], "score": -1.5},
        {"yseq": [4, 1, 2, 3, 4], "score": -2.0},
        {"yseq": [4, 4], "score": -9.0},
    ]
    assert decode_nbest(hyps, tokens, 5) == [("hello world", -1.0), ("hello worlds", -2.0)]
    assert len(decode_nbest(hyps, tokens, 1)) == 1


def test_audio_helpers():
    pcm = (np.sin(np.linspace(0, 100, 16000)) * 16000).astype("<i2").tobytes()
    audio = pcm16_to_float(pcm + b"\x00")  # odd trailing byte is ignored
    assert len(audio) == 16000 and -10 < peak_dbfs(audio) < -8
    assert peak_dbfs(np.zeros(16000, dtype=np.float32)) < -100
    assert len(resample(np.zeros(48000, dtype=np.float32), 48000)) == 16000


def test_phrasebook(tmp_path):
    d, p = tmp_path / "dictionary.txt", tmp_path / "prompts.txt"
    d.write_text("# names\nOpenClaw\nLyra\n")
    p.write_text("Summarise this thread and list the action items\nDraft a reply saying I'll be late\n")
    book = Phrasebook(d, p)
    assert book.glossary == ["OpenClaw", "Lyra"]
    hits = book.similar(["summer eyes this red and list the action item"])
    assert hits and hits[0][0].startswith("Summarise this thread")
    assert book.similar(["completely unrelated words about fishing boats"], min_score=90) == []
    book.remember("Rewrite this paragraph to be shorter")
    book.remember("Rewrite this paragraph to be shorter")
    book.remember("ok")  # too short to be worth keeping
    assert p.read_text().count("Rewrite this paragraph") == 1 and len(book.prompts) == 3


def stt(text="", probs=()):
    return SttResult(text=text, words=list(zip(text.split(), probs)), audible=bool(text))


def test_raw_pick():
    lips = LipResult(candidates=[("some eyes this red", -3.0), ("summarise this bread", -4.0)])
    assert cleanup.raw_pick(stt("summarise this thread", (0.9, 0.9, 0.9)), lips, []).text == "summarise this thread"
    assert cleanup.raw_pick(stt(), lips, [("Summarise this thread", 90.0)]).text == "Summarise this thread"
    assert cleanup.raw_pick(stt(), lips, [("Summarise this thread", 60.0)]).text == "some eyes this red"
    assert cleanup.raw_pick(stt(), LipResult(), []).text == ""


def test_evidence_shape():
    lips = LipResult(candidates=[("a b", -1.0)])
    e = cleanup.build_evidence(stt(), lips, ["Lyra"], [("A b c", 71.4)], "Slack", "x" * 1000)
    assert e["audio"] is None and e["lips"] == ["a b"] and len(e["text_before_cursor"]) == 600
    assert e["similar_prompts"] == [{"text": "A b c", "score": 71}]
    e = cleanup.build_evidence(stt("hi there", (0.5, 0.91)), lips, [], [], "", "")
    assert e["audio"] == {"text": "hi there", "words": [["hi", 0.5], ["there", 0.91]]}


class FakeMessages:
    def __init__(self, reply=None, error=None, stop="end_turn"):
        self.reply, self.error, self.stop, self.kwargs = reply, error, stop, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        content = [SimpleNamespace(type="text", text=self.reply)] if self.reply is not None else []
        return SimpleNamespace(stop_reason=self.stop, content=content)


class FakeCompletions(FakeMessages):
    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        message = SimpleNamespace(content=self.reply, refusal=None)
        finish = "stop" if self.stop == "end_turn" else self.stop
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish, message=message)])


def anthropic_cleaner(fake):
    import anthropic

    c = cleanup.AnthropicCleaner.__new__(cleanup.AnthropicCleaner)
    c._sdk, c._client, c.model = anthropic, SimpleNamespace(messages=fake), "claude-haiku-5-5"
    return c


def openai_cleaner(fake):
    import openai

    c = cleanup.OpenAICleaner.__new__(cleanup.OpenAICleaner)
    chat = SimpleNamespace(completions=fake)
    c._sdk, c._client, c.model = openai, SimpleNamespace(chat=chat), "gpt-4.1-mini"
    return c


PROVIDERS = [(anthropic_cleaner, FakeMessages), (openai_cleaner, FakeCompletions)]
GOOD = json.dumps({"text": "Summarise this thread.", "alternatives": ["Summarise this thread.", "Summarise this bread."]})


@pytest.mark.parametrize("make,fake", PROVIDERS)
def test_cleaner_success(make, fake):
    out = make(fake(GOOD)).clean({"audio": None}, cleanup.Cleaned(text="raw"))
    assert out.text == "Summarise this thread." and out.source == "model"
    assert out.alternatives == ["Summarise this bread."]


def test_request_shapes():
    m = FakeMessages(GOOD)
    anthropic_cleaner(m).clean({}, cleanup.Cleaned(text="raw"))
    assert m.kwargs["output_config"]["format"]["type"] == "json_schema"
    assert "temperature" not in m.kwargs and m.kwargs["thinking"] == {"type": "disabled"}
    f = FakeCompletions(GOOD)
    openai_cleaner(f).clean({"app": "Slack"}, cleanup.Cleaned(text="raw"))
    assert f.kwargs["response_format"]["json_schema"]["strict"] is True
    assert [m["role"] for m in f.kwargs["messages"]] == ["system", "user"]
    assert json.loads(f.kwargs["messages"][1]["content"]) == {"app": "Slack"}


def api_error(sdk_name, cls_name, status):
    import importlib

    import httpx

    sdk = importlib.import_module(sdk_name)
    response = httpx.Response(status, request=httpx.Request("POST", "https://example.invalid"))
    return getattr(sdk, cls_name)("nope", response=response, body=None)


@pytest.mark.parametrize("make,fake", PROVIDERS)
@pytest.mark.parametrize(
    "kwargs",
    [
        {"reply": "not json"},
        {"reply": json.dumps({"text": "  ", "alternatives": []})},
        {"reply": json.dumps({"text": "x", "alternatives": []}), "stop": "refusal"},
        {"reply": None},
        {"error": RuntimeError("no credentials")},
    ],
)
def test_cleaner_falls_back_to_raw(make, fake, kwargs):
    out = make(fake(**kwargs)).clean({}, cleanup.Cleaned(text="raw"))
    assert out.text == "raw" and out.source == "raw" and out.note


@pytest.mark.parametrize("make,fake,sdk", [(anthropic_cleaner, FakeMessages, "anthropic"), (openai_cleaner, FakeCompletions, "openai")])
def test_cleaner_reports_api_errors(make, fake, sdk):
    out = make(fake(error=api_error(sdk, "AuthenticationError", 401))).clean({}, cleanup.Cleaned(text="raw"))
    assert out.text == "raw" and "API key" in out.note
    out = make(fake(error=api_error(sdk, "RateLimitError", 429))).clean({}, cleanup.Cleaned(text="raw"))
    assert "rate limited" in out.note


def test_provider_follows_the_key_present(monkeypatch):
    import importlib

    def provider(env):
        for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "TACIT_CLEANUP_PROVIDER", "TACIT_CLEANUP_MODEL"):
            monkeypatch.delenv(k, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        c = importlib.reload(config)
        return c.CLEANUP_PROVIDER, c.CLEANUP_MODEL

    assert provider({"OPENAI_API_KEY": "x"}) == ("openai", "gpt-4.1-mini")
    assert provider({"ANTHROPIC_API_KEY": "x", "OPENAI_API_KEY": "y"}) == ("anthropic", "claude-haiku-5-5")
    assert provider({}) == ("anthropic", "claude-haiku-5-5")
    assert provider({"TACIT_CLEANUP_PROVIDER": "openai", "TACIT_CLEANUP_MODEL": "m"}) == ("openai", "m")
    provider({})


def test_token_is_private_and_stable(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HOME", tmp_path)
    monkeypatch.setattr(config, "TOKEN_FILE", tmp_path / "token")
    token = config.local_token()
    assert len(token) > 30 and config.local_token() == token
    assert (tmp_path / "token").stat().st_mode & 0o777 == 0o600
