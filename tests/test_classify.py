"""Tests for classify.py. They use a fake model, so they cost nothing and need no Azure sign-in.

Run from the repo folder:  python -m pytest -q
"""
import types

import httpx2
import pytest
from openai import BadRequestError

import classify

PROMPT = "test prompt"


class FakeResponses:
    """Stands in for client.responses: counts calls and returns a fixed answer."""

    def __init__(self, reject=(), refuse=()):
        self.calls = []
        self.reject, self.refuse = set(reject), set(refuse)

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        text = kwargs["input"]
        if text in self.reject:
            request = httpx2.Request("POST", "https://example.test/responses")
            raise BadRequestError("content filter", response=httpx2.Response(400, request=request), body=None)
        usage = types.SimpleNamespace(
            input_tokens=100, output_tokens=20,
            input_tokens_details=types.SimpleNamespace(cached_tokens=0))
        parsed = None if text in self.refuse else classify.Label(
            reason="Asks for curbside chargers.", label="ev_charging", parking_concern="no")
        return types.SimpleNamespace(output_parsed=parsed, output_text="I can't help with that.", usage=usage)


def fake_client(**kwargs):
    return types.SimpleNamespace(responses=FakeResponses(**kwargs))


@pytest.fixture
def conn(tmp_path):
    return classify.open_cache(tmp_path / "cache.db")


def test_normalize_collapses_whitespace():
    assert classify.normalize("  Install\n chargers\tnow ") == "Install chargers now"


def test_cache_key_ignores_spacing_but_not_prompt_or_text():
    a = classify.cache_key(PROMPT, classify.model_input("R", "Install  chargers"))
    b = classify.cache_key(PROMPT, classify.model_input("R", "Install chargers"))
    assert a == b
    assert a != classify.cache_key("another prompt", classify.model_input("R", "Install chargers"))
    assert a != classify.cache_key(PROMPT, classify.model_input("R", "Install meters"))


def test_repeated_text_is_sent_once(conn):
    client = fake_client()
    results, stats = classify.classify_texts(client, conn, PROMPT, ["a", "a", "b"])
    assert len(client.responses.calls) == 2
    assert stats["calls"] == 2 and set(results) == {"a", "b"}


def test_second_run_uses_the_cache(conn):
    classify.classify_texts(fake_client(), conn, PROMPT, ["a", "b"])
    client = fake_client()
    results, stats = classify.classify_texts(client, conn, PROMPT, ["a", "b"])
    assert client.responses.calls == []
    assert stats["cache_hits"] == 2 and results["a"]["from_cache"] is True


def test_no_cache_asks_again_and_leaves_cache_alone(conn):
    classify.classify_texts(fake_client(), conn, PROMPT, ["a"])
    client = fake_client()
    classify.classify_texts(client, conn, PROMPT, ["a"], use_cache=False)
    assert len(client.responses.calls) == 1
    assert conn.execute("SELECT COUNT(*) FROM labels").fetchone()[0] == 1


def test_call_limit_stops_before_going_over(conn):
    client = fake_client()
    with pytest.raises(classify.TooManyCalls):
        classify.classify_texts(client, conn, PROMPT, ["a", "b", "c"], max_calls=2)
    assert len(client.responses.calls) == 2


def test_rejected_request_is_recorded_and_not_cached(conn):
    results, stats = classify.classify_texts(fake_client(reject={"b"}), conn, PROMPT, ["a", "b"])
    assert stats["errors"] == 1 and results["b"]["error"] and results["a"]["label"] == "ev_charging"
    assert classify.cache_get(conn, classify.cache_key(PROMPT, "b")) is None


def test_missing_answer_is_an_error(conn):
    results, stats = classify.classify_texts(fake_client(refuse={"a"}), conn, PROMPT, ["a"])
    assert stats["errors"] == 1 and "no parsed answer" in results["a"]["error"]


def test_request_settings(conn):
    client = fake_client()
    classify.classify_texts(client, conn, PROMPT, ["a"])
    sent = client.responses.calls[0]
    assert sent["model"] == classify.DEPLOYMENT and sent["instructions"] == PROMPT
    assert sent["text_format"] is classify.Label and sent["store"] is False
    assert sent["reasoning"] == {"effort": "none"}


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_prompt_file_has_every_label_and_ruling(version):
    prompt = classify.prompt_path(version).read_text(encoding="utf-8")
    for label in ("ev_charging", "parking_or_curb", "clean_air", "other", "parking_concern"):
        assert label in prompt
    for n in range(1, 10):
        assert f"\n{n}. " in prompt


def test_v2_has_the_nov_5_rule_changes():
    v1 = classify.prompt_path("v1").read_text(encoding="utf-8")
    v2 = classify.prompt_path("v2").read_text(encoding="utf-8")
    for added in ("including ferries", "Asking for chargers at the curb or in a parking lot doesn't count by itself"):
        assert added in v2 and added not in v1


def test_prompt_versions_get_different_cache_keys():
    v1 = classify.prompt_path("v1").read_text(encoding="utf-8")
    v2 = classify.prompt_path("v2").read_text(encoding="utf-8")
    assert classify.cache_key(v1, "same text") != classify.cache_key(v2, "same text")


def test_frozen_prompt_is_unchanged():
    assert classify.load_prompt("v2")  # raises SystemExit if the file was edited


def test_edited_frozen_prompt_is_refused(tmp_path, monkeypatch):
    edited = tmp_path / "classify_v2.md"
    edited.write_text(classify.prompt_path("v2").read_text(encoding="utf-8") + "one more rule\n", encoding="utf-8")
    monkeypatch.setattr(classify, "PROMPTS_DIR", tmp_path)
    with pytest.raises(SystemExit):
        classify.load_prompt("v2")
