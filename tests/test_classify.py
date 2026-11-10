"""Tests for classify.py. They use a fake model, so they cost nothing and need no Azure sign-in.

Run from the repo folder:  python -m pytest -q
"""
import csv
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


# ---------- the `all` target: the full register ----------

def write_register(path, rows):
    """A tiny register with the real file's header names, stray spaces included."""
    header = ["Publication", "Borough", "Community Board", "Tracking  Code ", "Request ", "Explanation "]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def read_output(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_read_raw_tidies_the_header_names(tmp_path):
    path = tmp_path / "register.csv"
    write_register(path, [["20270217", "Bronx", "07", "207202701C", "Install chargers", "At the library"]])
    row = classify.read_raw(path)[0]
    assert row["Tracking Code"] == "207202701C"
    assert row["Request"] == "Install chargers" and row["Explanation"] == "At the library"


def test_run_all_labels_each_text_once_and_writes_every_row(tmp_path, conn):
    path, out = tmp_path / "register.csv", tmp_path / "all.csv"
    write_register(path, [
        ["20260630", "Bronx", "07", "A1", "Install chargers", "At the library"],
        ["20270217", "Bronx", "07", "A1", "Install chargers", "At the  library"],  # same text, next cycle
        ["20270217", "Queens", "01", "B2", "Fix potholes", "Main St"],
    ])
    client = fake_client()
    stats = classify.run_all(max_calls=10, raw_path=path, out_path=out, client=client, conn=conn)
    assert len(client.responses.calls) == 2 and stats["distinct"] == 2
    written = read_output(out)
    assert [r["publication"] for r in written] == ["20260630", "20270217", "20270217"]
    assert [r["tracking_code"] for r in written] == ["A1", "A1", "B2"]
    assert all(r["pred_label"] == "ev_charging" for r in written)


def test_run_all_stops_cleanly_at_the_limit_and_resumes(tmp_path, conn):
    path, out = tmp_path / "register.csv", tmp_path / "all.csv"
    write_register(path, [["20270217", "Bronx", "07", f"C{n}", "Install chargers", f"Site {n}"] for n in range(3)])
    stats = classify.run_all(max_calls=2, raw_path=path, out_path=out, client=fake_client(), conn=conn)
    assert stats["calls"] == 2 and not out.exists()
    assert conn.execute("SELECT COUNT(*) FROM labels").fetchone()[0] == 2
    client = fake_client()
    classify.run_all(max_calls=10, raw_path=path, out_path=out, client=client, conn=conn)
    assert len(client.responses.calls) == 1 and len(read_output(out)) == 3
