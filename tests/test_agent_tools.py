"""Tests for agent_tools.py. They make no Azure calls, so they cost nothing.

Run from the repo folder:  python -m pytest -q
"""
import json
import sqlite3

import agent_tools


def test_describe_data_returns_source_and_as_of():
    # The spec's rule: every tool returns its source and as-of date.
    result = agent_tools.describe_data()
    assert result["source"]
    assert result["as_of"] == agent_tools.WRITTEN_ON


def test_every_source_has_the_same_fields():
    for source in agent_tools.describe_data()["sources"]:
        assert set(source) == agent_tools.SOURCE_FIELDS, source.get("tool")


ALL_SEVEN = {
    "describe_data", "classify_request", "board_requests", "public_chargers",
    "ev_registrations", "home_charging_need", "gap_ranking",
}


def test_describe_data_covers_all_seven_tools():
    tools = {source["tool"] for source in agent_tools.SOURCES} | {"describe_data"}
    assert tools == ALL_SEVEN


def test_built_and_not_built_lists_cover_every_tool_once():
    result = agent_tools.describe_data()
    built, not_built = set(result["tools_built_now"]), set(result["tools_not_built"])
    assert not built & not_built
    assert built | not_built == ALL_SEVEN


def test_status_text_agrees_with_what_is_built():
    # When a tool gets built, its status must stop saying "Not built" (and the reverse).
    built = set(agent_tools.TOOL_FUNCTIONS)
    for source in agent_tools.SOURCES:
        says_not_built = source["tool_status"].startswith("Not built")
        assert says_not_built == (source["tool"] not in built), source["tool"]


def test_result_survives_json():
    # ask_agent.py sends results to the agent as JSON text, so they must convert cleanly.
    result = agent_tools.describe_data()
    assert json.loads(json.dumps(result)) == result


def test_classify_request_feeds_the_pipeline_the_right_input(monkeypatch):
    # Stand-ins for classify.py's Azure and cache parts, so no model is called.
    seen = {}

    def fake_classify_texts(client, conn, prompt, texts, max_calls):
        seen.update(prompt=prompt, texts=texts, max_calls=max_calls)
        answer = {"label": "ev_charging", "parking_concern": "no", "reason": "asks for chargers",
                  "from_cache": False, "error": ""}
        stats = {"calls": 1, "cache_hits": 0, "input_tokens": 900, "cached_input_tokens": 0,
                 "output_tokens": 40}
        return {texts[0]: answer}, stats

    monkeypatch.setattr(agent_tools.classify, "make_client", lambda: None)
    monkeypatch.setattr(agent_tools.classify, "open_cache", lambda: sqlite3.connect(":memory:"))
    monkeypatch.setattr(agent_tools.classify, "classify_texts", fake_classify_texts)

    result = agent_tools.classify_request("We need EV chargers on our streets.", "")
    # The board's words go in the Explanation line, as in the pipeline.
    assert seen["texts"] == ["Request: \nExplanation: We need EV chargers on our streets."]
    assert seen["prompt"] == agent_tools.classify.load_prompt("v2")  # the frozen prompt
    assert seen["max_calls"] == 1                                      # the spending brake
    assert result["label"] == "ev_charging"
    assert result["source"] and result["as_of"]


def test_every_tool_definition_has_a_function():
    # A tool Foundry knows about but this script can't run would crash ask_agent.py.
    defined = {tool.name for tool in agent_tools.TOOL_DEFINITIONS}
    assert defined == set(agent_tools.TOOL_FUNCTIONS)
