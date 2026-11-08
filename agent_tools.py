"""agent_tools.py: the tools ev-gap-agent can call.

Each tool has two halves:
- a Python function, which runs on your Mac when ask_agent.py runs it;
- a FunctionTool, which tells the agent the function's name, what it does,
  and its arguments. create_agent.py sends these to Foundry.

Built so far: describe_data and classify_request. The other five tools in the
agent spec get added here, one at a time, as their data is ready.
"""

from azure.ai.projects.models import FunctionTool

import classify

WRITTEN_ON = "2026-11-07"  # the date describe_data's text was last checked against the handoff

# Every entry in SOURCES has exactly these fields (tests/test_agent_tools.py checks it).
SOURCE_FIELDS = {"name", "tool", "what_it_is", "dataset", "as_of", "geography", "known_limits", "tool_status"}

SOURCES = [
    {
        "name": "Community board budget requests",
        "tool": "board_requests",
        "what_it_is": "What each community board asked the city to fund, in the board's own words, "
        "with the agency's response. 95,389 rows across nine fiscal-year cycles.",
        "dataset": "Register of Community Board Budget Requests, NYC Open Data vn4m-mk4t",
        "as_of": "Downloaded Oct 30, 2026 (the date in the file name)",
        "geography": "Community district. This file numbers boroughs alphabetically (1 Bronx, "
        "2 Brooklyn, 3 Manhattan, 4 Queens, 5 Staten Island; checked Nov 7 against board counts "
        "and borough names in the request text), unlike the district codes the tools use.",
        "known_limits": [
            "Evidence of what boards asked for, not a measure of demand. About 8 of 3,794 distinct "
            "FY2027 request texts are about EV charging (keyword count), from 7 of 59 districts: "
            "too few to rank districts by.",
            "The same text often repeats across fiscal years, so requests are counted by distinct text.",
            "Two FY2027 publications (20260630 and 20270217) carry different agency responses.",
            "Council district is filled in for only 2,447 of the 95,389 rows, and the coordinates "
            "can't stand in for it because they're wrong for Manhattan, Brooklyn and the Bronx, so "
            "there are no answers by council district.",
            "Latitude, longitude and NTA are wrong for Manhattan, Brooklyn and the Bronx, so there are "
            "no answers by neighborhood or coordinates.",
        ],
        "tool_status": "Not built: requests can't be searched or quoted yet. Labels exist only for "
        "the 35 hand-labeled dev requests; the full labeling run hasn't been done.",
    },
    {
        "name": "Public chargers",
        "tool": "public_chargers",
        "what_it_is": "Public charging stations and ports by level (Level 2, DC fast) per district.",
        "dataset": "US DOE Alternative Fuels Data Center station locator; NYC PlugNYC municipal "
        "lots and garages, NYC Open Data kj7g-u4gp",
        "as_of": "PlugNYC updated Sep 9, 2026; AFDC not recorded yet",
        "geography": "Station locations, to be placed in community districts by point-in-polygon "
        "on the district boundary file (5crt-au7u).",
        "known_limits": [
            "Not checked whether PlugNYC sites are already in AFDC (possible double counting).",
            "Not checked whether kj7g-u4gp has charger usage fields.",
        ],
        "tool_status": "Not built: the data hasn't been pulled.",
    },
    {
        "name": "EV registrations",
        "tool": "ev_registrations",
        "what_it_is": "Battery-electric and plug-in hybrid vehicles registered per district.",
        "dataset": "NYS DMV vehicle registrations, data.ny.gov w4pv-hbkt",
        "as_of": "Updated Nov 3, 2026 (dataset page, checked Nov 5)",
        "geography": "ZIP code. Needs a ZIP-to-district crosswalk, so district counts will be estimates.",
        "known_limits": [
            "Counts only people who already own an EV; few chargers can mean few EVs.",
            "The fuel-type values haven't been checked yet.",
        ],
        "tool_status": "Not built: fuel-type check and ZIP-to-district crosswalk not done.",
    },
    {
        "name": "Home-charging need",
        "tool": "home_charging_need",
        "what_it_is": "Households, households with a vehicle, share renting, and share in buildings "
        "of 5+ units (a proxy for having no home charger).",
        "dataset": "US Census American Community Survey 5-year estimates",
        "as_of": "Survey years not chosen yet",
        "geography": "Not decided: whether census tracts or PUMAs can be mapped to districts.",
        "known_limits": [
            "A proxy: the census doesn't ask about driveways or home chargers.",
            "The census tables haven't been chosen.",
        ],
        "tool_status": "Not built.",
    },
    {
        "name": "Gap ranking",
        "tool": "gap_ranking",
        "what_it_is": "Districts ranked by a gap ratio, showing each ratio's numerator and denominator.",
        "dataset": "Built from public_chargers, ev_registrations and home_charging_need",
        "as_of": "Same as its inputs",
        "geography": "Community district or borough; borough totals add the counts before dividing.",
        "known_limits": [
            "What 'underserved' means isn't decided yet: registered EVs per public port, or "
            "car-owning households likely without home charging per public port.",
        ],
        "tool_status": "Not built: waits on the three tools above and the 'underserved' decision.",
    },
    {
        "name": "Request classifier",
        "tool": "classify_request",
        "what_it_is": "Labels one board request and flags whether it raises parking or curb space "
        "as a need or a problem, with a one-sentence reason.",
        "dataset": "gpt-5.4-nano deployment with the frozen prompt v2 (prompts/classify_v2.md)",
        "as_of": "Prompt v2 frozen Nov 5, 2026",
        "geography": "One request at a time; no geography.",
        "known_limits": [
            "On the 35 dev requests, prompt v2 got 34 and then 35 labels right in two runs, "
            "and 35 of 35 parking flags. 35 requests is a small test.",
            "A model's label, not a person's.",
            "Tested only on requests that came with the board's explanation; text pasted without "
            "its request category hasn't been tested.",
        ],
        "tool_status": "Built: labels one request at a time with the pipeline's model and frozen prompt v2.",
    },
]

NOT_COVERED = [
    "Specific sites, blocks or curbs, grid capacity, or property ownership: there's no site, "
    "feeder or ownership data.",
    "Council districts, neighborhoods (NTAs) or coordinates from the requests file.",
    "Neighborhood names: there's no neighborhood-to-district table, so ask for the district number.",
    "How many residents would switch to an EV if chargers were nearby: no source measures it.",
]

DISTRICT_CODES = (
    "Three digits: borough digit, then district number. 207 is Bronx Community District 7. "
    "Borough digits: 1 Manhattan, 2 Bronx, 3 Brooklyn, 4 Queens, 5 Staten Island. These are "
    "the city's standard borough codes; the boundary file 5crt-au7u holds codes only, no "
    "borough names, so they weren't confirmed from that file."
)


def describe_data() -> dict:
    """Every source's as-of date, geography, known limits and tool status. Written by hand,
    except the two tool lists, which come from TOOL_FUNCTIONS so they can't go stale."""
    return {
        "source": "Hand-written in agent_tools.py from the project's handoff notes and agent spec",
        "as_of": WRITTEN_ON,
        "tools_built_now": sorted(TOOL_FUNCTIONS),
        "tools_not_built": [s["tool"] for s in SOURCES if s["tool"] not in TOOL_FUNCTIONS],
        "district_codes": DISTRICT_CODES,
        "sources": SOURCES,
        "not_covered": NOT_COVERED,
    }


CLASSIFIER_PROMPT = "v2"  # frozen Nov 5, 2026; classify.load_prompt refuses an edited copy


def classify_request(explanation: str, request_category: str) -> dict:
    """Label one board request exactly as the labeling pipeline would.

    It reuses classify.py's own functions: same deployment, frozen prompt, settings and
    cache, and a brake of one model call (none if the text is already cached).
    """
    prompt = classify.load_prompt(CLASSIFIER_PROMPT)
    # Fix 23: a user's sentence often ends the category with a period, and the agent sometimes
    # keeps it. None of the 35 dev-set categories ends with one, so drop it: the classifier then
    # gets the same input, and the same cache entry, as the pipeline.
    category = request_category.strip().removesuffix(".")
    # The pipeline's input is "Request: <category>\nExplanation: <board's words>".
    text = classify.model_input(category, explanation)
    conn = classify.open_cache()
    try:
        results, stats = classify.classify_texts(classify.make_client(), conn, prompt, [text], max_calls=1)
    finally:
        conn.close()
    print(f"  classifier: {stats['calls']} call, {stats['cache_hits']} cache hit, "
          f"{stats['input_tokens']} input + {stats['output_tokens']} output tokens, "
          f"about ${classify.cost(stats):.5f}")
    result = results[text]
    return {
        "source": f"{classify.DEPLOYMENT} (model version {classify.MODEL_VERSION}) with frozen prompt "
        f"{CLASSIFIER_PROMPT}: the same setup as the labeling pipeline",
        "as_of": "Prompt v2 frozen Nov 5, 2026",
        "label": result["label"],
        "parking_concern": result["parking_concern"],
        "reason": result["reason"],
        "from_cache": result["from_cache"],
        "error": result["error"],
        "dev_set_check": "On the 35 hand-labeled dev requests: 34 and then 35 labels right in two runs; "
        "35 of 35 parking flags",
    }


classify_request_tool = FunctionTool(
    name="classify_request",
    description="Label one community board budget request the way the labeling pipeline would: "
    "ev_charging, parking_or_curb, clean_air or other, plus whether it raises parking or curb space "
    "as a need or a problem (parking_concern), with the model's reason.",
    parameters={
        "type": "object",
        "properties": {
            "explanation": {
                "type": "string",
                "description": "The board's own words, exactly as the user gave them",
            },
            "request_category": {
                "type": "string",
                "description": "The request's category line from the city's list, if the user gave "
                "one, such as 'Other transportation infrastructure requests'; otherwise an empty string",
            },
        },
        "required": ["explanation", "request_category"],
        "additionalProperties": False,
    },
    strict=True,
)


describe_data_tool = FunctionTool(
    name="describe_data",
    description="List every data source: what it is, its dataset, its as-of date, its geography, "
    "its known limits, and whether its tool is built yet. Also lists what no source covers. "
    "Takes no arguments.",
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
    strict=True,
)

# What create_agent.py sends to Foundry, and what ask_agent.py runs. Keep the two in step.
TOOL_DEFINITIONS = [describe_data_tool, classify_request_tool]
TOOL_FUNCTIONS = {"describe_data": describe_data, "classify_request": classify_request}
