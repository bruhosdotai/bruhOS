import json
from bruhos.brain import parse_plan

PLAN_JSON = {"plan": [{"skill": "wave", "args": {}}]}


def test_indented_code_fence():
    raw = "   ```json\n" + json.dumps(PLAN_JSON) + "\n   ```\n"
    plans = parse_plan(raw)
    assert len(plans) == 1
    assert plans[0].skill == "wave"


def test_uppercase_json_tag():
    raw = "```JSON\n" + json.dumps(PLAN_JSON) + "\n```\n"
    plans = parse_plan(raw)
    assert len(plans) == 1
    assert plans[0].skill == "wave"


def test_trailing_whitespace_on_closing_fence():
    raw = "```json\n" + json.dumps(PLAN_JSON) + "\n```   \n"
    plans = parse_plan(raw)
    assert len(plans) == 1
    assert plans[0].skill == "wave"


def test_bare_code_fence_without_tag():
    raw = "```\n" + json.dumps(PLAN_JSON) + "\n```\n"
    plans = parse_plan(raw)
    assert len(plans) == 1
    assert plans[0].skill == "wave"


def test_code_fence_with_bytes_and_tab_indentation():
    raw = b"\t```json\n" + json.dumps(PLAN_JSON).encode() + b"\n\t```\n"
    plans = parse_plan(raw)
    assert len(plans) == 1
    assert plans[0].skill == "wave"
