"""A malformed chat-completions response must be a rejected plan, not a brain error.

Failing-test-first for a reproduced defect: on base, ``OpenAICompatPlanner._complete``
indexed straight into ``json.loads(raw)["choices"][0]["message"]["content"]``, so any
response without that exact shape raised ``KeyError`` / ``IndexError`` / ``TypeError``.
The runtime only records a rejected step for ``SkillError`` / ``IntegrationError`` /
``ValueError``; everything else becomes a bare ``"brain error: ..."`` end reason with no
step in the trace at all. A model gateway answering HTTP 200 with an error payload, an
empty choices list, or a choice without content is exactly a response that did not
contain a usable plan, so it belongs in the rejected bucket, with the step recorded.
"""

import json

import pytest

from bruhos.brain.adapters.openai import OpenAICompatPlanner
from bruhos.runtime import Runtime
from bruhos.sim.kinematic import SimBody


def _planner(raw: bytes) -> OpenAICompatPlanner:
    return OpenAICompatPlanner(
        base_url="https://x/v1", api_key="k",
        transport=lambda url, headers, body, timeout: raw,
    )


def _run(raw: bytes):
    body = SimBody()
    ep = Runtime(body, _planner(raw)).run("Wave hello.")
    return ep, body


@pytest.mark.parametrize("raw", [
    b'{"error": {"message": "rate limited"}}',
    b'{"choices": []}',
    b'{"choices": [{"message": {}}]}',
    b'[{"skill": "wave"}]',
])
def test_malformed_response_raises_value_error(raw):
    with pytest.raises(ValueError, match="no chat choice content"):
        _planner(raw).plan("Wave hello.", SimBody().observe(), [])


@pytest.mark.parametrize("raw", [
    b'{"error": {"message": "rate limited"}}',
    b'{"choices": []}',
    b'{"choices": [{"message": {}}]}',
    b'[{"skill": "wave"}]',
])
def test_malformed_response_is_a_rejected_step_not_a_brain_error(raw):
    ep, body = _run(raw)
    assert not ep.success
    assert ep.end_reason.startswith("plan rejected: ")
    assert len(ep.steps) == 1 and ep.steps[0].kind == "rejected"
    assert ep.steps[0].outcome["status"] == "rejected"
    assert body.world.events == []


def test_null_content_was_already_a_rejected_step():
    # Control: this shape already raised ValueError through parse_plan on base;
    # the fix must not regress it.
    ep, body = _run(b'{"choices": [{"message": {"content": null}}]}')
    assert not ep.success
    assert ep.end_reason.startswith("plan rejected: ")
    assert len(ep.steps) == 1 and ep.steps[0].kind == "rejected"
    assert body.world.events == []


def test_well_formed_response_still_plans():
    raw = json.dumps({"choices": [{"message": {"content": '{"plan": [{"skill": "wave", "args": {}}]}'}}]}).encode()
    ep, body = _run(raw)
    assert ep.success
    assert "wave" in body.world.events
