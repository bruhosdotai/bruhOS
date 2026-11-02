"""The OpenAI-compatible adapter must say what the brain actually returned.

`_complete()` read the reply as

    content = json.loads(raw)["choices"][0]["message"]["content"]

so a 200 whose body is not an OpenAI envelope escapes as `KeyError('choices')`
(or an `IndexError` for an empty `choices` list). `Runtime._plan` catches broad
`Exception` last, logs, and ends the episode with `end_reason = "brain error:
'choices'"` — the operator gets a repr of a dict key instead of "the gateway
answered 200 with something that is not a completion". That episode also seals
with zero steps, so the trace shows nothing about what was asked.

A `content: null` reply (tool-call-only, or a proxy that strips text) is worse:
it reaches `parse_plan(None)` and is reported as `Status.REJECTED`, which in this
repo means "the brain proposed a skill that may not reach the body" — a policy
violation, not a transport failure. Wrong diagnosis for whoever reads the trace.
"""

from __future__ import annotations

import json

import pytest

from bruhos.brain.adapters.openai import OpenAICompatPlanner
from bruhos.brain.base import IntegrationError
from bruhos.skills.registry import CATALOG


def _planner(body) -> OpenAICompatPlanner:
    p = OpenAICompatPlanner(base_url="http://127.0.0.1:1/v1", model="m", api_key="")
    p.transport = lambda url, headers, data, timeout: json.dumps(body).encode()
    return p


def _obs():
    from bruhos.schemas import Imu, LineReading, Observation

    return Observation(t=0.0, frame=0, objects={}, imu=Imu(fallen=False),
                       line=LineReading(visible=False, offset=0.0), holding=None)


@pytest.mark.parametrize("body,needle", [
    ({"detail": "upstream timed out"}, "choices"),
    ({"choices": []}, "choices"),
    ({"choices": [{"message": {}}]}, "content"),
])
def test_non_completion_body_raises_integration_error(body, needle):
    with pytest.raises(IntegrationError) as e:
        _planner(body).plan("Wave hello.", _obs(), CATALOG)
    assert needle in str(e.value).lower()


def test_gateway_error_is_not_reported_as_a_policy_violation():
    """A bad body must not surface as 'plan rejected' (Status.REJECTED)."""
    from bruhos.brain.adapters.rules import RulesPlanner
    from bruhos.runtime import Runtime
    from bruhos.sim.kinematic import SimBody

    ep = Runtime(SimBody(), _planner({"detail": "upstream timed out"})).run("Wave hello.")
    assert ep.success is False
    assert "upstream" in ep.end_reason and "choices" in ep.end_reason.lower(), ep.end_reason
    assert "KeyError" not in ep.end_reason


def test_well_formed_completion_still_parses():
    body = {"choices": [{"message": {"content": json.dumps({"plan": [{"skill": "stop", "args": {}}]})}}]}
    plan = _planner(body).plan("Stop.", _obs(), CATALOG)
    assert [c.skill for c in plan] == ["stop"]
