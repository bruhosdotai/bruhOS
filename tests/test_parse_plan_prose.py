"""parse_plan must recover the plan JSON when a model wraps it in prose.

Chat models routinely emit a sentence around the plan ("Here is the plan:
{...}"). The fence stripper only handled ```json blocks, so any surrounding
prose made json.loads fail and the *valid* plan was rejected as a brain error.
The first balanced JSON object in the text must be recovered instead; garbage
with no parseable object is still rejected.
"""

import pytest

from bruhos.brain.base import IntegrationError, parse_plan


PLAN = '{"plan": [{"skill": "wave", "args": {}}]}'


@pytest.mark.parametrize("text", [
    f"Here is the plan:\n{PLAN}",
    f"{PLAN}\nHope this helps!",
    f"Sure! ```json\n{PLAN}\n```",
    f"I will wave. {PLAN} Done.",
    f"```json\n{PLAN}\n```",
    "The plan: " + '{"plan": [{"skill": "walk_toward", "args": {"target": "red_block"}}, {"skill": "grab"}]}' + ".",
    ("note " + PLAN).encode(),
])
def test_prose_wrapping_is_recovered(text):
    calls = parse_plan(text)
    assert calls


def test_plain_plan_still_parses():
    assert parse_plan(PLAN)[0].skill == "wave"


def test_garbage_without_object_is_rejected():
    with pytest.raises(ValueError):
        parse_plan("I have no plan for you.")
    with pytest.raises(ValueError):
        parse_plan("   ")


def test_truncated_object_is_rejected():
    with pytest.raises(ValueError):
        parse_plan(f"prefix {PLAN[:-1]}")


def test_first_valid_object_wins_after_decoy():
    # An earlier unparseable brace must not shadow the real plan.
    calls = parse_plan(f"bad {{ \"oops\": }} good {PLAN}")
    assert calls[0].skill == "wave"


def test_escaped_quote_in_string_does_not_break_recovery():
    text = 'x {"plan": [{"skill": "wave", "args": {"note": "a } b \\" c"}}]}'
    calls = parse_plan(text)
    assert calls[0].skill == "wave"


def test_continuous_chunk_guard_applies_to_recovered_object():
    # Recovery must not bypass the joint-action guard.
    with pytest.raises(IntegrationError):
        parse_plan('plan: {"plan": [[1, 2, 3]]}')


def test_pathological_unbalanced_response_fails_fast():
    # A response of 100k never-balancing '{' must not wedge the planner: the
    # recovery scan is O(n^2), so it is capped by _RECOVERY_MAX_BYTES and the
    # garbage is rejected with the original JSONDecodeError in milliseconds.
    # Without the cap this takes tens of seconds at 60KB and hours at 4MiB.
    import time
    from bruhos.brain.base import _RECOVERY_MAX_BYTES

    hostile = "{" * (_RECOVERY_MAX_BYTES * 2)  # 128KB of unbalanced braces
    t0 = time.monotonic()
    with pytest.raises(ValueError):
        parse_plan(hostile)
    assert time.monotonic() - t0 < 2.0, "recovery scan wedged on unbalanced braces"


def test_recovery_still_works_just_under_cap():
    # A real plan wrapped in prose well within the cap is still recovered.
    from bruhos.brain.base import _RECOVERY_MAX_BYTES

    pad = "x" * (_RECOVERY_MAX_BYTES - len(PLAN) - 32)
    calls = parse_plan(f"Here is the plan:\n{PLAN}\n{pad}")
    assert calls[0].skill == "wave"
