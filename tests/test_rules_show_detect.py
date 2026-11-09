"""The rules planner must answer "show me" with a detect.

"Show me the red block." is a vision query, synonymous with "Find the red
block." -- but the planner only routed find/where/locate/search/detect/see
into the detect branch, so show returned no plan and the episode ended "no
plan for instruction" even though detect can do exactly that: scan until the
object is in view and report it.
"""

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.schemas import Observation, SkillCall
from bruhos.sim.kinematic import SimBody, World


def _plan(text):
    return RulesPlanner().plan(text, Observation(t=0.0, frame=0), [])


def _run(text):
    body = SimBody(World.tabletop())
    return body, Runtime(body, RulesPlanner()).run(text)


def test_show_me_plans_detect_with_target():
    assert _plan("Show me the red block.") == [SkillCall("detect", {"target": "red_block"})]
    assert _plan("Show me the box.") == [SkillCall("detect", {"target": "box"})]
    assert _plan("Show me the ball.") == [SkillCall("detect", {"target": "ball"})]


def test_show_me_end_to_end_succeeds():
    for text in ("Show me the red block.", "Show me the box."):
        _body, ep = _run(text)
        assert ep.success, text
        assert ep.end_reason == "completed", text


def test_find_and_scan_unaffected():
    assert _plan("Find the green block.") == [SkillCall("detect", {"target": "green_block"})]
    assert _plan("What do you see?") == [SkillCall("detect", {})]
