"""Closing the hand must grab; opening it already releases.

Failing-test-first for a reproduced defect: the rules planner maps
"Open your hand." to [release] but "Close your hand." -- the exact
opposite command -- plans nothing. The grab skill's own docstring is
"Close the hand on the object in reach", so the mapping is the skill's
documented meaning, not a new capability: [grab] closes the hand on
whatever is in reach and fails honestly ("nothing in reach" / "hand
full") when there is nothing to close on, exactly as [release] fails
honestly when the hand is empty. The same holds for the "the hand" /
bare "hand" spellings, which "Open the hand." / "Open hand." share.
"""

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.schemas import Observation, SkillCall
from bruhos.sim.kinematic import SimBody, World, WorldObject


def _plan(text):
    return RulesPlanner().plan(text, Observation(t=0.0, frame=0), [])


def _run(text, world=None):
    body = SimBody(world or World.tabletop())
    return body, Runtime(body, RulesPlanner()).run(text)


def test_close_hand_plans_grab():
    for text in ("Close your hand.", "Close the hand.", "Close hand."):
        assert _plan(text) == [SkillCall("grab", {})], text


def test_close_hand_on_a_named_object_plans_grab():
    # A hand command acts on what is already in reach: it closes the hand,
    # it does not navigate. This mirrors "Open your hand." -> [release],
    # which likewise releases without walking anywhere first.
    assert _plan("Close your hand on the red block.") == [SkillCall("grab", {})]


def test_open_hand_variants_plan_release():
    for text in ("Open your hand.", "Open the hand.", "Open hand."):
        assert _plan(text) == [SkillCall("release", {})], text


def test_close_hand_grabs_in_reach_end_to_end():
    world = World(objects=[WorldObject("block", "red", 0.08, 0.0),
                           WorldObject("box", None, 0.9, 0.0)])
    body, ep = _run("Close your hand.", world)
    assert ep.success, ep.end_reason
    assert [c["skill"] for c in ep.plan] == ["grab"]
    assert body.world.holding is not None and body.world.holding.name == "red_block"


def test_close_hand_with_nothing_in_reach_fails_honestly():
    body, ep = _run("Close your hand.")
    assert not ep.success
    assert "nothing in reach" in ep.end_reason
    assert body.world.holding is None


def test_unaffected_instructions_keep_their_plans():
    assert _plan("Open your hand.") == [SkillCall("release", {})]
    assert _plan("Pick up the red block.") == [
        SkillCall("walk_toward", {"target": "red_block"}), SkillCall("grab", {})]
    assert _plan("Grab the red block.") == [
        SkillCall("walk_toward", {"target": "red_block"}), SkillCall("grab", {})]
    assert _plan("Drop the red block.") == [SkillCall("release", {})]
