"""Detection target matching must resolve whitespace-separated tokens and stop words.

A planner or user frequently specifies targets with spaces (e.g. "red block",
"the red block", "the box", "red one"). The tokenizer previously split only on
underscores after hyphen replacement, leaving space-separated tokens intact so
"red block" became {"red block"}, which failed to match {"red", "block"} and
caused skills to spin 360 degrees and time out. Stop word removal ("the", "one")
was also bypassed because multi-word phrases were not tokenized on whitespace.
"""

from bruhos.schemas import Detection, Observation, SkillCall, Status
from bruhos.sim.kinematic import SimBody, World, WorldObject
from bruhos.skills import SkillContext, run_skill


def test_matches_whitespace_separated_target():
    d = Detection(label="block", color="red")
    assert d.matches("red block")
    assert d.matches("the red block")
    assert d.matches("the red one")
    assert d.matches("red one")


def test_matches_single_word_and_stopword_destination():
    d = Detection(label="box")
    assert d.matches("the box")
    assert d.matches("box")
    assert d.matches("the_box")


def test_matches_still_supports_underscores_and_hyphens():
    d = Detection(label="block", color="red")
    assert d.matches("red_block")
    assert d.matches("red-block")
    assert d.matches("the_red_block")


def test_matches_rejects_mismatch_and_stopword_only():
    d = Detection(label="block", color="red")
    assert not d.matches("blue block")
    assert not d.matches("green")
    assert not d.matches("the")
    assert not d.matches("one")
    assert not d.matches("")
    assert not d.matches("   ")


def test_compound_label_matches_spaced_and_underscored():
    d = Detection(label="soccer_ball", color="red")
    assert d.matches("soccer ball")
    assert d.matches("red soccer ball")
    assert d.matches("the red soccer ball")
    assert d.matches("soccer_ball")
    assert d.matches("red_soccer_ball")
    assert d.matches("ball")
    assert not d.matches("blue soccer ball")


def test_detect_skill_resolves_target_with_whitespace_in_sim():
    body = SimBody(World(objects=[WorldObject("block", "red", 0.8, 0.0)]))
    ctx = SkillContext(body)
    out = run_skill(ctx, SkillCall("detect", {"target": "red block"}))
    assert out.status is Status.OK, out.detail
    assert out.data["found"] == "red_block"


def test_walk_toward_skill_resolves_target_with_whitespace_in_sim():
    body = SimBody(World(objects=[WorldObject("block", "red", 0.5, 0.0)]))
    ctx = SkillContext(body)
    out = run_skill(ctx, SkillCall("walk_toward", {"target": "the red block"}))
    assert out.status is Status.OK, out.detail
