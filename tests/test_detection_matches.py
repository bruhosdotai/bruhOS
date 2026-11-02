from bruhos.schemas import Detection, Observation
from bruhos.sim.kinematic import SimBody
from bruhos.skills import SkillContext, run_skill
from bruhos.schemas import SkillCall, Status


def test_detection_matches_space_and_hyphen_delimiters():
    d = Detection(label="block", color="red")
    # Exact identifiers
    assert d.matches("red_block")
    assert d.matches("red")
    assert d.matches("block")

    # Space-delimited and hyphen-delimited targets
    assert d.matches("red block")
    assert d.matches("red-block")
    assert d.matches("red   block")

    # Target phrases with English stop words "the" and "one"
    assert d.matches("the red block")
    assert d.matches("the red one")
    assert d.matches("the_red_one")

    # Unrelated targets must not match
    assert not d.matches("blue block")
    assert not d.matches("green")
    assert not d.matches("")
    assert not d.matches("the one")


def test_observation_find_with_space_delimited_target():
    obs = Observation(
        t=0.0,
        frame=1,
        objects=[
            Detection(label="block", color="red", bearing_deg=0.0, distance_m=0.5),
            Detection(label="block", color="blue", bearing_deg=10.0, distance_m=0.8),
            Detection(label="box", color=None, bearing_deg=-5.0, distance_m=1.2),
        ],
    )
    assert obs.find("red block") is not None
    assert obs.find("red block").color == "red"
    assert obs.find("the red one") is not None
    assert obs.find("the red one").color == "red"
    assert obs.find("the box") is not None
    assert obs.find("the box").label == "box"


def test_skill_execution_with_natural_language_target():
    body = SimBody()
    ctx = SkillContext(body=body)
    # detect skill with space-separated target name
    outcome = run_skill(ctx, SkillCall("detect", {"target": "red block"}))
    assert outcome.status is Status.OK
    assert outcome.data.get("found") == "red_block"
