"""A detection must be findable by the name it reports, including multi-word labels."""

from bruhos.body.eliza_bridge import to_detection
from bruhos.schemas import Detection, Observation


def test_multiword_label_matches_its_own_name():
    d = to_detection({"label": "red soccer ball", "x": 0.5, "y": 0.0})
    assert (d.label, d.color, d.name) == ("soccer_ball", "red", "red_soccer_ball")
    assert d.matches(d.name)
    assert d.matches("soccer_ball")
    assert d.matches("red_ball")
    assert d.matches("ball")


def test_multiword_label_without_color():
    d = to_detection({"label": "trash can", "x": 1.0, "y": 0.2})
    assert d.name == "trash_can"
    assert d.matches("trash_can")
    assert d.matches("trash-can")


def test_find_returns_multiword_object_by_its_name():
    obs = Observation(t=0.0, frame=0, objects=[
        to_detection({"label": "blue block", "x": 0.3, "y": 0.0}),
        to_detection({"label": "red soccer ball", "x": 0.8, "y": 0.1}),
    ])
    for o in obs.to_dict()["objects"]:
        hit = obs.find(o["name"])
        assert hit is not None and hit.name == o["name"]


def test_still_rejects_wrong_color_and_unrelated_words():
    d = Detection("soccer_ball", "red")
    assert not d.matches("blue_soccer_ball")
    assert not d.matches("red_block")
    assert not d.matches("basketball")
    assert not d.matches("")
    assert not d.matches("the")


def test_walk_toward_reaches_multiword_object_in_sim():
    from bruhos.schemas import SkillCall, Status
    from bruhos.sim.kinematic import SimBody, World, WorldObject
    from bruhos.skills import SkillContext, run_skill

    body = SimBody(World(objects=[WorldObject("soccer_ball", "red", 0.6, 0.1)]))
    name = body.observe().objects[0].name
    assert name == "red_soccer_ball"
    out = run_skill(SkillContext(body), SkillCall("walk_toward", {"target": name}))
    assert out.status is Status.OK, out.detail
