"""Release-into-box must plan a delivery, not an in-place release.

"Drop the red block in the box" already plans detect + walk + grab +
walk-to-box + release, but "Release the red block in the box" fell through
to the plain drop branch and returned a bare [release]: executed wherever
the robot stood, reported success, and left the object on the floor instead
of in the box. "release" (and the "let go" phrase) were the only drop-branch
verbs missing from the into_box gate.
"""

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.schemas import Observation
from bruhos.sim.kinematic import SimBody
from bruhos.skills import CATALOG

DELIVERY = [
    ("detect", {"target": "red_block"}),
    ("walk_toward", {"target": "red_block"}),
    ("grab", {}),
    ("walk_toward", {"target": "box"}),
    ("release", {}),
]


def _plan(instruction, holding=None):
    obs = Observation(t=0.0, frame=0, holding=holding)
    return [(c.skill, c.args) for c in RulesPlanner().plan(instruction, obs, CATALOG)]


def test_release_into_box_plans_delivery():
    assert _plan("Release the red block in the box.") == DELIVERY


def test_release_into_the_box_plans_delivery():
    assert _plan("Release the red block into the box.") == DELIVERY


def test_release_to_box_plans_delivery():
    assert _plan("Release the red block to the box.") == DELIVERY


def test_let_go_into_box_plans_delivery():
    assert _plan("Let go of the red block in the box.") == DELIVERY


def test_release_into_box_holding_plans_delivery_not_in_place_release():
    assert _plan("Release the red block in the box.", holding="red_block") == DELIVERY


def test_bare_release_unchanged_when_holding():
    assert _plan("Release the red block.", holding="red_block") == [("release", {})]


def test_bare_release_unchanged_empty_hand():
    assert _plan("Release the red block.") == [("release", {})]


def test_drop_into_box_still_delivery():
    assert _plan("Drop the red block in the box.") == DELIVERY


def test_put_into_box_still_delivery():
    assert _plan("Put the red block in the box.") == DELIVERY


def test_runtime_release_into_box_delivers():
    body = SimBody()
    ep = Runtime(body, RulesPlanner()).run("Release the red block in the box.")
    assert ep.success is True
    assert body.world.get("red_block").in_box is True
    assert "in_box:red_block" in body.world.events
