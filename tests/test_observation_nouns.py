"""Object nouns the rules planner already accepts must resolve on the body too.

The published sim eval misses "Place the green cube into the box." when a brain
keeps the word cube: the kinematic object is green_block, and find() used to
require that exact token. cube/brick and bin/basket are the same aliases as
bruhos.brain.adapters.rules.NOUNS.
"""

import pytest

from bruhos.brain.adapters.rules import NOUNS
from bruhos.brain.base import Planner
from bruhos.runtime import Runtime
from bruhos.schemas import Detection, Observation, SkillCall
from bruhos.sim.kinematic import SimBody


def _scene() -> Observation:
    return Observation(
        t=0,
        frame=0,
        objects=[
            Detection("block", "red", bearing_deg=5.0, distance_m=0.3),
            Detection("block", "green", bearing_deg=-12.0, distance_m=0.8),
            Detection("box", None, bearing_deg=0.0, distance_m=1.0),
            Detection("ball", "yellow", bearing_deg=20.0, distance_m=0.5),
        ],
    )


@pytest.mark.parametrize(
    ("surface", "canonical"),
    [(surface, canonical) for surface, canonical in sorted(NOUNS.items()) if surface != canonical],
)
def test_find_resolves_every_rules_noun_alias(surface, canonical):
    found = _scene().find(f"green_{surface}" if canonical == "block" else surface)
    assert found is not None
    assert found.name == ("green_block" if canonical == "block" else canonical)


def test_green_cube_does_not_select_the_closer_red_block():
    scene = _scene()
    for target in ("green_cube", "green cube", "the green cube"):
        found = scene.find(target)
        assert found is not None and found.name == "green_block"


def test_unaliased_target_still_misses():
    assert _scene().find("green_pyramid") is None


class _GreenCubePlanner(Planner):
    name = "green-cube"
    version = "test"

    def plan(self, instruction, obs, catalog):
        return [
            SkillCall("walk_toward", {"target": "green_cube"}),
            SkillCall("grab"),
        ]


def test_walk_and_grab_green_cube_holds_the_green_block():
    body = SimBody()
    ep = Runtime(body, _GreenCubePlanner()).run("Place the green cube into the box.")
    assert ep.success, ep.end_reason
    assert body.world.holding is not None
    assert body.world.holding.name == "green_block"
