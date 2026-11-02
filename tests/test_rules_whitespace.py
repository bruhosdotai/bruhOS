import pytest

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.sim.kinematic import SimBody


@pytest.mark.parametrize("instruction", ["Pick   up the red block.", "Pick\tup the red block.", "Pick\nup the red block."])
def test_multispace_pickup_matches_normal_command(instruction):
    body = SimBody()
    episode = Runtime(body, RulesPlanner()).run(instruction)
    assert episode.success, episode.end_reason
    assert body.world.holding is not None
    assert body.world.holding.name == "red_block"


def test_phrase_does_not_match_inside_another_word():
    body = SimBody()
    episode = Runtime(body, RulesPlanner()).run("Pickup the red block.")
    assert not episode.plan
