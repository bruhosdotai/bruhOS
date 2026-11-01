import pytest

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.sim.kinematic import SimBody


@pytest.mark.parametrize("instruction", ["Forget upcoming tasks.", "Look aroundabouts."])
def test_phrase_fragments_inside_words_do_not_become_commands(instruction):
    body = SimBody()
    episode = Runtime(body, RulesPlanner()).run(instruction)
    assert not episode.plan
    assert not episode.success
    assert body.world.events == []


def test_real_multiword_command_remains_supported():
    episode = Runtime(SimBody(), RulesPlanner()).run("Look around.")
    assert episode.success
    assert episode.plan[0]["skill"] == "detect"
