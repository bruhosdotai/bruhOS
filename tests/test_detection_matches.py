"""A detection must be findable by the multi-word name it reports."""

import pytest

from bruhos.body.eliza_bridge import to_detection
from bruhos.schemas import Detection, Observation


@pytest.mark.parametrize(
    "target",
    ["red_soccer_ball", "soccer_ball", "ball", "red", "the red soccer ball", "RED-SOCCER-BALL"],
)
def test_bridge_multi_word_name_is_findable(target):
    d = to_detection({"label": "red soccer ball", "x": 0.4, "y": 0.0})
    assert d.name == "red_soccer_ball"
    obs = Observation(t=0.0, frame=1, objects=[d])
    assert obs.find(target) is d


@pytest.mark.parametrize("target", ["trash_can", "trash", "can", "the trash can"])
def test_multi_word_label_without_a_colour_is_findable(target):
    d = Detection(label="trash_can")
    obs = Observation(t=0.0, frame=1, objects=[d])
    assert obs.find(target) is d


@pytest.mark.parametrize("target", ["blue_soccer_ball", "red_box", "car", "soccer_ball car", ""])
def test_unrelated_or_wrong_colour_targets_are_still_refused(target):
    d = to_detection({"label": "red soccer ball", "x": 0.4, "y": 0.0})
    assert d.matches(target) is False


def test_single_word_behaviour_is_unchanged():
    d = Detection(label="block", color="red")
    assert d.matches("red_block") is True
    assert d.matches("block") is True
    assert d.matches("red") is True
    assert d.matches("one red block") is True
    assert d.matches("box") is False
