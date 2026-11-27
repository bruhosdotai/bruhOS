"""Detection/Observation target resolution (`red_soccer_ball`, `trash_can`, poetry)."""

from bruhos.body.eliza_bridge import to_detection
from bruhos.schemas import Detection, Observation


def _obs(*detections: Detection) -> Observation:
    return Observation(t=0.0, frame=0, objects=list(detections))


def test_multi_word_label_matches_own_reported_name():
    det = to_detection({"label": "red soccer ball", "x": 1.0, "y": 0.0})
    assert det.name == "red_soccer_ball"
    assert det.matches(det.name)
    assert _obs(det).find(det.name) is det


def test_multi_word_label_matches_color_label_and_partial_targets():
    det = to_detection({"label": "red soccer ball", "x": 1.0, "y": 0.0})
    for target in ("red_soccer_ball", "soccer_ball", "soccer", "ball", "red"):
        assert det.matches(target), target


def test_multi_word_label_without_color_matches_itself():
    det = to_detection({"label": "trash can", "x": 1.0, "y": 0.0})
    assert det.name == "trash_can"
    for target in ("trash_can", "trash", "can"):
        assert det.matches(target), target


def test_wrong_color_and_unrelated_words_are_rejected():
    det = to_detection({"label": "red soccer ball", "x": 1.0, "y": 0.0})
    for target in ("blue_soccer_ball", "blue", "box", "red_box", "soccer_ball_blue"):
        assert not det.matches(target), target


def test_separators_and_case_are_normalised():
    det = to_detection({"label": "red soccer ball", "x": 1.0, "y": 0.0})
    for target in ("Red Soccer Ball", "red-soccer-ball", "RED_SOCCER_BALL", "  red_soccer_ball  "):
        assert det.matches(target), target


def test_filler_tokens_still_ignored():
    det = Detection(label="block", color="red")
    assert det.matches("the red one")
    assert not det.matches("the one")      # nothing but filler resolves to no target


def test_empty_and_separator_only_targets_never_match():
    det = Detection(label="block", color="red")
    for target in ("", "   ", "_", "__", "-", "the one"):
        assert not det.matches(target), repr(target)


def test_single_word_behaviour_is_unchanged():
    det = Detection(label="block", color="red")
    assert det.matches("red_block") and det.matches("red") and det.matches("block")
    assert not det.matches("blue_block") and not det.matches("box")


def test_find_prefers_nearest_then_most_head_on():
    near_off = Detection(label="soccer_ball", color="red", distance_m=1.0, bearing_deg=40.0)
    far_on = Detection(label="soccer_ball", color="red", distance_m=3.0, bearing_deg=0.0)
    near_on = Detection(label="soccer_ball", color="green", distance_m=1.0, bearing_deg=5.0)
    obs = _obs(far_on, near_off, near_on)
    assert obs.find("red_soccer_ball") is near_off
    assert obs.find("soccer_ball") is near_on
    assert obs.find("blue_soccer_ball") is None
