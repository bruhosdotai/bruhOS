import pytest

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.eval import run_eval
from bruhos.sim.kinematic import SimBody


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan"), float("inf"), float("-inf")])
def test_invalid_fall_probability_is_rejected(probability):
    with pytest.raises(ValueError, match="fall_prob"):
        SimBody(fall_prob=probability)


@pytest.mark.parametrize("probability,fallen", [(0.0, False), (1.0, True)])
def test_probability_endpoints_keep_their_simulator_meaning(probability, fallen):
    body = SimBody(fall_prob=probability)
    assert body.act("forward")
    assert body.observe().imu.fallen is fallen


def test_eval_does_not_report_a_perfect_completion_rate_for_nan_probability():
    with pytest.raises(ValueError, match="fall_prob"):
        run_eval(RulesPlanner, fall_prob=float("nan"))
