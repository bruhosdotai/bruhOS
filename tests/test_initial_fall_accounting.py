"""An episode starting with a fallen robot must record the fall in its metrics.

When the robot is already on the floor at the start of an episode, the runtime
recovers before planning. The recovery step was counted in ``ep.recoveries``, but
``ep.falls`` remained 0, causing the sealed episode header, dashboard traces, and
CLI metrics to report an impossible state (0 falls, 1 recovery).
"""

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.eval import run_eval
from bruhos.runtime import Runtime
from bruhos.sim.kinematic import SimBody, World


def test_initial_fall_recorded_in_episode_metrics():
    body = SimBody(World(fallen=True, fell_forward=False))
    ep = Runtime(body, RulesPlanner()).run("Stand up.")
    assert ep.success
    assert ep.falls == 1
    assert ep.recoveries == 1
    assert ep.header()["falls"] == 1
    assert ep.header()["recoveries"] == 1


def test_initial_fall_in_eval_is_counted_natively():
    # Prompt 16 ("Stand up.") starts with world = { fallen = true, fell_forward = false }
    report = run_eval(RulesPlanner, fall_prob=0.0, seed=0)
    stand_up_results = [r for r in report.results if r.text == "Stand up."]
    assert len(stand_up_results) == 1
    assert stand_up_results[0].falls == 1
    assert stand_up_results[0].recovered == 1
