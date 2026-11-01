"""End-of-line roundoff must not add a gait cycle or consume the skill budget."""

import math
from fractions import Fraction

import pytest

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime, RuntimeConfig
from bruhos.schemas import SkillCall, Status
from bruhos.sim.kinematic import FORWARD_M, SimBody, World
from bruhos.skills import SkillContext, run_skill


@pytest.mark.parametrize("length, cycles", [(0.30, 6), (0.35, 7), (0.40, 8)])
def test_exact_length_line_finishes_within_its_gait_budget(length, cycles):
    body = SimBody(World.tabletop(line_m=length))
    outcome = run_skill(SkillContext(body, max_steps=cycles),
                        SkillCall("follow_line", {"steps": cycles + 1}))

    assert outcome.status is Status.OK
    assert outcome.detail == "end of line"
    assert outcome.data["steps"] == cycles
    assert body.n_locomotion == cycles
    assert body.world.line_m == 0.0
    assert not body.observe().line.visible


def test_runtime_completes_a_six_cycle_line_with_a_six_cycle_budget():
    body = SimBody(World.tabletop(line_m=0.30))
    runtime = Runtime(body, RulesPlanner(), config=RuntimeConfig(max_skill_steps=6, max_replans=0))

    episode = runtime.run("Follow the line.")

    assert episode.success, episode.end_reason
    assert body.n_locomotion == 6
    assert episode.steps[0].outcome["detail"] == "end of line"


def test_no_fall_is_injected_after_the_line_should_have_ended():
    body = SimBody(World.tabletop(line_m=0.30), falls_at={7})
    runtime = Runtime(body, RulesPlanner(), config=RuntimeConfig(max_recoveries=0))

    episode = runtime.run("Follow the line.")

    assert episode.success, episode.end_reason
    assert episode.falls == 0
    assert body.n_locomotion == 6


@pytest.mark.parametrize("length, cycles", [(0.301, 7), (0.351, 8), (0.401, 9), (math.nextafter(0.30, math.inf), 7), (1e-13, 1)])
def test_real_positive_line_remainders_still_require_a_step(length, cycles):
    body = SimBody(World.tabletop(line_m=length))
    outcome = run_skill(SkillContext(body, max_steps=cycles),
                        SkillCall("follow_line", {"steps": cycles + 1}))

    assert outcome.status is Status.OK
    assert outcome.data["steps"] == cycles
    assert body.n_locomotion == cycles


def test_a_fall_during_the_actual_line_is_still_reported():
    body = SimBody(World.tabletop(line_m=0.30), falls_at={6})
    runtime = Runtime(body, RulesPlanner(), config=RuntimeConfig(max_recoveries=0))

    episode = runtime.run("Follow the line.")

    assert not episode.success
    assert episode.falls == 1
    assert body.n_locomotion == 6


def test_gait_counts_match_exact_arithmetic_at_line_boundaries():
    for cycles in range(1, 201):
        boundary = cycles / 20
        for length in (math.nextafter(boundary, 0.0), boundary,
                       math.nextafter(boundary, math.inf)):
            expected = math.ceil(Fraction(length) / Fraction(FORWARD_M))
            body = SimBody(World(line_m=length))
            actual = 0
            while body.world.line_m > 0 and actual <= expected:
                body.act("forward")
                actual += 1
            assert actual == expected, (length, actual, expected)
