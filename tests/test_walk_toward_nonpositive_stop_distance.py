"""walk_toward must fail a non-positive stop distance instead of walking forever.

``walk_toward(stop_distance_m=0)`` (or a negative distance) can never satisfy
its stop condition: a detection's distance is a hypotenuse, always >= 0, and
the robot steps in 0.05 m increments, so ``distance <= stop_at`` never becomes
true. The skill scans, walks, loses the target, rescans and repeats until the
whole primitive budget is spent, then reports a misleading "walk timeout" —
hundreds of locomotion steps for an argument that was unsatisfiable on entry.
A stop distance that cannot be reached must fail at once, before any motion.
"""

import pytest

from bruhos.schemas import SkillCall, Status
from bruhos.sim.kinematic import SimBody
from bruhos.skills import SkillContext, run_skill


@pytest.mark.parametrize("stop_at", [0.0, -0.5])
def test_walk_toward_nonpositive_stop_distance_fails_fast(stop_at):
    body = SimBody()
    out = run_skill(SkillContext(body, max_steps=120),
                    SkillCall("walk_toward", {"target": "red_block", "stop_distance_m": stop_at}))
    assert out.status is Status.FAILED
    assert "stop_distance_m" in out.detail
    assert out.data["stop_distance_m"] == stop_at
    assert body.n_locomotion == 0 and body.t == 0.0  # rejected before any motion


def test_walk_toward_positive_stop_distance_still_reaches():
    body = SimBody()
    out = run_skill(SkillContext(body, max_steps=120),
                    SkillCall("walk_toward", {"target": "red_block", "stop_distance_m": 0.1}))
    assert out.status is Status.OK
