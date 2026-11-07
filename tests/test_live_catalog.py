"""A skill registered after import must reach the brain, not just the executor.

``Runtime`` handed the brain the import-time ``CATALOG`` snapshot while
``validate`` / ``run_skill`` / ``catalog_hash`` all read the live registry. A
skill registered after ``bruhos.skills`` was imported (the public ``@skill``
decorator is the extension point) was therefore executable -- a plan naming
it validated and ran -- but invisible to every brain, so no brain could ever
plan it. The sealed ``skillpack_hash`` already included the late skill while
the catalog the plan was built from did not.
"""

from bruhos.brain.base import Planner
from bruhos.runtime import Runtime
from bruhos.schemas import Outcome, SkillCall, Status
from bruhos.sim.kinematic import SimBody
from bruhos.skills import registry as reg


class CatalogDrivenPlanner(Planner):
    """Plans the late skill, but only if the catalog it was given names it."""

    name = "catalog-driven"

    def __init__(self):
        self.seen: list[str] = []

    def plan(self, instruction, obs, catalog):
        self.seen = [c["skill"] for c in catalog]
        if "zz_custom" in self.seen:
            return [SkillCall("zz_custom")]
        return [SkillCall("wave")]


def test_late_registered_skill_reaches_the_brain():
    @reg.skill(reg.SkillSpec("zz_custom", "test-only late skill"))
    def _impl(ctx, args):
        return Outcome(Status.OK, "custom ran")

    brain = CatalogDrivenPlanner()
    try:
        ep = Runtime(SimBody(), brain).run("Do the custom thing.")
    finally:
        reg.SKILLS.pop("zz_custom", None)
        reg._IMPL.pop("zz_custom", None)
    assert "zz_custom" in brain.seen
    assert ep.success, ep.end_reason
    assert [c["skill"] for c in ep.plan] == ["zz_custom"]
