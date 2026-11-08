"""`bruhos skills` must report the whitelist its own skillpack_hash commits to.

``bruhos/skills/registry.py`` freezes ``CATALOG = catalog()`` at import time, while
``catalog_hash()`` reads the live registry. ``cmd_skills`` printed that frozen snapshot
beside the live hash, so a skill registered through the public ``@skill`` decorator --
the extension point ``validate``, ``run_skill`` and the sealed episode header all honour --
was missing from the whitelist the command reported while the hash printed next to it
already committed to it. One line of output described two different skillpacks and said
nothing about which half was stale.
"""

import contextlib
import hashlib
import io
import json

from bruhos import cli
from bruhos.brain.base import Planner
from bruhos.runtime import Runtime
from bruhos.schemas import Outcome, SkillCall, Status
from bruhos.sim.kinematic import SimBody
from bruhos.skills import catalog_hash
from bruhos.skills.registry import SKILLS, _IMPL, catalog, skill, SkillSpec

NAME = "zz_cli_late"


class LatePlanner(Planner):
    name = "late"

    def plan(self, instruction, obs, catalog):
        return [SkillCall(NAME)]


def _register():
    @skill(SkillSpec(NAME, "registered after bruhos.skills was imported"))
    def _impl(ctx, args):
        return Outcome(Status.OK, "late skill ran")


def _unregister():
    SKILLS.pop(NAME, None)
    _IMPL.pop(NAME, None)


def _report() -> dict:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert cli.main(["skills"]) == 0
    return json.loads(buf.getvalue())


def _hash_of(skills: list) -> str:
    """The same digest catalog_hash() computes, over the list the command printed."""
    blob = json.dumps(skills, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def test_skills_command_reports_one_consistent_skillpack():
    _register()
    try:
        report = _report()
        names = [s["skill"] for s in report["skills"]]
        assert NAME in names, (
            f"`bruhos skills` left {NAME!r} out of the whitelist even though the live "
            f"registry it seals has {len(catalog())} skills and the report lists "
            f"{len(names)}")
        assert _hash_of(report["skills"]) == report["skillpack_hash"], (
            "the two halves of the report describe different skillpacks: the printed "
            "whitelist hashes to " + _hash_of(report["skills"])[:19]
            + " but the printed skillpack_hash is " + report["skillpack_hash"][:19])
    finally:
        _unregister()


def test_skills_command_covers_a_skill_that_ran_in_the_sim():
    _register()
    try:
        ep = Runtime(SimBody(), LatePlanner()).run("Do the late thing.")
        assert ep.success, ep.end_reason
        assert [c["skill"] for c in ep.plan] == [NAME]
        report = _report()
        assert report["skillpack_hash"] == ep.skillpack_hash == catalog_hash()
        assert NAME in [s["skill"] for s in report["skills"]], (
            f"a skill that just ran to completion in a simulated episode sealed under "
            f"{ep.skillpack_hash[:19]} is not in the whitelist `bruhos skills` reports "
            "for that same hash")
    finally:
        _unregister()
