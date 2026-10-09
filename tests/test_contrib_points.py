import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "contrib_score", Path(__file__).resolve().parent.parent / "contrib" / "score.py")
score = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(score)

RULES = score.load_rules()
NOW = datetime(2026, 10, 9, tzinfo=timezone.utc)


def pr(n, author, labels=(), merged="2026-10-01T10:00:00Z", files=None, base="main"):
    return {
        "number": n, "title": f"PR {n}", "url": f"https://x/{n}", "author": author,
        "labels": list(labels), "base": base, "created_at": "2026-09-30T00:00:00Z",
        "closed_at": merged or "2026-10-01T10:00:00Z", "merged_at": merged,
        "files": files if files is not None else [{"filename": "bruhos/x.py"}],
    }


def day(i):
    return f"2026-09-{i:02d}T10:00:00Z"


def run(prs, issues=(), rules=RULES, **kw):
    return score.compute(prs, list(issues), rules, NOW, **kw)


def by_login(result):
    return {c["login"]: c for c in result["contributors"]}


def test_merged_pr_points_by_label():
    out = by_login(run([
        pr(1, "plain"), pr(2, "core", labels=["core"]), pr(3, "sec", labels=["security"]),
        pr(4, "docs", labels=["docs"]), pr(5, "chore", labels=["chore", "feature"]),
    ]))
    assert out["plain"]["points"] == 10.0
    assert out["core"]["points"] == 20.0
    assert out["sec"]["points"] == 20.0
    assert out["docs"]["points"] == 3.0
    assert out["chore"]["points"] == 3.0


def test_highest_label_wins():
    assert by_login(run([pr(1, "alice", labels=["docs", "core"])]))["alice"]["points"] == 20.0


def test_unmerged_wrong_branch_and_team_and_bots_ignored():
    prs = [pr(1, "alice", merged=None), pr(2, "alice", base="dev"),
           pr(3, "bruhosdotai"), pr(4, "dependabot[bot]")]
    assert run(prs)["contributors"] == []


def test_bounty_and_no_hardware_bonus():
    a = by_login(run([pr(1, "alice", labels=["hardware-verified", "bounty:50"])]))["alice"]
    assert a["points"] == 10 + 50
    assert a["bounties"] == 50.0
    issues = [{"number": 11, "title": "ran on tonypi", "url": "u", "author": "dave",
               "labels": ["hardware-verified"], "created_at": "2026-10-03T00:00:00Z", "closed_at": None}]
    assert run([], issues)["contributors"] == []


def test_spam_issue_disqualifies():
    issues = [{"number": 12, "title": "x", "url": "u", "author": "alice", "labels": ["spam"],
               "created_at": "2026-10-03T00:00:00Z", "closed_at": "2026-10-03T01:00:00Z"}]
    out = by_login(run([pr(1, "alice", labels=["core"], merged=day(1)), pr(2, "alice", labels=["core"], merged=day(2))], issues))
    assert out["alice"]["disqualified"] and not out["alice"]["eligible"]


def test_daily_cap_and_bounty_exempt():
    prs = [pr(i, "alice", labels=["core"]) for i in range(1, 5)]   # 4 x 20 on one day
    prs.append(pr(9, "alice", labels=["chore", "bounty:50"]))
    a = by_login(run(prs))["alice"]
    assert a["points"] == 60.0 + 50.0


def test_spam_disqualifies():
    prs = [pr(i, "eve", labels=["core"], merged=day(i)) for i in range(1, 4)]
    prs += [pr(9, "eve", merged=None, labels=["spam"]), pr(10, "alice", labels=["core"], merged=day(5)),
            pr(11, "alice", labels=["core"], merged=day(6))]
    out = by_login(run(prs))
    assert out["eve"]["disqualified"] and not out["eve"]["eligible"] and out["eve"]["pool_share"] == 0
    assert out["alice"]["eligible"] and out["alice"]["pool_share"] == pytest.approx(1.0)


def test_min_points_to_be_eligible():
    out = by_login(run([pr(1, "alice"), pr(2, "bob", merged=day(1)), pr(3, "bob", merged=day(2))]))
    assert not out["alice"]["eligible"]       # 10 points
    assert out["bob"]["eligible"]             # 20 points


def test_snapshot_cuts_late_events():
    rules = {**RULES, "pool": {**RULES["pool"], "snapshot_at": "2026-10-02T00:00:00Z"}}
    out = by_login(run([pr(1, "alice"), pr(2, "alice", merged="2026-10-05T00:00:00Z")], rules=rules))
    assert out["alice"]["merged_prs"] == 1


def test_allocate_proportional_min_points_and_cap():
    shares = score.allocate({f"u{i}": 100.0 for i in range(10)} | {"small": 10.0}, 20, 0.15)
    assert "small" not in shares
    assert sum(shares.values()) == pytest.approx(1.0)
    assert shares["u0"] == pytest.approx(0.1)
    capped = score.allocate({"big": 1000.0, **{f"u{i}": 50.0 for i in range(8)}}, 20, 0.15)
    assert capped["big"] == pytest.approx(0.15)
    assert capped["u0"] == pytest.approx(0.85 / 8)
    assert score.allocate({"solo": 500.0}, 20, 0.15) == {"solo": pytest.approx(1.0)}


def test_pool_always_fully_distributed_with_few_contributors():
    four = score.allocate({"a": 925.0, "b": 30.0, "c": 25.0, "d": 20.0}, 20, 0.15)
    assert sum(four.values()) == pytest.approx(1.0)
    assert four["a"] == pytest.approx(0.925)
    seven = score.allocate({"a": 1000.0, **{f"u{i}": 50.0 for i in range(6)}}, 20, 0.15)
    assert seven["a"] == pytest.approx(0.15) and sum(seven.values()) == pytest.approx(1.0)


def test_supply_pct():
    prs = [pr(i, f"u{i}", labels=["core"]) for i in range(10)]
    row = run(prs)["contributors"][0]
    assert row["pool_share"] == pytest.approx(0.1)
    assert row["supply_pct"] == pytest.approx(0.5)


def test_wallet_only_when_added_by_owner():
    w = [{"filename": "contrib/wallets/alice.txt"}]
    prs = [pr(1, "alice", labels=["core"]), pr(2, "alice", files=w), pr(3, "bob", labels=["core"])]
    wallets = {"alice": "0x" + "a" * 40, "bob": "0x" + "b" * 40}
    out = by_login(run(prs, wallets=wallets))
    assert out["alice"]["wallet"] == "0x" + "a" * 40
    assert out["bob"]["wallet"] is None


def test_deterministic():
    prs = [pr(i, f"u{i % 3}", labels=[["core"], ["docs"], []][i % 3], merged=day(i)) for i in range(1, 12)]
    assert run(prs) == run(prs)
