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


def pr(n, author, lines=100, labels=(), merged="2026-10-01T10:00:00Z", reviews=(), files=None, base="main"):
    return {
        "number": n, "title": f"PR {n}", "url": f"https://x/{n}", "author": author,
        "labels": list(labels), "base": base, "created_at": "2026-09-30T00:00:00Z",
        "closed_at": merged or "2026-10-01T10:00:00Z", "merged_at": merged,
        "files": files if files is not None else [{"filename": "bruhos/x.py", "additions": lines, "deletions": 0}],
        "reviews": list(reviews),
    }


def rv(user, state="APPROVED", at="2026-10-01T09:00:00Z"):
    return {"user": user, "state": state, "submitted_at": at}


def run(prs, issues=(), rules=RULES, **kw):
    return score.compute(prs, list(issues), rules, NOW, **kw)


def by_login(result):
    return {c["login"]: c for c in result["contributors"]}


def test_merged_pr_points_size_and_label():
    out = by_login(run([pr(1, "alice", lines=300, labels=["core"]), pr(2, "bob", lines=5, labels=["chore"])]))
    assert out["alice"]["points"] == 22.5          # 10 * 1.5 * 1.5
    assert out["bob"]["points"] == 0.9             # 10 * 0.3 * 0.3


def test_highest_label_wins_no_stacking():
    out = by_login(run([pr(1, "alice", lines=100, labels=["docs", "security", "core"])]))
    assert out["alice"]["points"] == 20.0


def test_unmerged_wrong_branch_and_team_and_bots_ignored():
    prs = [pr(1, "alice", merged=None), pr(2, "alice", base="dev"),
           pr(3, "bruhosdotai"), pr(4, "dependabot[bot]")]
    assert run(prs)["contributors"] == []


def test_excluded_paths_do_not_count():
    files = [{"filename": "bun.lock", "additions": 5000, "deletions": 0},
             {"filename": "docs/assets/big.png", "additions": 900, "deletions": 0},
             {"filename": "bruhos/a.py", "additions": 8, "deletions": 0}]
    ev = by_login(run([pr(1, "alice", files=files)]))["alice"]["events"][0]
    assert ev["detail"]["lines"] == 8


def test_reviews_count_once_not_self_and_weekly_cap():
    prs = [pr(i, "alice", reviews=[rv("bob"), rv("bob", "CHANGES_REQUESTED"), rv("alice"), rv("carol", "COMMENTED")],
              merged=f"2026-10-0{i}T10:00:00Z") for i in range(5, 9)]
    out = by_login(run(prs))
    assert out["bob"]["reviews"] == 4
    assert out["bob"]["points"] == 8.0             # 4 x 2, same ISO week, under the 10 cap
    many = [pr(i, "alice", reviews=[rv("bob", at="2026-10-06T09:00:00Z")]) for i in range(1, 9)]
    assert by_login(run(many))["bob"]["points"] == 10.0
    assert "carol" not in out


def test_daily_cap_and_bounty_exempt():
    prs = [pr(i, "alice", lines=1000, labels=["security"]) for i in range(1, 4)]  # 40 each, same day
    prs.append(pr(9, "alice", lines=1, labels=["bounty:50"]))
    a = by_login(run(prs))["alice"]
    assert a["points"] == 60.0 + 50.0
    assert a["bounties"] == 50.0


def test_issue_labels_and_hardware_verified():
    issues = [
        {"number": 10, "title": "bug", "url": "u", "author": "dave", "labels": ["confirmed-bug"],
         "created_at": "2026-10-02T00:00:00Z", "closed_at": None},
        {"number": 11, "title": "ran on tonypi", "url": "u", "author": "dave", "labels": ["hardware-verified"],
         "created_at": "2026-10-03T00:00:00Z", "closed_at": None},
    ]
    d = by_login(run([], issues))["dave"]
    assert d["points"] == 18.0 and d["confirmed_bugs"] == 1 and d["hardware_verified"] == 1


def test_spam_disqualifies():
    prs = [pr(1, "eve", lines=400, labels=["core"]), pr(2, "eve", merged=None, labels=["spam"])]
    prs += [pr(3, "alice", lines=400, labels=["core"])]
    out = run(prs)
    e = by_login(out)["eve"]
    assert e["disqualified"] and not e["eligible"] and e["pool_share"] == 0
    assert by_login(out)["alice"]["eligible"]


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
    alone = score.allocate({"solo": 500.0}, 20, 0.15)
    assert alone == {"solo": pytest.approx(0.15)}


def test_supply_pct_and_totals():
    out = run([pr(i, f"u{i}", lines=300, labels=["core"]) for i in range(10)])
    row = out["contributors"][0]
    assert row["pool_share"] == pytest.approx(0.1)
    assert row["supply_pct"] == pytest.approx(0.5)
    assert out["totals"]["pool_unallocated"] == pytest.approx(0.0, abs=1e-6)


def test_wallet_only_when_added_by_owner():
    w = [{"filename": "contrib/wallets/alice.txt", "additions": 1, "deletions": 0}]
    prs = [pr(1, "alice", lines=300, labels=["core"]), pr(2, "alice", files=w), pr(3, "bob", lines=300, labels=["core"])]
    wallets = {"alice": "0x" + "a" * 40, "bob": "0x" + "b" * 40}
    out = by_login(run(prs, wallets=wallets))
    assert out["alice"]["wallet"] == "0x" + "a" * 40
    assert out["bob"]["wallet"] is None


def test_deterministic():
    prs = [pr(i, f"u{i % 3}", lines=40 * i, reviews=[rv(f"u{(i + 1) % 3}")]) for i in range(1, 12)]
    assert run(prs) == run(prs)
