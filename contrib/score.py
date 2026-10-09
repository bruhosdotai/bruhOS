"""Contributor points for the bruhOS contributor pool.

Reads merged PRs and spam-labelled issues from the GitHub API, applies
contrib/rules.toml, and writes points.json (leaderboard + per-event breakdown).

    GITHUB_TOKEN=... python contrib/score.py --repo bruhosdotai/bruhOS --out points.json

Stdlib only. Deterministic for a given set of inputs and `now`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tomllib
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

API = "https://api.github.com"
RULES = Path(__file__).with_name("rules.toml")
WALLETS_DIR = "contrib/wallets/"
BOUNTY_RE = re.compile(r"^bounty:(\d+(?:\.\d+)?)$")
WALLET_RE = re.compile(r"^(0x[0-9a-fA-F]{40}|[1-9A-HJ-NP-Za-km-z]{32,44})$")


# ---------------------------------------------------------------- rules ----

def load_rules(path: Path = RULES) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def pr_points(labels: list[str], table: dict, default: float) -> tuple[float, str | None]:
    """Points for a merged PR: the highest-paying label wins, otherwise the default."""
    hits = [(float(table[l]), l) for l in labels if l in table]
    if not hits:
        return default, None
    return max(hits)


def bounty(labels: list[str]) -> float:
    return sum(float(m.group(1)) for l in labels if (m := BOUNTY_RE.match(l)))


def is_bot(login: str, user_type: str = "User") -> bool:
    return user_type == "Bot" or login.endswith("[bot]")


# --------------------------------------------------------------- events ----

def build_events(prs: list[dict], issues: list[dict], rules: dict) -> list[dict]:
    """Raw scoring events, before caps. Each event: user, kind, ref, at, points, detail."""
    pts, repo = rules["points"], rules["repo"]
    ev: list[dict] = []

    for pr in prs:
        author, labels = pr["author"], pr["labels"]
        ref = {"type": "pr", "number": pr["number"], "title": pr["title"], "url": pr["url"]}
        when = pr["merged_at"] or pr["closed_at"] or pr["created_at"]

        if "spam" in labels:
            ev.append(dict(user=author, kind="spam", ref=ref, at=when, points=pts["spam"], detail={}))
            continue
        if not pr["merged_at"] or pr.get("base") != repo["branch"]:
            continue

        p, lab = pr_points(labels, pts["labels"], pts["merged_pr"])
        ev.append(dict(user=author, kind="merged_pr", ref=ref, at=pr["merged_at"],
                       points=p, detail={"label": lab}))
        if (b := bounty(labels)) > 0:
            ev.append(dict(user=author, kind="bounty", ref=ref, at=pr["merged_at"], points=b, detail={}))

    for it in issues:
        ref = {"type": "issue", "number": it["number"], "title": it["title"], "url": it["url"]}
        if "spam" in it["labels"]:
            ev.append(dict(user=it["author"], kind="spam", ref=ref,
                           at=it["closed_at"] or it["created_at"], points=pts["spam"], detail={}))
    return ev


def apply_caps(events: list[dict], rules: dict) -> list[dict]:
    """Daily cap on positive points; bounties are exempt."""
    daily_cap = rules["caps"]["daily_points"]
    day_used: dict[tuple, float] = defaultdict(float)
    out = []
    for e in sorted(events, key=lambda e: (e["at"], e["ref"]["number"], e["kind"], e["user"])):
        e = dict(e, credited=e["points"])
        if e["points"] > 0:
            ts = parse_ts(e["at"])
            if e["kind"] != "bounty":
                dk = (e["user"], ts.date().isoformat())
                e["credited"] = max(0.0, min(e["credited"], daily_cap - day_used[dk]))
                day_used[dk] += e["credited"]
        e["credited"] = round(e["credited"], 4)
        out.append(e)
    return out


def effective_cap(n_eligible: int, max_share: float) -> float:
    """The pool is always fully distributed: when the cap cannot place the whole pool
    (fewer than 1/max_share eligible contributors), there is no cap."""
    return max_share if n_eligible * max_share >= 1.0 - 1e-12 else 1.0


def allocate(points: dict[str, float], min_points: float, max_share: float) -> dict[str, float]:
    """Pool share per eligible user, proportional to points, capped at the effective
    cap. Excess above the cap is redistributed to the others, so shares sum to 1."""
    active = {u: p for u, p in points.items() if p >= min_points and p > 0}
    max_share = effective_cap(len(active), max_share)
    shares: dict[str, float] = {}
    remaining = 1.0
    while active:
        total = sum(active.values())
        over = [u for u, p in active.items() if p / total * remaining > max_share + 1e-12]
        if not over:
            shares.update({u: p / total * remaining for u, p in active.items()})
            break
        for u in over:
            shares[u] = max_share
            remaining -= max_share
            del active[u]
    return shares


# -------------------------------------------------------------- compute ----

def compute(prs: list[dict], issues: list[dict], rules: dict, now: datetime,
            wallets: dict[str, str] | None = None, avatars: dict[str, str] | None = None,
            repo_name: str = "") -> dict:
    pool, repo = rules["pool"], rules["repo"]
    team = {t.lower() for t in repo["team"]}
    snapshot = parse_ts(pool.get("snapshot_at") or None)
    wallets, avatars = wallets or {}, avatars or {}

    def counted(login: str, typ: str = "User") -> bool:
        return bool(login) and login.lower() not in team and not is_bot(login, typ)

    prs = [p for p in prs if counted(p["author"], p.get("author_type", "User"))]
    issues = [i for i in issues if counted(i["author"], i.get("author_type", "User"))]

    events = build_events(prs, issues, rules)
    if snapshot:
        events = [e for e in events if parse_ts(e["at"]) <= snapshot]
    events = apply_caps(events, rules)

    # A wallet counts only if the owner added it in their own merged PR.
    wallet_ok = {
        p["author"] for p in prs if p["merged_at"]
        and any(f["filename"] == f"{WALLETS_DIR}{p['author']}.txt" for f in p["files"])
    }

    users: dict[str, dict] = {}
    for e in events:
        u = users.setdefault(e["user"], {
            "login": e["user"], "points": 0.0, "merged_prs": 0, "bounties": 0.0,
            "disqualified": False, "first_at": e["at"], "last_at": e["at"], "events": [],
        })
        u["points"] += e["credited"]
        u["last_at"] = max(u["last_at"], e["at"])
        u["first_at"] = min(u["first_at"], e["at"])
        k = e["kind"]
        if k == "merged_pr": u["merged_prs"] += 1
        elif k == "bounty": u["bounties"] += e["credited"]
        elif k == "spam": u["disqualified"] = True
        u["events"].append({x: e[x] for x in ("kind", "ref", "at", "points", "credited", "detail")})

    for u in users.values():
        u["points"] = round(max(0.0, u["points"]), 2)
        u["events"].sort(key=lambda e: e["at"], reverse=True)

    eligible_pts = {l: u["points"] for l, u in users.items() if not u["disqualified"]}
    shares = allocate(eligible_pts, pool["min_points"], pool["max_share"])

    rows = []
    for login, u in users.items():
        share = shares.get(login, 0.0)
        w = wallets.get(login)
        rows.append({
            **u,
            "avatar": avatars.get(login, f"https://github.com/{login}.png?size=96"),
            "eligible": login in shares,
            "pool_share": round(share, 6),
            "supply_pct": round(share * pool["supply_pct"], 6),
            "wallet": w if (w and login in wallet_ok) else None,
        })
    rows.sort(key=lambda r: (-r["points"], r["login"].lower()))
    for i, r in enumerate(rows, 1):
        r["rank"] = i

    recent = sorted(
        ({"user": e["user"], **{x: e[x] for x in ("kind", "ref", "at", "credited")}} for e in events),
        key=lambda e: e["at"], reverse=True,
    )[:30]
    allocated = sum(shares.values())
    return {
        "generated_at": now.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "repo": repo_name,
        "rules": {
            "supply_pct": pool["supply_pct"], "min_points": pool["min_points"],
            "max_share": pool["max_share"],
            "effective_max_share": round(effective_cap(len(shares), pool["max_share"]), 6),
            "snapshot_at": pool.get("snapshot_at") or None,
            "daily_points": rules["caps"]["daily_points"],
        },
        "totals": {
            "contributors": len(rows),
            "eligible": len(shares),
            "points": round(sum(r["points"] for r in rows), 2),
            "eligible_points": round(sum(eligible_pts.get(l, 0) for l in shares), 2),
            "merged_prs": sum(r["merged_prs"] for r in rows),
            "pool_allocated": round(allocated, 6),
            "pool_unallocated": round(1 - allocated, 6),
        },
        "contributors": rows,
        "recent": recent,
    }


# --------------------------------------------------------------- github ----

class GitHub:
    def __init__(self, repo: str, token: str | None):
        self.repo, self.token = repo, token
        self.calls = 0

    def get(self, path: str, **params) -> list | dict:
        url = f"{API}/repos/{self.repo}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "bruhos-contrib-points",
            **({"Authorization": f"Bearer {self.token}"} if self.token else {}),
        })
        self.calls += 1
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)

    def pages(self, path: str, **params) -> list:
        out, page = [], 1
        while True:
            batch = self.get(path, per_page=100, page=page, **params)
            out.extend(batch)
            if len(batch) < 100:
                return out
            page += 1


def _labels(obj: dict) -> list[str]:
    return [l["name"] for l in obj.get("labels", [])]


def fetch(gh: GitHub, cache: dict) -> tuple[list[dict], list[dict], dict[str, str], dict]:
    avatars: dict[str, str] = {}
    prs, new_cache = [], {}
    for p in gh.pages("/pulls", state="closed", sort="created", direction="asc"):
        user = p.get("user") or {}
        login = user.get("login", "")
        avatars[login] = user.get("avatar_url", "")
        item = {
            "number": p["number"], "title": p["title"], "url": p["html_url"],
            "author": login, "author_type": user.get("type", "User"),
            "labels": _labels(p), "base": p["base"]["ref"],
            "created_at": p["created_at"], "closed_at": p["closed_at"], "merged_at": p["merged_at"],
            "files": [],
        }
        if p["merged_at"]:
            key = str(p["number"])
            if key in cache and "files" in cache[key]:
                item["files"] = cache[key]["files"]
            else:
                item["files"] = [{"filename": f["filename"]} for f in gh.pages(f"/pulls/{p['number']}/files")]
            new_cache[key] = {"files": item["files"]}
        prs.append(item)

    issues = []
    for i in gh.pages("/issues", state="all"):
        if "pull_request" in i:
            continue
        labels = _labels(i)
        if "spam" not in labels:
            continue
        user = i.get("user") or {}
        avatars[user.get("login", "")] = user.get("avatar_url", "")
        issues.append({
            "number": i["number"], "title": i["title"], "url": i["html_url"],
            "author": user.get("login", ""), "author_type": user.get("type", "User"),
            "labels": labels, "created_at": i["created_at"], "closed_at": i.get("closed_at"),
        })
    return prs, issues, {k: v for k, v in avatars.items() if k and v}, new_cache


def read_wallets(root: Path) -> dict[str, str]:
    out = {}
    d = root / WALLETS_DIR
    if d.is_dir():
        for f in d.glob("*.txt"):
            addr = f.read_text().strip()
            if WALLET_RE.match(addr):
                out[f.stem] = addr
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", "bruhosdotai/bruhOS"))
    ap.add_argument("--rules", type=Path, default=RULES)
    ap.add_argument("--out", type=Path, default=Path("points.json"))
    ap.add_argument("--cache", type=Path, help="per-PR file list cache (read and rewritten)")
    args = ap.parse_args(argv)

    cache = {}
    if args.cache and args.cache.is_file():
        try:
            cache = json.loads(args.cache.read_text())
        except json.JSONDecodeError:
            cache = {}

    gh = GitHub(args.repo, os.environ.get("GITHUB_TOKEN"))
    prs, issues, avatars, new_cache = fetch(gh, cache)
    result = compute(prs, issues, load_rules(args.rules), datetime.now(timezone.utc),
                     wallets=read_wallets(Path(__file__).resolve().parent.parent),
                     avatars=avatars, repo_name=args.repo)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    if args.cache:
        args.cache.write_text(json.dumps(new_cache, separators=(",", ":")) + "\n")
    t = result["totals"]
    print(f"{t['contributors']} contributors, {t['eligible']} eligible, {t['points']} points, "
          f"{t['merged_prs']} merged PRs, {gh.calls} API calls -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
