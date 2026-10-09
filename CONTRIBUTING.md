# Contributing to bruhOS

Thanks for building the bruh. This page covers how to send work and how contributor points —
and the contributor token pool — are calculated.

## Sending work

1. Open an issue or pick an existing one (`good first issue` and `bounty:N` are good places to start).
2. Fork, branch, and keep the PR focused on one thing.
3. `pip install -e ".[dev]" && pytest` must pass. New skills, brains and bodies need tests.
4. Never change a skill's name or schema to fit a model. The runtime does not bend to checkpoints.
5. A maintainer reviews, labels and merges. Only merged work earns points.

## Contributor pool

**5% of the total bruhOS token supply** is reserved for contributors. It is split by GitHub
contribution points only — nothing else.

```
your tokens = your points ÷ total points of all eligible contributors × 5% of total supply
```

- **Eligible:** at least **20 points** and not disqualified.
- **The whole 5% is always distributed** to eligible contributors. Nothing is kept back.
- **Cap:** with 7 or more eligible contributors, one contributor receives at most **15% of the
  pool** (0.75% of total supply), and anything above the cap is redistributed to the others.
  With fewer than 7, there is no cap and the pool is split purely by points.
- **Core team** members are not part of this pool.
- **Snapshot:** points are frozen about 30 days before TGE (targeted for end of 2026). The final
  list is published, there is a 7-day window to raise objections, and the allocation unlocks in
  full at TGE.

Live leaderboard: **https://bruhos.ai/contributors/** — updated every 30 minutes by
[`contrib/score.py`](contrib/score.py) with the rules in [`contrib/rules.toml`](contrib/rules.toml).
Anyone can re-run it and get the same numbers.

## How points are earned

| What | Points |
| --- | --- |
| Merged PR | **+10** |
| Merged PR labelled `core` or `security` — runtime, brain, trace, security fixes | **+20** |
| Merged PR labelled `docs` or `chore` — docs, typos, housekeeping | **+3** |
| `bounty:N` on a merged PR — set by maintainers on priority issues | **+N** |
| `spam` on any PR or issue | **−20** and disqualified |

Maintainers set labels at review time. If a PR has several labels, the highest-paying one counts.
PRs that are closed without merging cost nothing. Try things.

### Limits

- At most **60 points per contributor per UTC day** (bounties are exempt).
- Bots and core team accounts are ignored.
- Splitting one change into many tiny PRs or generated filler is labelled `chore` or `spam`.

## Fine print

Points and the leaderboard are an estimate until the snapshot. Rules may be tuned before the
snapshot through public PRs to `contrib/rules.toml`; the rules in force at the snapshot are final.
Token allocations are not a promise of value, and may be subject to the laws of your country.
Contributing does not create employment or any other relationship with bruhOS.

By contributing you agree that your work is licensed under [Apache-2.0](LICENSE).
