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
- **Cap:** one contributor receives at most **15% of the pool** (0.75% of total supply). Anything
  above the cap is redistributed to the others. If there are too few contributors for the whole
  pool to be placed under the cap, the remainder is not distributed.
- **Core team** members are not part of this pool.
- **Snapshot:** points are frozen about 30 days before TGE (targeted for end of 2026). The final
  list is published, there is a 7-day window to raise objections, and the allocation unlocks in
  full at TGE.

Live leaderboard: **https://bruhos.ai/contributors/** — updated every 30 minutes by
[`contrib/score.py`](contrib/score.py) with the rules in [`contrib/rules.toml`](contrib/rules.toml).
Anyone can re-run it and get the same numbers.

## How points are earned

### Merged pull requests

```
points = 10 × size multiplier × label multiplier
```

| Lines changed | Multiplier |
| --- | --- |
| ≤ 10 | 0.3 |
| ≤ 50 | 0.7 |
| ≤ 150 | 1.0 |
| ≤ 500 | 1.5 |
| > 500 | 2.0 (max) |

Lines = additions + deletions, excluding lockfiles, images, video, model weights, datasets,
traces and wallet files.

| PR label | Multiplier |
| --- | --- |
| `security` | 2.0 |
| `core` — runtime, brain, trace | 1.5 |
| `skill` / `body` — new skills, robot adapters | 1.3 |
| `feature` | 1.2 |
| `test` | 1.0 |
| `docs` | 0.6 |
| `chore` | 0.3 |
| no label | 1.0 |

The highest matching label counts; labels do not stack. Maintainers set labels at review time.

Examples: a 300-line `core` PR = 10 × 1.5 × 1.5 = **22.5**. A one-line typo fix with `chore` = **0.9**.

### Other work

| What | Points |
| --- | --- |
| `hardware-verified` — the change was run on a real robot, with video or logs attached | +15 |
| `bounty:N` on a merged PR — set by maintainers on priority issues | +N |
| `confirmed-bug` — an issue you opened was confirmed as a real bug | +3 |
| Review — approve or request changes on someone else's PR that gets merged | +2 (max 10 per week) |
| `spam` on any PR or issue | −20 and disqualified |

PRs that are closed without merging cost nothing. Try things.

### Limits

- At most **60 points per contributor per UTC day** (bounties are exempt).
- Only one review per PR per reviewer counts. Reviewing your own PR counts for nothing.
- Bots and core team accounts are ignored.
- Splitting one change into many tiny PRs, generated filler, or farming reviews is treated as spam.

## Registering your wallet

Add one file, `contrib/wallets/<your-github-username>.txt`, containing only your address, and
open a PR from that same GitHub account. A wallet counts only when it was added in a merged PR
authored by its owner. To change it, open another PR.

## Fine print

Points and the leaderboard are an estimate until the snapshot. Rules may be tuned before the
snapshot through public PRs to `contrib/rules.toml`; the rules in force at the snapshot are final.
Token allocations are not a promise of value, and may be subject to the laws of your country.
Contributing does not create employment or any other relationship with bruhOS.

By contributing you agree that your work is licensed under [Apache-2.0](LICENSE).
