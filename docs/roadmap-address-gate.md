# Roadmap: open-mic address gate

**Not in v1.** Press-to-talk already implies command intent, so bruhOS web skips address detection.

When ambient / open mic / WonderEcho always-on lands:

1. **Labels:** `addressed` | `about` | `unrelated` (optimize precision on `addressed`).
2. **Primary:** `POST /v1/systemone` with a Laya-class model (`laya:latest`; also `clef`, `clef-flash`, score models on zerollama).
3. **Lab:** labeled cases + prompt packs + bench vs a heuristic baseline before enabling skill dispatch.
4. **Act only on `addressed`** — feed transcript into the planner; ignore `about` / `unrelated`.

Until then, `POST /api/ptt` returns `{ "assumed_command": true, "executed": false }`.
