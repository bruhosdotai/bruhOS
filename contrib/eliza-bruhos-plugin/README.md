# @bruhos/plugin-skills (eliza / odiza)

Reference plugin so an elizaOS or odiza agent can move the TonyPi **without** importing Hiwonder on the agent host.

## Contract

- Pi exposes `POST http://<pi>:8090/v1/skills/run` with body `{ "skill": "go", "args": { "direction": "forward", "steps": 3 } }`.
- Response: `{ "skill", "args", "status", "detail", "data", "ok", "episode_id" }`.
- Plugin registers one Action per skill. The handler only HTTP-POSTs to the Pi and returns `{ success, text: detail, data }`.
- Ear (or any client) talks to the agent via `POST /bruhos/message` `{ "text", "roomId" }` → `{ "text" }` after the turn finishes (actions already ran).

```
ear STT ──► agent /bruhos/message ──► Action.handler ──► Pi /v1/skills/run ──► Outcome
                              └──► spoken text back to ear TTS
```

Selecting odiza vs upstream eliza is which process loads this plugin. The wire format is the same.

## Env

| Variable | Meaning |
| --- | --- |
| `BRUHOS_SKILLS_URL` | e.g. `http://192.168.1.50:8090/v1/skills/run` |
| `BRUHOS_SKILLS_TOKEN` | optional bearer token (Pi has no auth today) |

## Install sketch

```bash
# on the agent host (not the Pi)
cp -r contrib/eliza-bruhos-plugin /path/to/your/eliza/plugins/bruhos
# register in character plugins: "@bruhos/plugin-skills"
```

See `src/index.ts` for the Action handler shape (`name`, `similes`, `description`, `validate`, `handler` → `{ success, text, data }`).
