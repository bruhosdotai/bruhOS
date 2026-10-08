"""bruhos command line.

    bruhos run "Put the red one in the box."            # kinematic sim + offline rules brain
    bruhos run "Wave hello." --config configs/tonypi.toml --body tonypi
    bruhos eval --brain rules --fall-prob 0.05
    bruhos skills
    bruhos verify traces/<episode_id>
    bruhos lerobot traces/<episode_id>
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from typing import Any

from bruhos import __version__


def _load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def _brain(name: str | None, cfg: dict[str, Any]):
    from bruhos.brain import make_planner

    bcfg = dict(cfg.get("brain", {}))
    name = name or bcfg.pop("name", "rules")
    bcfg.pop("name", None)
    return make_planner(name, **bcfg)


def _body(name: str, cfg: dict[str, Any], fall_prob: float, seed: int):
    if name == "sim":
        from bruhos.sim.kinematic import SimBody
        return SimBody(fall_prob=fall_prob, seed=seed)
    if name == "tonypi":
        from bruhos.body.tonypi import TonyPiBody
        return TonyPiBody(cfg)
    raise SystemExit(f"unknown body {name!r} (sim | tonypi)")


def cmd_run(a: argparse.Namespace) -> int:
    from bruhos.runtime import Runtime
    from bruhos.speech import SpeechConfig
    from bruhos.trace import LocalSink, SinkDispatcher

    cfg = _load_config(a.config)
    SpeechConfig(**cfg.get("speech", {}))
    body = _body(a.body, cfg, a.fall_prob, a.seed)
    sinks = SinkDispatcher([LocalSink(a.traces or cfg.get("trace", {}).get("root", "traces"))])
    rt = Runtime(body, _brain(a.brain, cfg), sinks)
    try:
        ep = rt.run(a.instruction)
    except KeyboardInterrupt:
        rt.stop()
        return 130
    finally:
        sinks.flush(5)
        body.close()
    for s in ep.steps:
        o = s.outcome
        args = ", ".join(f"{k}={v}" for k, v in s.call["args"].items())
        print(f"  {s.index:>2} {s.kind:<8} {s.call['skill']}({args}) -> {o['status']}  {o['detail']}")
    print(json.dumps({"episode_id": ep.episode_id, "success": ep.success, "end_reason": ep.end_reason,
                      "falls": ep.falls, "recoveries": ep.recoveries, "trace_root": ep.trace_root}, indent=2))
    for _, sink, receipt in sinks.receipts:
        print(f"  {sink}: {receipt}")
    return 0 if ep.success else 1


def cmd_eval(a: argparse.Namespace) -> int:
    from bruhos.eval import run_eval

    cfg = _load_config(a.config)
    report = run_eval(lambda: _brain(a.brain, cfg), fall_prob=a.fall_prob, seed=a.seed)
    for r in report.results:
        mark = "PASS" if r.passed else "FAIL"
        extra = f"  falls={r.falls} recovered={r.recovered}" if r.falls else ""
        why = "" if r.passed else f"  ({r.end_reason}; unmet: {', '.join(r.failed_checks)})"
        print(f"  {mark}  {r.text:<40} {' > '.join(r.plan) or '-'}{extra}{why}")
    print(json.dumps(report.summary(), indent=2))
    return 0


def cmd_skills(a: argparse.Namespace) -> int:
    from bruhos.skills import CATALOG, catalog_hash

    print(json.dumps({"skillpack_hash": catalog_hash(), "skills": CATALOG}, indent=2))
    return 0


def cmd_verify(a: argparse.Namespace) -> int:
    from bruhos.trace import load_episode, verify_episode

    ep = load_episode(a.path)
    ok = verify_episode(ep)
    print(f"{'OK' if ok else 'TAMPERED'}  trace_root={ep.trace_root}  steps={len(ep.steps)}")
    return 0 if ok else 2


def cmd_lerobot(a: argparse.Namespace) -> int:
    from bruhos.trace import LEROBOT_FEATURES, load_episode, to_lerobot_frames

    ep = load_episode(a.path)
    print(json.dumps({"features": LEROBOT_FEATURES, "frames": to_lerobot_frames(ep)}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bruhos", description="bruhOS — the skill OS that stands next to the robot.")
    p.add_argument("--version", action="version", version=f"bruhos {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run one English instruction")
    r.add_argument("instruction")
    r.add_argument("--body", default="sim", choices=["sim", "tonypi"])
    r.add_argument("--brain", default=None, help="rules | openai | qwen_vl | groot_s2 | lerobot_policy")
    r.add_argument("--config")
    r.add_argument("--traces")
    r.add_argument("--fall-prob", type=float, default=0.0)
    r.add_argument("--seed", type=int, default=0)
    r.set_defaults(fn=cmd_run)

    e = sub.add_parser("eval", help="run the 20-prompt English eval in the kinematic sim")
    e.add_argument("--brain", default=None)
    e.add_argument("--config")
    e.add_argument("--fall-prob", type=float, default=0.0)
    e.add_argument("--seed", type=int, default=0)
    e.set_defaults(fn=cmd_eval)

    sub.add_parser("skills", help="print the skill whitelist").set_defaults(fn=cmd_skills)

    v = sub.add_parser("verify", help="re-hash a trace directory")
    v.add_argument("path")
    v.set_defaults(fn=cmd_verify)

    lr = sub.add_parser("lerobot", help="print LeRobot-aligned frames for a trace")
    lr.add_argument("path")
    lr.set_defaults(fn=cmd_lerobot)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
