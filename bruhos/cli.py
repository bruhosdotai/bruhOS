"""bruhos command line.

    bruhos run "Put the red one in the box."            # kinematic sim + offline rules brain
    bruhos run "Wave hello." --config configs/tonypi.toml --body tonypi
    bruhos run "Wave hello." --config configs/zerollama.toml --brain openai
    bruhos say "Hello." --config configs/zerollama.toml
    bruhos listen --config configs/zerollama.toml
    bruhos web --config configs/zerollama.toml   # PTT + Take picture (see pipeline)
    bruhos ear --config configs/zerollama.toml   # WonderEcho wake → listen → reply
    bruhos scan --config configs/zerollama.toml
    bruhos roam --body sim --steps 30 --seed 0
    bruhos roam --config configs/tonypi.toml --body tonypi --calibrate 5 --out traces/roam.json
    bruhos depth --config configs/tonypi.toml --out /tmp/depth.png   # monocular metres
    bruhos vl-bench --config configs/zerollama.toml --images /tmp/scan
    bruhos ground-exp --config configs/zerollama.toml --images /tmp/scan --backend vl-boxes
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


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _agent_cfg(cfg: dict[str, Any]):
    from bruhos.agent import AgentConfig

    raw = dict(cfg.get("agent", {}))
    return AgentConfig(**{k: v for k, v in raw.items() if k in AgentConfig.__dataclass_fields__})


def _body_cfg_for_agent(cfg: dict[str, Any], agent) -> dict[str, Any]:
    """Merge tonypi.toml when agent.body=tonypi so ear does not use [body]=sim."""
    body_name = agent.body or "tonypi"
    merged = dict(cfg)
    path = agent.body_config
    if path:
        from pathlib import Path

        p = Path(path)
        if not p.is_file() and cfg:
            # relative to the main config's directory if needed
            pass
        if p.is_file():
            with open(p, "rb") as f:
                merged = _deep_merge(tomllib.load(f), merged)
        else:
            # try next to cwd / Sites
            for cand in (Path("/root/Sites/bruhOS") / path, Path(path)):
                if cand.is_file():
                    with open(cand, "rb") as f:
                        merged = _deep_merge(tomllib.load(f), merged)
                    break
    # force body name for TonyPiBody
    merged.setdefault("body", {})
    if isinstance(merged["body"], dict):
        merged["body"] = dict(merged["body"])
        merged["body"]["name"] = body_name
    return merged


def _make_skill_hub(cfg: dict[str, Any], agent):
    from bruhos.skills import SkillHub

    body_name = agent.body or "tonypi"
    body_cfg = _body_cfg_for_agent(cfg, agent)
    body = _body(body_name, body_cfg, fall_prob=0.0, seed=0)
    return SkillHub(body, max_skill_steps=max(6, int(agent.max_steps) * 20))


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


def _speech_cfg(cfg: dict[str, Any]):
    from bruhos.speech import SpeechConfig
    return SpeechConfig(**cfg.get("speech", {}))


def cmd_run(a: argparse.Namespace) -> int:
    from bruhos.runtime import Runtime
    from bruhos.trace import LocalSink, SinkDispatcher

    cfg = _load_config(a.config)
    _speech_cfg(cfg)
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


def cmd_say(a: argparse.Namespace) -> int:
    from bruhos.speech import make_tts

    cfg = _load_config(a.config)
    sc = _speech_cfg(cfg)
    if a.tts:
        sc.tts = a.tts
    if a.playback:
        sc.playback = a.playback
    make_tts(sc).say(a.text)
    return 0


def cmd_listen(a: argparse.Namespace) -> int:
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    from bruhos.listen import normalize_file
    from bruhos.speech import make_asr

    cfg = _load_config(a.config)
    sc = _speech_cfg(cfg)
    if a.asr:
        sc.asr = a.asr

    wav = a.file
    tmp_rec: str | None = None
    norm: Path | None = None
    if not wav:
        arecord = shutil.which("arecord")
        if not arecord:
            raise SystemExit("arecord not found; pass --file path.wav")
        f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        f.close()
        tmp_rec = f.name
        wav = tmp_rec
        print(f"recording {a.seconds}s from {a.device} ...", flush=True)
        subprocess.run(
            [arecord, "-D", a.device, "-f", "S16_LE", "-r", str(a.rate), "-c", str(a.channels),
             "-d", str(a.seconds), wav],
            check=True,
        )
    try:
        src = Path(wav)
        normed = normalize_file(src)
        if normed != src:
            norm = normed
        text = make_asr(sc).transcribe(str(normed))
    finally:
        if tmp_rec:
            Path(tmp_rec).unlink(missing_ok=True)
        if norm is not None:
            norm.unlink(missing_ok=True)
    print(text)
    return 0 if text else 1


def cmd_ground_exp(a: argparse.Namespace) -> int:
    from pathlib import Path

    from bruhos.scan.ground import format_report_table, run_experiments

    cfg = _load_config(a.config) if a.config else {}
    scan = cfg.get("scan", {})
    brain = cfg.get("brain", {})
    cam = cfg.get("camera", {})
    base = a.base_url or scan.get("base_url") or brain.get("base_url")
    backends = a.backend or ["vl-boxes"]
    if "vl-boxes" in backends and not base:
        raise SystemExit("vl-boxes needs [scan].base_url or --base-url")
    out = Path(a.out or "/tmp/ground_exp")
    print(f"ground-exp backends={backends} images={a.images} → {out}", flush=True)
    report = run_experiments(
        image_dir=Path(a.images),
        out_dir=out,
        backends=backends,
        base_url=base,
        vl_model=a.vl_model or scan.get("model") or "ministral-3:3b",
        classes=[c.strip() for c in a.classes.split(",") if c.strip()] if a.classes else None,
        quality_gate=not a.all_images,
        hfov_deg=float(cam.get("hfov_deg", 62.0)),
        timeout=float(a.timeout or scan.get("timeout", 180.0)),
        api_key=scan.get("api_key") or brain.get("api_key") or "",
        yolo_weights=a.yolo_weights,
        florence_model=a.florence_model,
        florence_task=a.florence_task,
    )
    print(format_report_table(report), flush=True)
    print(flush=True)
    print(json.dumps(report.get("summary"), indent=2))
    print(f"wrote {out / 'report.json'}", flush=True)
    return 0 if any(r.get("ok") for r in report.get("rows") or []) else 1


def cmd_vl_bench(a: argparse.Namespace) -> int:
    import urllib.error
    from pathlib import Path

    from bruhos.scan.bench import bench_models, format_bench_table

    cfg = _load_config(a.config)
    scan = cfg.get("scan", {})
    brain = cfg.get("brain", {})
    base = a.base_url or scan.get("base_url") or brain.get("base_url")
    if not base:
        raise SystemExit("need [scan].base_url or --base-url")
    models = a.model if a.model else None
    print(f"benchmarking vision models at {base} on {a.images} ...", flush=True)
    try:
        report = bench_models(
            base_url=base,
            image_dir=Path(a.images),
            models=models,
            timeout=float(a.timeout or scan.get("timeout", 180.0)),
            quality_gate=not a.all_images,
            api_key=scan.get("api_key") or brain.get("api_key") or "",
        )
    except (OSError, RuntimeError, ValueError, urllib.error.URLError) as e:
        raise SystemExit(str(e)) from e
    print(format_bench_table(report), flush=True)
    print(flush=True)
    print(json.dumps(report, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {a.out}", flush=True)
    return 0 if any(r["ok"] for r in report["models"]) else 1


def cmd_depth(a: argparse.Namespace) -> int:
    """One-shot monocular metric depth (Depth Anything V2 indoor Small)."""
    import cv2
    import numpy as np
    from pathlib import Path

    from bruhos.roam.depth import MetricDepth, grab_bgr

    if a.image:
        bgr = cv2.imread(a.image)
        if bgr is None:
            raise SystemExit(f"could not read image {a.image}")
    else:
        cfg = _load_config(a.config) if a.config else {}
        cam = cfg.get("camera", {})
        device = a.device if a.device is not None else cam.get("device", 0)
        cap = cv2.VideoCapture(device)
        if not cap.isOpened():
            raise SystemExit(f"could not open camera {device!r}")
        try:
            w = int(cam.get("width", 640))
            h = int(cam.get("height", 480))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            bgr = grab_bgr(cap)
        finally:
            cap.release()
        if bgr is None:
            raise SystemExit("camera read failed")

    est = MetricDepth(checkpoint=a.checkpoint, input_size=a.input_size)
    band = est.bands(bgr)
    print(
        f"depth L/C/R = {band.left_m:.2f} / {band.center_m:.2f} / {band.right_m:.2f} m "
        f"(min={band.min_m:.2f}, {band.elapsed_s:.1f}s)",
        flush=True,
    )
    if a.out:
        depth = est.infer(bgr)
        # Save a quick vis: normalize band for PNG
        vis = depth.copy()
        lo, hi = float(np.percentile(vis, 5)), float(np.percentile(vis, 95))
        vis = np.clip((vis - lo) / max(1e-6, hi - lo), 0, 1)
        gray = (vis * 255).astype(np.uint8)
        color = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
        split = np.full((bgr.shape[0], 8, 3), 255, dtype=np.uint8)
        combo = np.hstack([bgr, split, color])
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(a.out, combo)
        print(f"wrote {a.out}", flush=True)
    return 0


def cmd_roam(a: argparse.Namespace) -> int:
    from bruhos.roam import RoamConfig, run_roam
    from bruhos.sim.kinematic import SimBody, World

    cfg = _load_config(a.config)
    roam_raw = dict(cfg.get("roam", {}))
    imu = cfg.get("imu", {})
    use_depth = bool(roam_raw.get("use_depth", False))
    if getattr(a, "depth", None) is True:
        use_depth = True
    if getattr(a, "no_depth", False):
        use_depth = False
    rc = RoamConfig(
        step_m=float(roam_raw.get("step_m", 0.05)),
        cell_m=float(roam_raw.get("cell_m", 0.05)),
        progress_ratio=float(roam_raw.get("progress_ratio", 0.4)),
        sample_gap_s=float(roam_raw.get("sample_gap_s", 0.12)),
        max_steps=int(a.steps if a.steps is not None else roam_raw.get("max_steps", 40)),
        max_bumps=int(roam_raw.get("max_bumps", 15)),
        turn_min_deg=float(roam_raw.get("turn_min_deg", 60.0)),
        turn_max_deg=float(roam_raw.get("turn_max_deg", 150.0)),
        fall_forward_deg=float(roam_raw.get("fall_forward_deg", 15.0)),
        fall_back_deg=float(roam_raw.get("fall_back_deg", 160.0)),
        wifi_iface=str(roam_raw.get("wifi_iface", "wlan0")),
        wifi_n=float(roam_raw.get("wifi_n", 2.5)),
        calibrate_steps=int(a.calibrate if a.calibrate is not None else 0),
        use_wifi=not a.no_wifi,
        seed=int(a.seed),
        forward_sign=float(imu.get("forward_sign", 1)),
        use_depth=use_depth,
        depth_every=int(roam_raw.get("depth_every", 3)),
        feet_block_m=float(roam_raw.get("feet_block_m", 0.28)),
        ahead_block_m=float(roam_raw.get("ahead_block_m", 0.45)),
        depth_input_size=int(roam_raw.get("depth_input_size", 252)),
    )
    body_name = a.body
    if body_name == "sim":
        body = SimBody(world=World.room(), seed=a.seed)
    else:
        body = _body(body_name, cfg, fall_prob=0.0, seed=a.seed)
    try:
        result = run_roam(body, rc)
    except (RuntimeError, OSError) as e:
        raise SystemExit(str(e)) from e
    finally:
        body.close()
    print(result.map.ascii(), flush=True)
    print(flush=True)
    payload = result.to_dict()
    print(json.dumps(payload, indent=2))
    if a.out:
        from pathlib import Path
        Path(a.out).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {a.out}", flush=True)
    return 0 if result.end_reason in ("done", "max_steps", "max_bumps") else 1


def cmd_scan(a: argparse.Namespace) -> int:
    from bruhos.scan import format_focus_table, run_scan
    from bruhos.scan.camera_tune import format_retune_summary

    cfg = _load_config(a.config)
    if not cfg.get("head") and not cfg.get("camera"):
        raise SystemExit("scan needs [head] and [camera] in --config (see configs/zerollama.toml)")
    try:
        result = run_scan(
            cfg,
            settle_s=a.settle_ms / 1000.0 if a.settle_ms is not None else None,
            save_dir=a.save_dir,
            retune=a.retune,
        )
    except (ImportError, RuntimeError, OSError, ValueError) as e:
        raise SystemExit(str(e)) from e
    focus = result.get("focus")
    if focus:
        print(format_focus_table(focus), flush=True)
        print(flush=True)
    if result.get("retune"):
        print(format_retune_summary(result["retune"]), flush=True)
        print(flush=True)
    print(json.dumps(result, indent=2))
    return 0


def _chat_from_cfg(cfg: dict, sc: Any) -> Any:
    from bruhos.listen import ChatReply, DEFAULT_SYSTEM

    web = cfg.get("web", {})
    brain = cfg.get("brain", {})
    return ChatReply(
        base_url=brain.get("base_url") or sc.base_url,
        model=brain.get("model"),
        api_key=brain.get("api_key", sc.api_key),
        api_key_env=brain.get("api_key_env", sc.api_key_env),
        timeout=float(brain.get("timeout", sc.timeout)),
        system=web.get("system_prompt") or DEFAULT_SYSTEM,
    )


def cmd_web(a: argparse.Namespace) -> int:
    from bruhos.web import serve

    cfg = _load_config(a.config)
    sc = _speech_cfg(cfg)
    web = cfg.get("web", {})
    host = a.host or web.get("host", "0.0.0.0")
    port = a.port if a.port is not None else int(web.get("port", 8090))
    http_dir = a.http_dir or sc.http_dir or web.get("http_dir", "/tmp/bruh-audio")
    tls = a.tls if a.tls is not None else bool(web.get("tls", False))
    speak = bool(web.get("speak_replies", True))
    max_turns = int(web.get("max_turns", 12))
    chat = _chat_from_cfg(cfg, sc)
    see_cfg = None
    if cfg.get("camera") or cfg.get("scan"):
        see_cfg = cfg
    skills = None
    agent = _agent_cfg(cfg)
    if agent.harness in ("native", "plugin") and agent.body:
        try:
            skills = _make_skill_hub(cfg, agent)
            print(f"[web] skill hub body={agent.body}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[web] skill hub unavailable: {e}", flush=True)
    serve(
        host, port, http_dir, sc, chat,
        speak_replies=speak,
        max_turns=max_turns,
        tls=tls,
        cert_dir=web.get("cert_dir"),
        see_cfg=see_cfg,
        skills=skills,
    )
    return 0


def cmd_ear(a: argparse.Namespace) -> int:
    from bruhos.agent import make_harness
    from bruhos.listen.conversation import Conversation
    from bruhos.listen.ear import EarConfig, run_ear

    cfg = _load_config(a.config)
    sc = _speech_cfg(cfg)
    # Ear wants local speaker; keep http write if configured as both/http
    if sc.playback == "http":
        sc.playback = "both"
    if not sc.aplay_device:
        sc.aplay_device = "plughw:Device,0"
    ear_cfg = EarConfig(**{
        k: v for k, v in cfg.get("ear", {}).items()
        if k in EarConfig.__dataclass_fields__
    })
    if a.port:
        ear_cfg.port = a.port
    if a.device:
        ear_cfg.device = a.device
    if a.seconds is not None:
        ear_cfg.seconds = a.seconds
    if a.debug is not None:
        ear_cfg.debug = a.debug
    chat = _chat_from_cfg(cfg, sc)
    max_turns = int(cfg.get("web", {}).get("max_turns", 16))
    agent = _agent_cfg(cfg)
    hub = None
    harness = None
    try:
        if agent.harness == "plugin":
            harness = make_harness(agent, None, chat)
        else:
            hub = _make_skill_hub(cfg, agent)
            harness = make_harness(agent, hub, chat)
        print(f"[ear] harness={agent.harness} body={agent.body}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[ear] harness unavailable ({e}); chat-only", flush=True)
        harness = None
    try:
        run_ear(
            sc, chat, ear_cfg, Conversation(max_turns=max_turns),
            harness=harness, once=a.once,
        )
    finally:
        if hub is not None:
            try:
                hub.body.close()
            except Exception:  # noqa: BLE001
                pass
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bruhos", description="bruhOS — the skill OS that stands next to the robot.")
    p.add_argument("--version", action="version", version=f"bruhos {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run one English instruction")
    r.add_argument("instruction")
    r.add_argument("--body", default="sim", choices=["sim", "tonypi"])
    r.add_argument("--brain", default=None, help="rules | openai | qwen_vl | groot_s2 | lerobot_policy")
    r.add_argument("--config", help="e.g. configs/tonypi.toml or configs/zerollama.toml")
    r.add_argument("--traces")
    r.add_argument("--fall-prob", type=float, default=0.0)
    r.add_argument("--seed", type=int, default=0)
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("say", help="TTS: speak text (local espeak or OpenAI-compatible remote)")
    s.add_argument("text")
    s.add_argument("--config")
    s.add_argument("--tts", default=None, help="local | openai | print")
    s.add_argument("--playback", default=None, help="aplay | http | both | none")
    s.set_defaults(fn=cmd_say)

    li = sub.add_parser("listen", help="ASR: record or transcribe audio (wav/webm; local or remote)")
    li.add_argument("--config")
    li.add_argument("--asr", default=None, help="local | openai")
    li.add_argument("--file", help="existing audio file; if omitted, arecord is used")
    li.add_argument("--device", default="plughw:Device,0", help="ALSA device for arecord")
    li.add_argument("--seconds", type=int, default=3)
    li.add_argument("--rate", type=int, default=16000)
    li.add_argument("--channels", type=int, default=1)
    li.set_defaults(fn=cmd_listen)

    w = sub.add_parser("web", help="LAN PTT UI + /tts.wav server (use --tls for Safari mic)")
    w.add_argument("--config")
    w.add_argument("--host", default=None)
    w.add_argument("--port", type=int, default=None)
    w.add_argument("--http-dir", default=None, help="directory for /tts.wav etc.")
    w.add_argument("--tls", action=argparse.BooleanOptionalAction, default=None,
                   help="HTTPS with self-signed cert (needed for Safari mic on LAN)")
    w.set_defaults(fn=cmd_web)

    ear = sub.add_parser("ear", help="WonderEcho wake → record → STT → chat → TTS")
    ear.add_argument("--config", required=True, help="e.g. configs/zerollama.toml")
    ear.add_argument("--port", default=None, help="serial port (default [ear].port /dev/ttyUSB0)")
    ear.add_argument("--device", default=None, help="ALSA capture device")
    ear.add_argument("--seconds", type=float, default=None, help="record length after wake")
    ear.add_argument("--once", action="store_true", help="one wake cycle then exit")
    ear.add_argument("--debug", action=argparse.BooleanOptionalAction, default=None,
                     help="verbose ALSA/serial/wav stats (default: [ear].debug)")
    ear.set_defaults(fn=cmd_ear)

    rm = sub.add_parser("roam", help="bump-and-turn walk; IMU bumps + optional WiFi AP fit")
    rm.add_argument("--config", default=None, help="e.g. configs/tonypi.toml")
    rm.add_argument("--body", default="sim", choices=["sim", "tonypi"])
    rm.add_argument("--steps", type=int, default=None, help="max forward steps (default [roam].max_steps)")
    rm.add_argument("--calibrate", type=int, default=None, help="open-floor calibrate steps before roam")
    rm.add_argument("--seed", type=int, default=0)
    rm.add_argument("--out", default=None, help="write JSON map/trail here")
    rm.add_argument("--no-wifi", action="store_true", help="skip WiFi scanner and fit")
    rm.add_argument("--depth", action="store_true", help="vertical depth look (feet+ahead)")
    rm.add_argument("--no-depth", action="store_true", help="disable depth even if [roam].use_depth")
    rm.set_defaults(fn=cmd_roam)

    dp = sub.add_parser(
        "depth",
        help="monocular metric depth (Depth Anything V2 indoor Small) → L/C/R metres",
    )
    dp.add_argument("--config", default=None, help="camera device/size from config")
    dp.add_argument("--image", default=None, help="JPEG/PNG path (else grab from camera)")
    dp.add_argument("--device", type=int, default=None, help="OpenCV camera index")
    dp.add_argument("--checkpoint", default=None, help="path to metric hypersim vits .pth")
    dp.add_argument("--input-size", type=int, default=308, help="model input (lower=faster; default 308 on Pi)")
    dp.add_argument("--out", default=None, help="write RGB|depth preview PNG")
    dp.set_defaults(fn=cmd_depth)

    scn = sub.add_parser("scan", help="pan head over FOV grid; VL inventory via zerollama")
    scn.add_argument("--config", required=True, help="e.g. configs/zerollama.toml")
    scn.add_argument("--settle-ms", type=int, default=None, help="ms to wait after each head move")
    scn.add_argument("--save-dir", default=None, help="write per-pose JPEGs here")
    scn.add_argument("--retune", action=argparse.BooleanOptionalAction, default=None,
                     help="per-pose camera control sweep (default: [scan].retune)")
    scn.set_defaults(fn=cmd_scan)

    vb = sub.add_parser("vl-bench", help="run scan stills through all cozmic VL models; compare")
    vb.add_argument("--config", required=True, help="e.g. configs/zerollama.toml")
    vb.add_argument("--images", required=True, help="directory of jpgs (e.g. /tmp/scan)")
    vb.add_argument("--base-url", default=None, help="override [scan].base_url")
    vb.add_argument("--model", action="append", default=None, help="limit to model id (repeatable)")
    vb.add_argument("--timeout", type=float, default=None)
    vb.add_argument("--all-images", action="store_true", help="skip quality gate; send every jpg")
    vb.add_argument("--out", default=None, help="write full JSON report here")
    vb.set_defaults(fn=cmd_vl_bench)

    ge = sub.add_parser(
        "ground-exp",
        help="grounded detection experiments (bbox + bearing); backends: vl-boxes, yolo-world, florence",
    )
    ge.add_argument("--config", default=None, help="e.g. configs/zerollama.toml")
    ge.add_argument("--images", required=True, help="directory of jpgs (e.g. /tmp/scan)")
    ge.add_argument("--out", default="/tmp/ground_exp", help="overlays + report.json")
    ge.add_argument(
        "--backend", action="append", default=None,
        help="vl-boxes | yolo-world | florence (repeatable; default vl-boxes)",
    )
    ge.add_argument("--base-url", default=None)
    ge.add_argument("--vl-model", default=None, help="model id for vl-boxes backend")
    ge.add_argument("--classes", default=None, help="comma list for yolo-world / florence ground")
    ge.add_argument("--yolo-weights", default="yolov8s-world.pt")
    ge.add_argument("--florence-model", default="microsoft/Florence-2-base")
    ge.add_argument("--florence-task", default="od", help="od | dense | ground | open")
    ge.add_argument("--timeout", type=float, default=None)
    ge.add_argument("--all-images", action="store_true", help="skip quality gate")
    ge.set_defaults(fn=cmd_ground_exp)

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
