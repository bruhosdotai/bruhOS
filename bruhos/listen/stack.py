"""Choose the ear's speech and reply servers.

Default is ``local``. The Pi can hear and answer with no other machine up.
``remote`` keeps the zerollama URLs already in the config. That server holds
a larger model already loaded, so a reply comes back faster than TinyLlama
on this CPU. Scan, detect, and the eye stay on whatever the config says.
Why: those jobs are not the spoken turn, and pointing them at TinyLlama
would make vision fail.
"""

from __future__ import annotations

# llama-server on this Pi. The router id is the models-dir folder name.
# Piper is ONNX, so the voice is not a model id here.
LOCAL_BASE = "http://127.0.0.1:11435/v1"
LOCAL_ASR_MODEL = "qwen3-asr"
LOCAL_CHAT_MODEL = "tinyllama"
LOCAL_PIPER_VOICE = "en_US-lessac-medium"
STACKS = ("local", "remote")


def resolve_ear_stack(cfg: dict, override: str | None = None) -> str:
    # Missing key means local. Remote is an upgrade, not the thing a
    # robot should require before it can answer.
    raw = override if override else (cfg.get("ear") or {}).get("stack") or "local"
    name = str(raw).strip().lower()
    if name not in STACKS:
        raise SystemExit(f"unknown ear stack {name!r} (local | remote)")
    return name


def apply_ear_stack(cfg: dict, stack: str) -> dict:
    """Return the config the ear should run. ``remote`` is the config unchanged."""
    name = resolve_ear_stack(cfg, stack)
    if name == "remote":
        return cfg
    out = dict(cfg)
    speech = dict(cfg.get("speech") or {})
    speech.update({
        "asr": "openai",
        "tts": "piper",
        "base_url": LOCAL_BASE,
        "asr_model": LOCAL_ASR_MODEL,
        "tts_model": LOCAL_PIPER_VOICE,
        # Duplex is zerollama's one socket (hear, think, speak). llama-server
        # does not speak that protocol, so local records a clip, then replies.
        "realtime": "off",
    })
    speech.pop("chat_model", None)
    brain = dict(cfg.get("brain") or {})
    brain["name"] = "openai"
    brain["base_url"] = LOCAL_BASE
    brain["model"] = LOCAL_CHAT_MODEL
    out["speech"] = speech
    out["brain"] = brain
    return out
