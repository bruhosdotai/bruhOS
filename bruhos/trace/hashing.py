"""Canonical JSON + SHA-256 step hashes + Merkle trace_root."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from bruhos.schemas import Episode, Step


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def step_hash(step: Step, prev: str) -> str:
    """Each step commits to the previous one, so reordering or deleting steps changes the root."""
    return sha256_hex(prev.encode() + canonical(step.body()))


def merkle_root(leaves: list[str]) -> str:
    if not leaves:
        return sha256_hex(b"")
    level = [bytes.fromhex(h) for h in leaves]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [hashlib.sha256(level[i] + level[i + 1]).digest() for i in range(0, len(level), 2)]
    return level[0].hex()


def seal(ep: Episode) -> str:
    """Fill step hashes and the episode trace_root. Returns trace_root.

    The genesis hash commits to the episode header (minus trace_root itself),
    so editing success, plan, end_reason or any other header field after sealing
    invalidates the root just as editing a step does.
    """
    genesis = ep.header()
    genesis.pop("trace_root")
    prev = sha256_hex(canonical(genesis))
    for s in ep.steps:
        s.hash = step_hash(s, prev)
        prev = s.hash
    ep.trace_root = merkle_root([s.hash for s in ep.steps])
    return ep.trace_root


def verify_episode(ep: Episode) -> bool:
    claimed = [s.hash for s in ep.steps]
    root = ep.trace_root
    try:
        return seal(ep) == root and [s.hash for s in ep.steps] == claimed
    finally:
        for s, h in zip(ep.steps, claimed):
            s.hash = h
        ep.trace_root = root
