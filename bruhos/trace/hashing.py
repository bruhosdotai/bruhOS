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
    """Fill step hashes and the episode trace_root. Returns trace_root."""
    prev = sha256_hex(canonical({"episode_id": ep.episode_id, "instruction": ep.instruction,
                                 "skillpack_hash": ep.skillpack_hash}))
    for s in ep.steps:
        s.hash = step_hash(s, prev)
        prev = s.hash
    # With no steps the Merkle root would be the constant hash of empty input,
    # identical for every empty episode and independent of the genesis hash.
    # Commit the genesis hash itself so a step-less episode still has a root
    # that depends on its identity.
    ep.trace_root = prev if not ep.steps else merkle_root([s.hash for s in ep.steps])
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
