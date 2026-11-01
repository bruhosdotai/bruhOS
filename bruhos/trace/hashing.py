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
    """Merkle root of the step hashes.

    Odd nodes are promoted, never duplicated, and the leaf count is folded into
    the root. Duplicating the trailing leaf made ``[a, b, c]`` and
    ``[a, b, c, c]`` hash to the same root, so a ``trace_root`` did not commit
    to a unique step list (the CVE-2012-2459 class of ambiguity).
    """
    if not leaves:
        return sha256_hex(b"\x00")
    level = [bytes.fromhex(h) for h in leaves]
    while len(level) > 1:
        nxt = [hashlib.sha256(level[i] + level[i + 1]).digest()
               for i in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            nxt.append(level[-1])          # promote, never duplicate
        level = nxt
    return sha256_hex(b"\x01" + len(leaves).to_bytes(8, "big") + level[0])


def seal(ep: Episode) -> str:
    """Fill step hashes and the episode trace_root. Returns trace_root."""
    prev = sha256_hex(canonical({"episode_id": ep.episode_id, "instruction": ep.instruction,
                                 "skillpack_hash": ep.skillpack_hash}))
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
