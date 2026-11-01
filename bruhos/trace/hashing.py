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


def _seed(ep: Episode) -> str:
    """Digest of the episode header (everything but the trace_root itself).

    Commits the outcome of the run — success, end_reason, falls, recoveries, plan,
    brain/body identities — alongside the steps, so an edited header no longer verifies.
    """
    header = ep.header()
    header.pop("trace_root", None)
    return sha256_hex(canonical(header))


def seal(ep: Episode) -> str:
    """Fill step hashes and the episode trace_root. Returns trace_root.

    The step chain is seeded with the header digest and trace_root combines that
    digest with the Merkle root of the step hashes, so neither the steps nor the
    outcome header can be edited without failing verify_episode.
    """
    seed = _seed(ep)
    prev = seed
    for s in ep.steps:
        s.hash = step_hash(s, prev)
        prev = s.hash
    merkle = merkle_root([s.hash for s in ep.steps])
    ep.trace_root = sha256_hex(bytes.fromhex(seed) + bytes.fromhex(merkle))
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
