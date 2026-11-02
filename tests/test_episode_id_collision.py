import random
import time
from types import SimpleNamespace
import uuid

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.runtime import runtime as runtime_module
from bruhos.sim.kinematic import SimBody
from bruhos.trace import TraceWriter, load_episode, verify_episode


def same_prefix_uuids():
    """Find two distinct valid UUID4 values in a fixed synthetic UUID stream."""
    rng = random.Random(0)
    seen = {}
    for _ in range(20_000):
        value = uuid.UUID(int=rng.getrandbits(128), version=4)
        prefix = value.hex[:6]
        if prefix in seen:
            assert value != seen[prefix]
            return seen[prefix], value
        seen[prefix] = value
    raise AssertionError('Fixed stream no longer contains the expected prefix collision')


def episodes(monkeypatch, values):
    values = iter(values)
    monkeypatch.setattr(runtime_module, 'uuid', SimpleNamespace(uuid4=lambda: next(values)))
    monkeypatch.setattr(runtime_module, 'time', SimpleNamespace(
        time=time.time, strftime=lambda _: '20261102T110000'))
    runtime = Runtime(SimBody(), RulesPlanner())
    return runtime.run('Wave hello.'), runtime.run('Stop.')


def test_distinct_uuid_values_keep_distinct_episode_ids_within_one_second(monkeypatch):
    first, second = episodes(monkeypatch, same_prefix_uuids())
    assert first.episode_id != second.episode_id


def test_second_episode_does_not_replace_the_first_saved_trace(tmp_path, monkeypatch):
    first, second = episodes(monkeypatch, same_prefix_uuids())
    writer = TraceWriter(tmp_path)
    first_path = writer.write(first)
    writer.write(second)
    saved = load_episode(first_path)
    assert saved.instruction == first.instruction
    assert saved.trace_root == first.trace_root
    assert verify_episode(saved)
    assert len(list(tmp_path.glob('*/episode.json'))) == 2


def test_existing_different_prefix_case_keeps_two_verifiable_records(tmp_path, monkeypatch):
    values = [uuid.UUID('11111111-1111-4111-8111-111111111111'),
              uuid.UUID('22222222-2222-4222-8222-222222222222')]
    first, second = episodes(monkeypatch, values)
    assert first.episode_id.startswith('20261102T110000-')
    assert second.episode_id.startswith('20261102T110000-')
    writer = TraceWriter(tmp_path)
    assert verify_episode(load_episode(writer.write(first)))
    assert verify_episode(load_episode(writer.write(second)))
    assert first.episode_id != second.episode_id
