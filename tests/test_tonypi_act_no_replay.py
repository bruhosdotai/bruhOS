"""act() must run each primitive exactly once, even when the SDK raises.

On main, ``act()`` fell back on *any* ``TypeError``:

```python
try:
    self.agc.runActionGroup(group, **kw)
except TypeError:
    self.agc.runActionGroup(group)      # replay
```

The fallback exists for old firmware images whose ``runActionGroup`` takes no
``path`` argument. But a ``TypeError`` raised *inside* the SDK (the usual
``ret[0]`` on ``None`` shape) also triggers it, so one primitive runs the action
group twice -- a humanoid asked for ``forward`` walks twice as far -- and the
original error is replaced by whatever the replay does.
"""

from __future__ import annotations

import pytest

from bruhos.body.tonypi import TonyPiBody, _accepts_path_kwarg


class WrappedAgc:
    """Modern SDK: accepts path=."""

    def __init__(self, exc: BaseException | None = None):
        self.calls: list[dict] = []
        self.exc = exc

    def runActionGroup(self, name, path=None, **kw):
        self.calls.append({"name": name, "path": path})
        if self.exc:
            raise self.exc
        return 0


class OldAgc:
    """Old firmware image: runActionGroup(name) only."""

    def __init__(self):
        self.calls: list[str] = []

    def runActionGroup(self, name):
        self.calls.append(name)
        return 0


def _body(agc, action_path="/home/pi/ActionGroups"):
    b = object.__new__(TonyPiBody)
    b.name = "tonypi"
    b.actions = {"forward": "go_forward", "stand": "stand", "grab": "grab", "walk_toward": "go_forward"}
    b.agc = agc
    b.action_path = action_path
    b._agc_takes_path = _accepts_path_kwarg(agc.runActionGroup)   # mirrors __init__
    import threading

    b._lock = threading.Lock()
    b._halted = threading.Event()
    b.holding = None
    b.turn_step_deg = 15.0
    return b


def test_error_inside_the_action_group_is_not_replayed():
    agc = WrappedAgc(exc=TypeError("'NoneType' object is not subscriptable"))
    body = _body(agc)
    with pytest.raises(TypeError):
        body.act("forward")
    assert [c["name"] for c in agc.calls] == ["go_forward"], "action group replayed after an SDK failure"


def test_old_firmware_without_path_kwarg_still_runs_once():
    agc = OldAgc()
    body = _body(agc)
    assert body.act("forward") is True
    assert agc.calls == ["go_forward"]


def test_path_is_passed_when_the_sdk_supports_it():
    agc = WrappedAgc()
    body = _body(agc)
    assert body.act("forward") is True
    assert agc.calls == [{"name": "go_forward", "path": "/home/pi/ActionGroups"}]
