"""`bruhos run` must build its trace sinks from the config's [trace] sinks list.

Before this, cmd_run read only [trace].root and always wrote to a single
LocalSink: a config asking for sinks = ["ipfs"] silently produced a local
trace instead, and a misspelled sink name was dropped without a word.
"""

from __future__ import annotations

import pytest

from bruhos.cli import _sinks, main
from bruhos.trace import ChainSink, IpfsSink, LocalSink


def names(dispatcher) -> list[str]:
    return [s.name for s in dispatcher.sinks]


def test_default_is_local():
    d = _sinks({}, None)
    assert names(d) == ["local"]
    d.close()


def test_config_sinks_are_built_in_order():
    d = _sinks({"trace": {"sinks": ["ipfs", "local"]}}, None)
    assert names(d) == ["ipfs", "local"]
    assert isinstance(d.sinks[0], IpfsSink)
    assert isinstance(d.sinks[1], LocalSink)
    d.close()


def test_chain_sink_is_built():
    d = _sinks({"trace": {"sinks": ["chain"]}}, None)
    assert isinstance(d.sinks[0], ChainSink)
    d.close()


def test_root_and_cli_override_reach_local_sink(tmp_path):
    cfg = {"trace": {"root": str(tmp_path / "from-config"), "sinks": ["local"]}}
    d = _sinks(cfg, None)
    assert str(d.sinks[0].writer.root) == str(tmp_path / "from-config")
    d.close()
    d = _sinks(cfg, str(tmp_path / "from-cli"))
    assert str(d.sinks[0].writer.root) == str(tmp_path / "from-cli")
    d.close()


def test_unknown_sink_name_is_rejected_at_startup():
    with pytest.raises(SystemExit, match="unknown trace sink"):
        _sinks({"trace": {"sinks": ["bogus"]}}, None)


def test_run_with_ipfs_only_writes_no_local_trace(tmp_path, capsys):
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(f'[trace]\nroot = "{tmp_path}/traces"\nsinks = ["ipfs"]\n')
    # No IPFS daemon is listening, so the sink reports an error receipt; the
    # point is that it was dispatched at all, and nothing was written locally.
    assert main(["run", "Wave hello.", "--config", str(cfg)]) == 0
    out = capsys.readouterr().out
    assert "ipfs:" in out
    assert "local:" not in out
    assert not (tmp_path / "traces").exists()


def test_run_rejects_unknown_sink_before_running(tmp_path):
    cfg = tmp_path / "cfg.toml"
    cfg.write_text('[trace]\nsinks = ["bogus"]\n')
    with pytest.raises(SystemExit, match="unknown trace sink"):
        main(["run", "Wave hello.", "--config", str(cfg)])
