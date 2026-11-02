from bruhos.cli import main


def test_run_returns_failure_when_the_local_trace_cannot_be_saved(tmp_path, capsys):
    destination = tmp_path / "not-a-directory"
    destination.write_text("occupied")
    result = main(["run", "Wave hello.", "--traces", str(destination)])
    output = capsys.readouterr()
    assert '"success": true' in output.out
    assert "error:" in output.out
    assert result != 0
    assert "trace" in output.err.lower()


def test_run_still_returns_success_when_the_trace_is_saved(tmp_path, capsys):
    result = main(["run", "Wave hello.", "--traces", str(tmp_path)])
    capsys.readouterr()
    assert result == 0
    assert len(list(tmp_path.glob("*/episode.json"))) == 1
