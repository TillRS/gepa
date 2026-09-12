"""Verify prompt-only campaign priority without launching model or Docker work."""

import shlex
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from examples.common.experiment_models import EXPERIMENT_MODELS
from examples.terminalbench import run_ablations
from examples.terminalbench.main import REPO_ROOT, build_parser


@pytest.mark.parametrize("model", EXPERIMENT_MODELS)
@pytest.mark.parametrize("dry_run", [False, True])
def test_campaign_runs_prompt_only_first_and_forwards_shared_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], model: str, dry_run: bool
) -> None:
    """Exercise the launcher with twelve separate cells and six prompt-only runs first."""
    runner = Mock()
    monkeypatch.setattr(run_ablations.subprocess, "run", runner)
    root = tmp_path / "campaign with spaces"
    run_ablations.main(
        [
            "--run-root",
            str(root),
            *(["--dry-run"] if dry_run else []),
            "--student-model",
            model,
            "--proposer-model",
            model,
            "--student-api-base",
            "http://localhost:8000/v1",
            "--proposer-api-base",
            "http://localhost:8000/v1",
            "--seed",
            "17",
            "--n-concurrent",
            "3",
        ]
    )
    commands = [shlex.split(line) for line in capsys.readouterr().out.splitlines()]
    assert len(commands) == 12
    assert all(
        command[:6] == ["uv", "run", "--no-sync", "python", "-m", "examples.terminalbench.main"] for command in commands
    )
    cells = [build_parser().parse_args(command[6:]) for command in commands]
    assert [cell.optimization_scope for cell in cells] == ["system_prompt"] * 6 + ["all_text"] * 6
    assert [(cell.condition, cell.budget) for cell in cells[:6]] == [
        ("vanilla", "standard"),
        ("react_v2", "standard"),
        ("react_v2_random", "standard"),
        ("action", "standard"),
        ("vanilla", "double"),
        ("react_v2", "double"),
    ]
    assert [(cell.condition, cell.budget) for cell in cells[6:]] == [
        (cell.condition, cell.budget) for cell in cells[:6]
    ]
    assert len({cell.run_dir for cell in cells}) == 12
    for cell in cells:
        assert cell.run_dir.parent == root / cell.optimization_scope
        assert cell.harbor_work_dir == cell.run_dir / "harbor"
        assert cell.student_model == cell.proposer_model == model
        assert cell.student_api_base == cell.proposer_api_base == "http://localhost:8000/v1"
        assert cell.seed == 17 and cell.n_concurrent == 3
    assert not root.exists()
    if dry_run:
        runner.assert_not_called()
    else:
        assert [call.args[0] for call in runner.call_args_list] == commands
        assert all(call.kwargs == {"cwd": REPO_ROOT, "check": True} for call in runner.call_args_list)


def test_campaign_stops_before_later_cells_when_a_run_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep later ablations from launching after a failed optimization run."""
    runner = Mock(side_effect=[None, subprocess.CalledProcessError(1, "optimization")])
    monkeypatch.setattr(run_ablations.subprocess, "run", runner)
    with pytest.raises(subprocess.CalledProcessError):
        run_ablations.main(["--run-root", str(tmp_path)])
    assert runner.call_count == 2


@pytest.mark.parametrize(
    "option", ["--optimization-scope=all_text", "--condition=action", "--budget=double", "--run-dir=/tmp"]
)
def test_campaign_rejects_overrides_to_its_order_and_run_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, option: str
) -> None:
    """Reject conflicting campaign flags before any cell can launch."""
    runner = Mock()
    monkeypatch.setattr(run_ablations.subprocess, "run", runner)
    with pytest.raises(SystemExit):
        run_ablations.main(["--run-root", str(tmp_path), option])
    runner.assert_not_called()
