"""Offline checks for frozen Terminal-Bench comparisons and repeated Pass@1."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from examples.common.experiment_models import DEEPSEEK_V4_FLASH_MODEL, QWEN3_8_27B_MODEL
from examples.terminalbench import evaluate
from examples.terminalbench.main import (
    CAMPAIGN_CELLS,
    EXPERIMENT_MANIFESTS,
    RUN_CONTRACT_FILENAME,
    build_parser,
    build_run_contract,
    ensure_run_contract,
    seed_candidate,
)
from gepa.adapters.terminal_bench_adapter import (
    HarborEvaluation,
    HarborExecutionError,
    HarborTrialResult,
    load_terminalbench_manifest,
)
from gepa.core.state import GEPAState, ValsetEvaluation


def _write_run(
    root: Path, experiment: str, condition: str, model: str = QWEN3_8_27B_MODEL, budget: str = "standard"
) -> Path:
    """Create a real checkpoint whose validation winner is not the last candidate."""
    label = f"{condition}{'_2x' if budget == 'double' else ''}"
    run_dir = root / label
    args = build_parser().parse_args(
        [
            "--experiment",
            experiment,
            "--condition",
            condition,
            "--budget",
            budget,
            "--student-model",
            model,
            "--proposer-model",
            model,
            "--student-api-base",
            "http://localhost:8000/v1",
            "--proposer-api-base",
            "http://localhost:8000/v1",
            "--run-dir",
            str(run_dir),
            "--harbor-work-dir",
            str(run_dir / "harbor"),
        ]
    )
    manifest = load_terminalbench_manifest(EXPERIMENT_MANIFESTS[experiment])
    initial, family = seed_candidate(model, "auto", experiment)
    contract = build_run_contract(args, manifest, manifest.tasks("train"), manifest.tasks("val"), condition, family)
    ensure_run_contract(run_dir, contract)
    count = len(manifest.splits["val"])
    state = GEPAState(initial, ValsetEvaluation({}, dict.fromkeys(range(count), 0.0)))
    state.i = contract["optimization_budget"]["max_iterations"] - 1
    state.total_num_evals = 1000
    state.num_full_ds_evals = 3
    component = next(iter(initial))
    for name, score in [("winner", 1.0), ("last", 0.0)]:
        candidate = {**initial, component: initial[component] + f"\n{label}-{name}"}
        state.update_state_with_new_program(
            [0],
            candidate,
            ValsetEvaluation({}, dict.fromkeys(range(count), score)),
            None,
            100,
            iteration_id=f"{condition}-{name}",
        )
    state.save(str(run_dir))
    return run_dir


def _write_comparison(root: Path, experiment: str, model: str = QWEN3_8_27B_MODEL) -> dict[str, Path]:
    """Create all six distinct standard/double-budget source checkpoints."""
    return {
        label: _write_run(root, experiment, condition, model, budget)
        for label, (condition, budget) in CAMPAIGN_CELLS.items()
    }


def _fake_runner(manifest, comparison, output_dir: Path, *, fail_on_call: int | None = None) -> Mock:
    """Return distinct Harbor jobs with repetition scores zero, one-half, and one."""
    completed = Counter()
    attempts = 0
    by_digest = {harness["candidate_digest"]: label for label, harness in comparison["harnesses"].items()}

    def run(task_ids, candidate):
        """Require freezing first and emit one binary reward per held-out task."""
        nonlocal attempts
        attempts += 1
        assert json.loads((output_dir / evaluate.FROZEN_COMPARISON_FILENAME).read_text()) == comparison
        assert task_ids == manifest.splits["test"]
        if attempts == fail_on_call:
            raise HarborExecutionError("simulated interrupted Harbor job")
        digest = manifest.candidate_digest(candidate)
        label = by_digest[digest]
        completed[label] += 1
        passes = (completed[label] - 1) * len(task_ids) // 2
        trials = {
            task_id: HarborTrialResult(
                task_id=task_id,
                reward=float(index < passes),
                rewards={"reward": float(index < passes)},
                errors=[],
                atif_trajectories=[],
                raw_result={},
                trial_dir=output_dir / f"trial-{index}",
            )
            for index, task_id in enumerate(task_ids)
        }
        return HarborEvaluation(
            evaluation_id=f"evaluation-{attempts}",
            candidate_digest=digest,
            config_path=output_dir / f"job-{attempts}.json",
            job_dir=output_dir / f"job-{attempts}",
            returncode=0,
            stdout_path=output_dir / "stdout.log",
            stderr_path=output_dir / "stderr.log",
            trials=trials,
        )

    return Mock(run=Mock(side_effect=run))


@pytest.mark.parametrize("experiment", EXPERIMENT_MANIFESTS)
@pytest.mark.parametrize("model", [QWEN3_8_27B_MODEL, DEEPSEEK_V4_FLASH_MODEL])
def test_evaluation_cli_freezes_validation_winners_and_repeats_test_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, experiment: str, model: str
) -> None:
    """Test all benchmark/model arms through the CLI without any optimization or model call."""
    run_dirs = _write_comparison(tmp_path, experiment, model)
    manifest, comparison = evaluate.freeze_comparison(run_dirs)
    for condition in CAMPAIGN_CELLS:
        assert comparison["source_runs"][condition]["selected_candidate_index"] == 1
        epochs = 8 if condition.endswith("_2x") else 4
        assert comparison["source_runs"][condition]["contract"]["optimization_budget"]["training_epochs"] == epochs
        assert any(f"{condition}-winner" in text for text in comparison["harnesses"][condition]["documents"].values())
    output_dir = tmp_path / "test"
    runner = _fake_runner(manifest, comparison, output_dir)
    factory = Mock(return_value=runner)
    monkeypatch.setattr(evaluate, "HarborCLI", factory)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            *[argument for label, path in run_dirs.items() for argument in ("--run-dir", f"{label}={path}")],
            "--output-dir",
            str(output_dir),
        ],
    )

    evaluate.main()
    assert runner.run.call_count == 21
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["complete"] is True
    assert summary["protocol"]["optimization_runs_per_configuration"] == 1
    assert summary["protocol"]["attempts_per_task_per_repetition"] == 1
    for scores in summary["harnesses"].values():
        assert scores["repetition_pass_at_1"] == [0.0, 0.5, 1.0]
        assert scores["mean_pass_at_1"] == scores["std_pass_at_1"] == 0.5
        assert scores["task_attempts"] == (120 if experiment == "tb2" else 60)
    kwargs = factory.call_args.kwargs
    contract = comparison["shared_configuration"]
    assert kwargs["student_model"] == model
    assert kwargs["student_api_base"] == contract["student_api_base"]
    assert kwargs["student_agent_kwargs"]["model_info"] == contract["student_model_info"]
    assert kwargs["student_agent_kwargs"]["llm_kwargs"] == {
        "num_retries": contract["student_num_retries"],
        **contract["student_decoding"],
        **contract["student_request_overrides"],
    }
    evaluate.main()
    assert runner.run.call_count == 21


@pytest.mark.parametrize(
    "specifications",
    [
        ["vanilla"],
        ["vanilla="],
        ["unknown=/tmp/run"],
        ["vanilla=/tmp/first", "vanilla=/tmp/second"],
    ],
)
def test_evaluation_cli_rejects_invalid_or_duplicate_cell_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, specifications: list[str]
) -> None:
    """Reject ambiguous source mappings before reading runs or constructing Harbor."""
    freeze = Mock()
    harbor = Mock()
    monkeypatch.setattr(evaluate, "freeze_comparison", freeze)
    monkeypatch.setattr(evaluate, "HarborCLI", harbor)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            *[argument for specification in specifications for argument in ("--run-dir", specification)],
            "--output-dir",
            str(tmp_path / "test"),
        ],
    )
    with pytest.raises(SystemExit):
        evaluate.main()
    freeze.assert_not_called()
    harbor.assert_not_called()


def test_interrupted_evaluation_resumes_only_missing_repetitions(tmp_path: Path) -> None:
    """Keep completed attempts unchanged and withhold a summary until all 21 jobs finish."""
    run_dirs = _write_comparison(tmp_path, "tb2")
    manifest, comparison = evaluate.freeze_comparison(run_dirs)
    output_dir = tmp_path / "test"
    runner = _fake_runner(manifest, comparison, output_dir, fail_on_call=5)

    with pytest.raises(HarborExecutionError, match="interrupted"):
        evaluate.evaluate_comparison(manifest, comparison, output_dir, runner)
    saved = {path.name: path.read_bytes() for path in output_dir.glob("*-repetition-*.json")}
    assert len(saved) == 4
    assert not (output_dir / "summary.json").exists()
    summary = evaluate.evaluate_comparison(manifest, comparison, output_dir, runner)
    assert runner.run.call_count == 22
    assert summary["complete"] is True
    assert all((output_dir / name).read_bytes() == content for name, content in saved.items())
    assert all(row["mean_pass_at_1"] == 0.5 for row in summary["harnesses"].values())


def test_unchanged_winners_still_receive_separate_test_repetitions(tmp_path: Path) -> None:
    """Preserve fresh attempts when validation selects the initial harness in all six runs."""
    run_dirs = _write_comparison(tmp_path, "tb4")
    for run_dir in run_dirs.values():
        state = GEPAState.load(str(run_dir))
        state.prog_candidate_val_subscores[1] = dict.fromkeys(state.prog_candidate_val_subscores[1], 0.0)
        state.save(str(run_dir))
    manifest, comparison = evaluate.freeze_comparison(run_dirs)
    assert all(run["selected_candidate_index"] == 0 for run in comparison["source_runs"].values())
    assert len({harness["candidate_digest"] for harness in comparison["harnesses"].values()}) == 1
    output_dir = tmp_path / "test"
    runner = _fake_runner(manifest, comparison, output_dir)
    summary = evaluate.evaluate_comparison(manifest, comparison, output_dir, runner)
    assert runner.run.call_count == 21
    assert len(list(output_dir.glob("*-repetition-*.json"))) == 21
    assert all(row["task_attempts"] == 60 for row in summary["harnesses"].values())


@pytest.mark.parametrize(
    "damage",
    [
        "incomplete",
        "partial_validation",
        "partial_train",
        "different_seed",
        "different_model",
        "different_selector",
        "missing_cell",
        "wrong_budget",
        "incomplete_double",
        "different_condition",
        "different_policy",
        "different_feedback",
        "different_failure_policy",
        "different_context_policy",
        "manifestor_trace_cap",
        "manifestor_temperature",
        "controller_top_p",
        "manifestor_top_p",
        "react_v2_proposer_top_p",
        "missing_failure_policy",
    ],
)
def test_invalid_source_runs_are_rejected_before_test_execution(tmp_path: Path, damage: str) -> None:
    """Reject unfinished optimization, pilot splits, and unmatched experimental settings."""
    run_dirs = _write_comparison(tmp_path, "tb4")
    forest = run_dirs["react_v2"]
    if damage == "different_model":
        forest = _write_run(tmp_path / "other-model", "tb4", "react_v2", DEEPSEEK_V4_FLASH_MODEL)
        run_dirs["react_v2"] = forest
    if damage == "missing_cell":
        run_dirs.pop("action")
    elif damage == "wrong_budget":
        run_dirs["react_v2_2x"] = forest
    elif damage == "incomplete_double":
        state = GEPAState.load(str(run_dirs["react_v2_2x"]))
        state.i = 31
        state.save(str(run_dirs["react_v2_2x"]))
    if damage in {"incomplete", "partial_validation"}:
        state = GEPAState.load(str(forest))
        if damage == "incomplete":
            state.i -= 1
        else:
            del state.prog_candidate_val_subscores[1][0]
        state.save(str(forest))
    elif damage in {
        "partial_train",
        "different_seed",
        "different_selector",
        "different_condition",
        "different_policy",
        "different_feedback",
        "different_failure_policy",
        "different_context_policy",
        "manifestor_trace_cap",
        "manifestor_temperature",
        "controller_top_p",
        "manifestor_top_p",
        "react_v2_proposer_top_p",
        "missing_failure_policy",
    }:
        path = forest / RUN_CONTRACT_FILENAME
        contract = json.loads(path.read_text())
        if damage == "partial_train":
            contract["train_task_ids"].pop()
        elif damage == "different_selector":
            contract["module_selector"] = "round_robin"
        elif damage == "different_condition":
            contract["condition"] = "action"
        elif damage == "different_policy":
            contract["controller_selection"] = "uniform_random"
        elif damage == "different_feedback":
            contract["reflection_feedback"]["max_bytes_per_verifier_log"] = 2048
        elif damage == "different_failure_policy":
            contract["failure_policy"]["harbor_max_retries"] = 1
        elif damage == "different_context_policy":
            contract["reflection_context"]["minimum_reference_chars"] = 8000
        elif damage == "manifestor_trace_cap":
            contract["manifestor_traces_chars"] = 8000
        elif damage == "manifestor_temperature":
            contract["manifestor_temperature"] = 0.0
        elif damage.endswith("_top_p"):
            contract["reflection_role_decoding"][damage.removesuffix("_top_p")]["requested"]["top_p"] = 0.5
        elif damage == "missing_failure_policy":
            del contract["failure_policy"]
        else:
            contract["seed"] = 19
        path.write_text(json.dumps(contract))
    with pytest.raises(ValueError):
        evaluate.freeze_comparison(run_dirs)


def test_frozen_output_rejects_a_changed_validation_winner(tmp_path: Path) -> None:
    """Prevent replacing an optimized harness after test feedback has been observed."""
    run_dirs = _write_comparison(tmp_path, "tb4")
    forest = run_dirs["react_v2_2x"]
    manifest, comparison = evaluate.freeze_comparison(run_dirs)
    output_dir = tmp_path / "test"
    runner = _fake_runner(manifest, comparison, output_dir, fail_on_call=2)
    with pytest.raises(HarborExecutionError):
        evaluate.evaluate_comparison(manifest, comparison, output_dir, runner)
    state = GEPAState.load(str(forest))
    state.program_candidates[1]["instruction_prompt"] += "\nChanged after testing"
    state.save(str(forest))
    _, changed = evaluate.freeze_comparison(run_dirs)
    with pytest.raises(ValueError, match="different frozen comparison"):
        evaluate.evaluate_comparison(manifest, changed, output_dir, runner)
    assert runner.run.call_count == 2


@pytest.mark.parametrize(
    "damage", ["missing_task", "nonbinary", "wrong_candidate", "wrong_repetition", "duplicate_job"]
)
def test_corrupt_saved_repetition_is_not_silently_reused(tmp_path: Path, damage: str) -> None:
    """Reject incomplete, mismatched, or duplicate test evidence during resume."""
    run_dirs = _write_comparison(tmp_path, "tb4")
    manifest, comparison = evaluate.freeze_comparison(run_dirs)
    output_dir = tmp_path / "test"
    runner = _fake_runner(manifest, comparison, output_dir)
    evaluate.evaluate_comparison(manifest, comparison, output_dir, runner)
    path = output_dir / "initial-repetition-1.json"
    record = json.loads(path.read_text())
    if damage == "missing_task":
        record["scores"].pop(manifest.splits["test"][0])
    elif damage == "nonbinary":
        record["scores"][manifest.splits["test"][0]] = 0.5
    elif damage == "wrong_candidate":
        record["candidate_digest"] = "another-candidate"
    elif damage == "wrong_repetition":
        record["repetition"] = 3
    else:
        other = json.loads((output_dir / "initial-repetition-2.json").read_text())
        record["evaluation_id"] = other["evaluation_id"]
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        evaluate.evaluate_comparison(manifest, comparison, output_dir, runner)
    assert runner.run.call_count == 21
