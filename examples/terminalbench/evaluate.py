"""Freeze completed Terminal-Bench comparisons and repeat held-out Pass@1 tests."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import statistics
from pathlib import Path
from typing import Any

from examples.terminalbench.main import (
    EVALUATION_PROTOCOL,
    EXPERIMENT_MANIFESTS,
    REPO_ROOT,
    RUN_CONTRACT_FILENAME,
    TEST_REPETITIONS,
    build_run_contract,
)
from gepa.adapters.terminal_bench_adapter import HarborCLI, TerminalBenchManifest, load_terminalbench_manifest
from gepa.core.result import GEPAResult
from gepa.core.state import GEPAState

FROZEN_COMPARISON_FILENAME = "frozen-comparison.json"
METHOD_SPECIFIC_FIELDS = {
    "condition",
    "proposer_backend",
    "reflection_level",
    "max_proposer_model_calls",
    "semantic_action_space",
    "semantic_controller_policy",
    "manifest",
}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically replace an artifact so interruption cannot leave partial JSON."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def load_completed_run(run_dir: Path, condition: str) -> tuple[TerminalBenchManifest, dict[str, Any]]:
    """Select the validation winner from one completed, trusted local checkpoint.

    Args:
        run_dir: Directory produced by the Terminal-Bench optimization CLI.
        condition: Required method, either ``vanilla`` or ``react_v2``.

    Returns:
        Pinned manifest and the initial/selected harnesses with run provenance.

    Raises:
        ValueError: The run is incomplete, uses a partial split, or has drifted
            from its recorded protocol, model settings, or seed candidate.
    """
    contract = json.loads((run_dir / RUN_CONTRACT_FILENAME).read_text())
    manifest = load_terminalbench_manifest(EXPERIMENT_MANIFESTS[contract["experiment"]])
    expected = build_run_contract(
        argparse.Namespace(**contract),
        manifest,
        manifest.tasks("train"),
        manifest.tasks("val"),
        condition,
        contract["template_family"],
    )
    if {**contract, "manifest": str(manifest.path)} != expected:
        raise ValueError(
            f"{run_dir}: expected a matching {condition} run on the complete training and validation splits"
        )
    state = GEPAState.load(str(run_dir))
    completed_iterations = state.i + 1
    if completed_iterations != contract["optimization_budget"]["max_iterations"]:
        raise ValueError(f"{run_dir}: optimization has not completed its four-epoch budget")
    result = GEPAResult.from_state(state)
    if manifest.candidate_digest(result.candidates[0]) != contract["seed_document_digest"]:
        raise ValueError(f"{run_dir}: checkpoint seed differs from the run contract")
    expected_val_ids = set(range(len(manifest.splits["val"])))
    if any(set(scores) != expected_val_ids for scores in result.val_subscores):
        raise ValueError(f"{run_dir}: candidate validation coverage is incomplete")
    if not all(math.isfinite(score) for score in result.val_aggregate_scores):
        raise ValueError(f"{run_dir}: validation scores must be finite")
    selected = result.candidates[result.best_idx]
    manifest.validate_candidate(selected)
    return manifest, {
        "run_dir": str(run_dir.resolve()),
        "contract": contract,
        "completed_iterations": completed_iterations,
        "optimization_metric_calls": result.total_metric_calls,
        "selected_candidate_index": result.best_idx,
        "validation_scores": result.val_aggregate_scores,
        "initial": result.candidates[0],
        "selected": selected,
    }


def freeze_comparison(vanilla_run_dir: Path, forest_run_dir: Path) -> tuple[TerminalBenchManifest, dict[str, Any]]:
    """Freeze both validation winners before any test result can influence selection.

    Args:
        vanilla_run_dir: Completed vanilla GEPA run for one benchmark/model arm.
        forest_run_dir: Matching completed FOREST run.

    Returns:
        Manifest and a comparison containing all three immutable harness texts.

    Raises:
        ValueError: Methods differ on a shared experimental setting.
    """
    manifest, vanilla = load_completed_run(vanilla_run_dir, "vanilla")
    _, forest = load_completed_run(forest_run_dir, "react_v2")
    shared = {key: value for key, value in vanilla["contract"].items() if key not in METHOD_SPECIFIC_FIELDS}
    other = {key: value for key, value in forest["contract"].items() if key not in METHOD_SPECIFIC_FIELDS}
    if shared != other or vanilla["initial"] != forest["initial"]:
        raise ValueError("GEPA and FOREST must share benchmark, model settings, seed, splits, and training budget")
    candidates = {"initial": vanilla["initial"], "vanilla": vanilla["selected"], "react_v2": forest["selected"]}
    return manifest, {
        "schema_version": 1,
        "protocol": dict(EVALUATION_PROTOCOL),
        "shared_configuration": shared,
        "source_runs": {
            label: {key: value for key, value in run.items() if key not in {"initial", "selected"}}
            for label, run in (("vanilla", vanilla), ("react_v2", forest))
        },
        "harnesses": {
            label: {"documents": candidate, "candidate_digest": manifest.candidate_digest(candidate)}
            for label, candidate in candidates.items()
        },
    }


def _validate_repetition(record: dict[str, Any], identity: dict[str, Any], task_ids: list[str]) -> None:
    """Reject reused or incomplete results instead of silently altering Pass@1."""
    if any(record.get(key) != value for key, value in identity.items()):
        raise ValueError("Test repetition does not match its frozen harness and repetition index")
    scores = record.get("scores")
    if not isinstance(scores, dict) or set(scores) != set(task_ids):
        raise ValueError("Test repetition must contain every held-out task exactly once")
    if any(score not in (0.0, 1.0) for score in scores.values()):
        raise ValueError("Pass@1 requires binary official verifier rewards")
    if not isinstance(record.get("evaluation_id"), str) or not record["evaluation_id"]:
        raise ValueError("Test repetition has no Harbor evaluation identity")


def evaluate_comparison(
    manifest: TerminalBenchManifest, comparison: dict[str, Any], output_dir: Path, harbor: HarborCLI
) -> dict[str, Any]:
    """Resume three fresh test repetitions per frozen harness and summarize Pass@1.

    Args:
        manifest: Pinned benchmark shared by both optimization runs.
        comparison: Frozen initial harness and both validation-selected winners.
        output_dir: Dedicated comparison directory; use one writer at a time.
        harbor: Runner with the recorded student model and runtime settings.

    Returns:
        Mean and sample standard deviation over three complete test repetitions.

    Raises:
        ValueError: Frozen identity changed or saved test results are invalid.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = output_dir / FROZEN_COMPARISON_FILENAME
    if frozen_path.exists():
        if json.loads(frozen_path.read_text()) != comparison:
            raise ValueError("This test directory belongs to a different frozen comparison")
    else:
        if any(output_dir.glob("*-repetition-*.json")) or (output_dir / "summary.json").exists():
            raise ValueError("Existing test results have no frozen comparison")
        _write_json(frozen_path, comparison)

    task_ids = manifest.splits["test"]
    records: dict[tuple[str, int], dict[str, Any]] = {}
    identities = {
        (label, repetition): {
            "harness": label,
            "repetition": repetition,
            "candidate_digest": harness["candidate_digest"],
        }
        for repetition in range(1, TEST_REPETITIONS + 1)
        for label, harness in comparison["harnesses"].items()
    }
    seen_evaluations: set[str] = set()
    for (label, repetition), identity in identities.items():
        path = output_dir / f"{label}-repetition-{repetition}.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        _validate_repetition(record, identity, task_ids)
        if record["evaluation_id"] in seen_evaluations:
            raise ValueError("Each test repetition must use a distinct Harbor evaluation")
        seen_evaluations.add(record["evaluation_id"])
        records[label, repetition] = record

    for (label, repetition), identity in identities.items():
        if (label, repetition) in records:
            continue
        print(f"Testing {label}, repetition {repetition}/{TEST_REPETITIONS} ({len(task_ids)} tasks)", flush=True)
        evaluation = harbor.run(task_ids, comparison["harnesses"][label]["documents"])
        record = {
            **identity,
            "candidate_digest": evaluation.candidate_digest,
            "evaluation_id": evaluation.evaluation_id,
            "job_dir": str(evaluation.job_dir),
            "config_path": str(evaluation.config_path),
            "scores": {task_id: trial.reward for task_id, trial in evaluation.trials.items()},
        }
        _validate_repetition(record, identity, task_ids)
        if evaluation.evaluation_id in seen_evaluations:
            raise ValueError("Each test repetition must use a distinct Harbor evaluation")
        seen_evaluations.add(evaluation.evaluation_id)
        _write_json(output_dir / f"{label}-repetition-{repetition}.json", record)
        records[label, repetition] = record

    summary: dict[str, Any] = {
        "complete": True,
        "experiment": manifest.experiment,
        "student_model": comparison["shared_configuration"]["student_model"],
        "protocol": dict(EVALUATION_PROTOCOL),
        "test_task_count": len(task_ids),
        "score_units": "fraction",
        "harnesses": {},
    }
    for label, harness in comparison["harnesses"].items():
        scores = [
            statistics.mean(records[label, repetition]["scores"].values())
            for repetition in range(1, TEST_REPETITIONS + 1)
        ]
        summary["harnesses"][label] = {
            "candidate_digest": harness["candidate_digest"],
            "repetition_pass_at_1": scores,
            "mean_pass_at_1": statistics.mean(scores),
            "std_pass_at_1": statistics.stdev(scores),
            "task_attempts": len(task_ids) * TEST_REPETITIONS,
        }
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    """Evaluate one matched benchmark/model comparison from completed local runs."""
    parser = argparse.ArgumentParser(description="Three frozen Terminal-Bench Pass@1 test repetitions")
    parser.add_argument("--vanilla-run-dir", type=Path, required=True)
    parser.add_argument("--forest-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--harbor-executable", default="harbor")
    parser.add_argument("--docker-executable", default="docker")
    args = parser.parse_args()
    manifest, comparison = freeze_comparison(args.vanilla_run_dir, args.forest_run_dir)
    contract = comparison["shared_configuration"]
    harbor = HarborCLI(
        manifest=manifest,
        student_model=contract["student_model"],
        student_api_base=contract["student_api_base"],
        work_dir=args.output_dir / "harbor",
        agent_python_path=REPO_ROOT,
        n_concurrent=contract["n_concurrent"],
        harbor_executable=args.harbor_executable,
        docker_executable=args.docker_executable,
        process_timeout_sec=contract["harbor_process_timeout_sec"],
        student_agent_kwargs={
            "model_info": contract["student_model_info"],
            "llm_kwargs": {
                "num_retries": contract["student_num_retries"],
                **contract["student_decoding"],
                **contract["student_request_overrides"],
            },
        },
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / ".evaluation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another evaluation is already writing to this output directory")
        summary = evaluate_comparison(manifest, comparison, args.output_dir, harbor)
    for label, scores in summary["harnesses"].items():
        print(f"{label}: Pass@1 {scores['mean_pass_at_1']:.2%} +/- {scores['std_pass_at_1']:.2%}")
    print(f"Saved {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
