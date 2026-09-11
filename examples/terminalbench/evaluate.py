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
    CAMPAIGN_CELLS,
    EVALUATION_PROTOCOL,
    EXPERIMENT_MANIFESTS,
    FOREST_CONDITIONS,
    REPO_ROOT,
    RUN_CONTRACT_FILENAME,
    TEST_REPETITIONS,
    build_run_contract,
)
from gepa.adapters.terminal_bench_adapter import (
    HarborCLI,
    TerminalBenchManifest,
    TerminusAdapter,
    load_terminalbench_manifest,
)
from gepa.core.result import GEPAResult
from gepa.core.state import GEPAState
from gepa.strategies.text_limits import resolve_text_limits

FROZEN_COMPARISON_FILENAME = "frozen-comparison.json"
METHOD_SPECIFIC_FIELDS = {
    "condition",
    "budget",
    "optimization_budget",
    "controller_selection",
    "proposer_backend",
    "reflection_level",
    "reflection_role_decoding",
    "max_proposer_model_calls",
    "react_execution",
    "semantic_action_space",
    "semantic_controller_policy",
    "stateless_selector_policy",
    "manifest",
}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically replace an artifact so interruption cannot leave partial JSON."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def load_completed_run(run_dir: Path, condition: str, budget: str) -> tuple[TerminalBenchManifest, dict[str, Any]]:
    """Select the validation winner from one completed, trusted local checkpoint.

    Args:
        run_dir: Directory produced by the Terminal-Bench optimization CLI.
        condition: Required method from the six-configuration campaign.
        budget: Required standard or double training budget for this cell.

    Returns:
        Pinned manifest and the initial/selected harnesses with run provenance.

    Raises:
        ValueError: The run is incomplete, uses a partial split, or has drifted
            from its recorded protocol, model settings, or seed candidate.
    """
    contract = json.loads((run_dir / RUN_CONTRACT_FILENAME).read_text())
    if contract.get("budget") != budget:
        raise ValueError(f"{run_dir}: expected the {budget} budget for {condition}")
    if contract.get("reflection_level") != (2 if condition in FOREST_CONDITIONS else 0):
        raise ValueError(f"{run_dir}: reflection level does not match the campaign method {condition}")
    if contract.get("experiment") not in EXPERIMENT_MANIFESTS:
        raise ValueError(f"{run_dir}: only Terminal-Bench 2.1 runs may enter final comparison")
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
        epochs = contract["optimization_budget"]["training_epochs"]
        raise ValueError(f"{run_dir}: optimization has not completed its {epochs}-epoch budget")
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


def freeze_comparison(run_dirs: dict[str, Path]) -> tuple[TerminalBenchManifest, dict[str, Any]]:
    """Freeze all six validation winners before any test result can influence selection.

    Args:
        run_dirs: One completed directory per campaign cell for one benchmark/model.

    Returns:
        Manifest and a comparison containing all seven immutable harness texts.

    Raises:
        ValueError: A campaign cell is missing or shared experimental settings differ.
    """
    if set(run_dirs) != set(CAMPAIGN_CELLS):
        raise ValueError(f"Final testing requires exactly these six campaign cells: {', '.join(CAMPAIGN_CELLS)}")
    runs = {}
    for label, (condition, budget) in CAMPAIGN_CELLS.items():
        manifest, runs[label] = load_completed_run(run_dirs[label], condition, budget)
    vanilla = runs["vanilla"]
    manifest = load_terminalbench_manifest(EXPERIMENT_MANIFESTS[vanilla["contract"]["experiment"]])
    shared = {key: value for key, value in vanilla["contract"].items() if key not in METHOD_SPECIFIC_FIELDS}
    for label, run in runs.items():
        other = {key: value for key, value in run["contract"].items() if key not in METHOD_SPECIFIC_FIELDS}
        if shared != other or vanilla["initial"] != run["initial"]:
            raise ValueError(f"{label}: all six runs must share benchmark, model settings, seed, and splits")
    candidates = {"initial": vanilla["initial"], **{label: run["selected"] for label, run in runs.items()}}
    return manifest, {
        "schema_version": 2,
        "protocol": dict(EVALUATION_PROTOCOL),
        "shared_configuration": shared,
        "source_runs": {
            label: {key: value for key, value in run.items() if key not in {"initial", "selected"}}
            for label, run in runs.items()
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
        manifest: Pinned benchmark shared by all six optimization runs.
        comparison: Frozen initial harness and six validation-selected winners.
        output_dir: Dedicated comparison directory; use one writer at a time.
        harbor: Runner with the recorded student model and runtime settings.

    Returns:
        Mean and sample standard deviation over three complete test repetitions.

    Raises:
        ValueError: Frozen identity changed or saved test results are invalid.
    """
    adapter = TerminusAdapter(manifest, harbor)
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
        batch = adapter.evaluate(manifest.tasks("test"), comparison["harnesses"][label]["documents"])
        job = batch.outputs[0]
        record = {
            **identity,
            "candidate_digest": job["candidate_digest"],
            "evaluation_id": job["evaluation_id"],
            "job_dir": job["job_dir"],
            "config_path": job["config_path"],
            "scores": {output["task_id"]: score for output, score in zip(batch.outputs, batch.scores, strict=True)},
        }
        _validate_repetition(record, identity, task_ids)
        if record["evaluation_id"] in seen_evaluations:
            raise ValueError("Each test repetition must use a distinct Harbor evaluation")
        seen_evaluations.add(record["evaluation_id"])
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
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        metavar="CELL=PATH",
        help=f"Repeat once for each of: {', '.join(CAMPAIGN_CELLS)}",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--harbor-executable", default="harbor")
    parser.add_argument("--docker-executable", default="docker")
    args = parser.parse_args()
    run_dirs = {}
    for specification in args.run_dir:
        label, separator, path = specification.partition("=")
        if not separator or label not in CAMPAIGN_CELLS or not path or label in run_dirs:
            parser.error("Each --run-dir must specify a distinct supported CELL=PATH")
        run_dirs[label] = Path(path)
    manifest, comparison = freeze_comparison(run_dirs)
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
        text_limits=resolve_text_limits(contract["text_limits"]),
        student_agent_kwargs={
            "token_limits": contract["token_limits"],
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
