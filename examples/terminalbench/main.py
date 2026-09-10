"""Configure separate Terminal-Bench 2 prompt and Terminal-Bench 4 text experiments.

The held-out test split is not evaluated automatically.

* ``vanilla`` uses stock free-form GEPA reflection.
* ``react_v2`` uses the Controller -> Manifestor -> ReAct V2 workflow.

Within each model arm, all conditions use the same official Harbor rewards,
manifest, student/proposer model, task splits, editable documents, and metric-call
budget. The experiment must be selected explicitly; neither is primary.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, cast

from examples.common.experiment_models import (
    EXPERIMENT_NUM_RETRIES,
    QWEN3_8_27B_MODEL,
    QWEN3_8_27B_MODEL_INFO,
    experiment_decoding,
    validate_experiment_model_pair,
)
from examples.common.react_v2 import resolve_template_family
from gepa import optimize
from gepa.adapters.terminal_bench_adapter import (
    HarborCLI,
    TerminalBenchAdapter,
    TerminalBenchManifest,
    TerminalBenchTask,
    load_terminalbench_manifest,
)
from gepa.adapters.terminal_bench_adapter.documents import (
    BUNDLE_VERSION,
    seed_documents,
)
from gepa.strategies.document_template import TEMPLATE_FAMILIES
from gepa.strategies.intervention import CONTROLLER_POLICY_CONTRACT, SEMANTIC_ACTION_CATALOGS

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_MANIFESTS = {
    "tb2-system-prompt": Path(__file__).with_name("terminalbench-v2-manifest.json"),
    "tb4-agent-text": Path(__file__).with_name("terminalbench-v4-manifest.json"),
}
SYSTEM_PROMPT_SEED_PATH = Path(__file__).with_name("terminus-system-prompt.txt")

RUN_CONTRACT_FILENAME = "terminalbench-run-contract.json"
TemplateFamily = Literal["generic", "openai", "anthropic", "google", "alibaba"]


def seed_candidate(student_model: str, template_family: str, experiment: str) -> tuple[dict[str, str], TemplateFamily]:
    """Build the experiment's seed with the selected provider template.

    Args:
        student_model: Task model used for automatic provider inference.
        template_family: Explicit provider family or ``"auto"``.
        experiment: Single system prompt or the full agent text and skills.

    Returns:
        Editable components and their resolved template family.
    """
    resolved_family = cast(TemplateFamily, resolve_template_family(template_family, student_model))
    if experiment == "tb2-system-prompt":
        template = TEMPLATE_FAMILIES[resolved_family]["system_prompt"]
        section = {
            "generic": "Task",
            "openai": "Instructions",
            "anthropic": "Instructions",
            "google": "Instructions",
            "alibaba": "Objective",
        }[resolved_family]
        return {
            "system_prompt": template.render({section: SYSTEM_PROMPT_SEED_PATH.read_text(encoding="utf-8")})
        }, resolved_family
    if experiment != "tb4-agent-text":
        raise ValueError(f"Unknown Terminal-Bench experiment: {experiment!r}")
    return seed_documents(resolved_family), resolved_family


def ensure_run_contract(run_dir: Path, contract: dict[str, Any]) -> Path:
    """Write the run contract or reject an incompatible resumable directory.

    Args:
        run_dir: Experiment directory that owns the resumable state.
        contract: Complete material configuration for the requested run.

    Returns:
        Path to the existing or newly written contract file.

    Raises:
        ValueError: Existing state has a different contract, or legacy GEPA
            state has no contract to validate.
    """
    path = run_dir / RUN_CONTRACT_FILENAME
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != contract:
            raise ValueError(f"Run directory {run_dir} contains a different Terminal-Bench configuration.")
        return path
    if (run_dir / "gepa_state.bin").exists():
        raise ValueError(
            f"Run directory {run_dir} has GEPA state but no {RUN_CONTRACT_FILENAME}; choose a clean directory."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    return path


def build_parser() -> argparse.ArgumentParser:
    """Build the experiment CLI without launching any evaluation.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="GEPA on pinned Terminal-Bench 2 or 4 through Harbor")
    parser.add_argument(
        "--experiment",
        choices=tuple(EXPERIMENT_MANIFESTS),
        required=True,
        help="TB2: one unified system prompt; TB4: all 13 prompts and two skills",
    )
    parser.add_argument(
        "--condition",
        choices=("vanilla", "react_v2", "action"),
        required=True,
        help="Optimization condition to run",
    )
    parser.add_argument(
        "--student-model",
        default=QWEN3_8_27B_MODEL,
        help="Terminus model; use the same supported model as --proposer-model",
    )
    parser.add_argument(
        "--proposer-model",
        default=QWEN3_8_27B_MODEL,
        help="GEPA proposer; use the same supported model as --student-model",
    )
    parser.add_argument("--student-api-base", default=None)
    parser.add_argument("--proposer-api-base", default=None)
    parser.add_argument("--max-metric-calls", type=int, required=True)
    parser.add_argument("--reflection-minibatch-size", type=int, default=3)
    parser.add_argument("--n-concurrent", type=int, default=1)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--edit-tool-set",
        choices=("minimal", "broad"),
        default="broad",
        help="Edit tools used by ReAct V2",
    )
    parser.add_argument(
        "--reflection-level",
        type=int,
        choices=(1, 2),
        default=2,
        help="Reflection level: region only, or region plus an applied semantic action",
    )
    parser.add_argument(
        "--template-family",
        choices=("auto", "generic", "openai", "anthropic", "google", "alibaba"),
        default="auto",
    )
    parser.add_argument("--manifest", type=Path, default=None, help="Optional manifest path; must match --experiment")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--harbor-work-dir", type=Path, required=True)
    parser.add_argument("--harbor-executable", default="harbor")
    parser.add_argument("--docker-executable", default="docker")
    parser.add_argument(
        "--harbor-process-timeout-sec",
        type=float,
        default=None,
        help="Optional whole-job timeout; default leaves long-horizon runs to task-level Harbor timeouts",
    )
    return parser


def build_run_contract(
    args: argparse.Namespace,
    manifest: TerminalBenchManifest,
    trainset: list[TerminalBenchTask],
    valset: list[TerminalBenchTask],
    condition: str,
    resolved_family: str,
) -> dict[str, Any]:
    """Record every material axis needed for safe resume and comparison.

    Args:
        args: Parsed Terminal-Bench CLI arguments.
        manifest: Validated pinned benchmark manifest.
        trainset: Selected training tasks in manifest order.
        valset: Selected validation tasks in manifest order.
        condition: Canonical optimization condition.
        resolved_family: Provider template family used by the student prompt.

    Returns:
        JSON-serializable run contract including exact task identities.
    """
    validate_experiment_model_pair(args.student_model, args.proposer_model)
    if manifest.experiment != args.experiment:
        raise ValueError("--manifest must match the selected --experiment")
    candidate, _ = seed_candidate(args.student_model, resolved_family, args.experiment)
    operated = condition == "react_v2"
    reflection_level = args.reflection_level if operated else 0
    return {
        "schema_version": 6,
        "experiment": manifest.experiment,
        "optimization_target": "system_prompt" if manifest.experiment == "tb2-system-prompt" else "agent_text",
        "condition": condition,
        "component_kinds": manifest.component_kinds,
        "document_bundle_version": BUNDLE_VERSION if manifest.experiment == "tb4-agent-text" else None,
        "seed_document_digest": manifest.candidate_digest(candidate),
        "dataset": manifest.dataset,
        "split_policy": manifest.split_policy,
        "task_refs": manifest.task_refs,
        "test_task_ids": [task.task_id for task in manifest.tasks("test")],
        "edit_tool_set": args.edit_tool_set,
        "harbor_process_timeout_sec": args.harbor_process_timeout_sec,
        "manifest": str(manifest.path),
        "max_metric_calls": args.max_metric_calls,
        "n_concurrent": args.n_concurrent,
        "proposer_api_base": args.proposer_api_base,
        "proposer_backend": "react_v2" if operated else "stateless",
        "proposer_decoding": experiment_decoding(args.proposer_model),
        "proposer_model": args.proposer_model,
        "proposer_num_retries": EXPERIMENT_NUM_RETRIES,
        "reflection_level": reflection_level,
        "reflection_minibatch_size": args.reflection_minibatch_size,
        "max_proposer_model_calls": 8 if operated else None,
        "semantic_action_space": deepcopy(SEMANTIC_ACTION_CATALOGS) if reflection_level == 2 else None,
        "semantic_controller_policy": deepcopy(CONTROLLER_POLICY_CONTRACT) if reflection_level == 2 else None,
        "seed": args.seed,
        "student_api_base": args.student_api_base,
        "student_decoding": experiment_decoding(args.student_model),
        "student_model": args.student_model,
        "student_model_info": dict(QWEN3_8_27B_MODEL_INFO) if args.student_model == QWEN3_8_27B_MODEL else None,
        "student_num_retries": EXPERIMENT_NUM_RETRIES,
        "template_family": resolved_family,
        "train_task_ids": [task.task_id for task in trainset],
        "val_task_ids": [task.task_id for task in valset],
    }


def main() -> None:
    """Validate the pinned harness and start the requested GEPA condition.

    Raises:
        ValueError: Training or validation selection is empty.
    """
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_experiment_model_pair(args.student_model, args.proposer_model)
    except ValueError as exc:
        parser.error(str(exc))
    manifest_path = args.manifest or EXPERIMENT_MANIFESTS[args.experiment]
    manifest = load_terminalbench_manifest(manifest_path)
    if manifest.experiment != args.experiment:
        parser.error("--manifest must match the selected --experiment")
    trainset = manifest.tasks("train", args.train_limit)
    valset = manifest.tasks("val", args.val_limit)
    if not trainset or not valset:
        raise ValueError("train and validation selections must both be non-empty")

    candidate, resolved_family = seed_candidate(args.student_model, args.template_family, args.experiment)
    condition = "react_v2" if args.condition == "action" else args.condition
    contract = build_run_contract(args, manifest, trainset, valset, condition, resolved_family)
    ensure_run_contract(args.run_dir, contract)

    student_agent_kwargs: dict[str, Any] = {
        "llm_kwargs": {"num_retries": EXPERIMENT_NUM_RETRIES, **experiment_decoding(args.student_model)}
    }
    if args.student_model == QWEN3_8_27B_MODEL:
        student_agent_kwargs["model_info"] = dict(QWEN3_8_27B_MODEL_INFO)

    harbor = HarborCLI(
        manifest=manifest,
        student_model=args.student_model,
        student_api_base=args.student_api_base,
        work_dir=args.harbor_work_dir,
        agent_python_path=REPO_ROOT,
        n_concurrent=args.n_concurrent,
        harbor_executable=args.harbor_executable,
        docker_executable=args.docker_executable,
        student_agent_kwargs=student_agent_kwargs,
        process_timeout_sec=args.harbor_process_timeout_sec,
    )
    harbor.check_requirements()
    adapter = TerminalBenchAdapter(manifest, harbor)

    reflection_lm_kwargs: dict[str, Any] = {
        "num_retries": EXPERIMENT_NUM_RETRIES,
        **experiment_decoding(args.proposer_model),
    }
    if args.proposer_api_base is not None:
        reflection_lm_kwargs["api_base"] = args.proposer_api_base

    reflection_level = 0 if condition == "vanilla" else args.reflection_level
    optimize(
        seed_candidate=candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=args.proposer_model,
        reflection_lm_kwargs=reflection_lm_kwargs,
        max_metric_calls=args.max_metric_calls,
        reflection_minibatch_size=args.reflection_minibatch_size,
        run_dir=str(args.run_dir),
        seed=args.seed,
        reflection_level=reflection_level,
        edit_tool_set=args.edit_tool_set,
        component_kinds=manifest.component_kinds,
        template_family=resolved_family,
        template_model=args.student_model,
    )


if __name__ == "__main__":
    main()
