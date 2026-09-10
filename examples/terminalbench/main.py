"""Configure full agent-text experiments on Terminal-Bench 2 and 4.

The held-out test split is not evaluated automatically.

* ``vanilla`` uses stock free-form GEPA reflection.
* ``react_v2`` uses the Controller -> Manifestor -> ReAct V2 workflow.
* ``react_v2_random`` replaces only the Controller with uniform selection.
* ``action`` uses semantic action selection and a stateless section rewrite.

Within each model arm, all conditions use the same official Harbor rewards,
manifest, student/proposer model, task splits, and editable documents. All four
methods run for four epochs; vanilla and full FOREST also run for eight epochs.
The experiment must be selected explicitly; neither is primary.
"""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, cast

from examples.common.experiment_models import (
    EXPERIMENT_NUM_RETRIES,
    QWEN3_8_27B_MODEL,
    experiment_decoding,
    experiment_model_info,
    experiment_model_version,
    experiment_request_overrides,
    validate_experiment_model_pair,
)
from examples.common.react_v2 import build_react_v2_strategy, resolve_template_family
from examples.terminalbench.reflection import ComponentActionReflectionLM
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
from gepa.adapters.terminal_bench_adapter.terminal_bench_adapter import (
    FAILURE_POLICY_CONTRACT,
    REFLECTION_FEEDBACK_CONTRACT,
)
from gepa.lm import LM
from gepa.strategies.action_space import stateless_selector_policy_contract
from gepa.strategies.intervention import (
    CONTROLLER_POLICY_CONTRACT,
    SEMANTIC_ACTION_CATALOGS,
    UNIFORM_RANDOM_CONTROLLER_POLICY_CONTRACT,
)
from gepa.strategies.proposal_sampling import SingleMutationSampling
from gepa.utils.stop_condition import MaxCandidateProposalsStopper

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_MANIFESTS = {
    "tb2": Path(__file__).with_name("terminalbench-v2-manifest.json"),
    "tb4": Path(__file__).with_name("terminalbench-v4-manifest.json"),
}
RUN_CONTRACT_FILENAME = "terminalbench-run-contract.json"
CONDITIONS_BY_BUDGET = {
    "standard": ("vanilla", "react_v2", "react_v2_random", "action"),
    "double": ("vanilla", "react_v2"),
}
TRAINING_EPOCHS_BY_BUDGET = {"standard": 4, "double": 8}
CAMPAIGN_CELLS = {
    f"{condition}{'_2x' if budget == 'double' else ''}": (condition, budget)
    for budget, conditions in CONDITIONS_BY_BUDGET.items()
    for condition in conditions
}
FOREST_CONDITIONS = {"react_v2", "react_v2_random"}
TEST_REPETITIONS = 3
EVALUATION_PROTOCOL = {
    "optimization_runs_per_configuration": 1,
    "test_repetitions": TEST_REPETITIONS,
    "attempts_per_task_per_repetition": 1,
    "selection_metric": "mean_validation_reward",
    "test_metric": "pass_at_1",
    "standard_deviation_ddof": 1,
}
TemplateFamily = Literal["generic", "openai", "anthropic", "google", "alibaba"]


def seed_candidate(student_model: str, template_family: str, experiment: str) -> tuple[dict[str, str], TemplateFamily]:
    """Build the experiment's seed with the selected provider template.

    Args:
        student_model: Task model used for automatic provider inference.
        template_family: Explicit provider family or ``"auto"``.
        experiment: Benchmark receiving the shared full agent text and skills.

    Returns:
        Editable components and their resolved template family.
    """
    resolved_family = cast(TemplateFamily, resolve_template_family(template_family, student_model))
    if experiment not in EXPERIMENT_MANIFESTS:
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
        help="Both benchmarks optimize the full prompt, tool-description, and skill text",
    )
    parser.add_argument(
        "--condition",
        choices=CONDITIONS_BY_BUDGET["standard"],
        required=True,
        help="Optimization condition to run",
    )
    parser.add_argument(
        "--budget",
        choices=tuple(CONDITIONS_BY_BUDGET),
        default="standard",
        help="Four epochs for all methods; double gives vanilla and full FOREST eight epochs",
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
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=None,
        help="Optional early-stop cap on task evaluations, in addition to the selected epoch budget",
    )
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
    if not trainset or not valset:
        raise ValueError("train and validation selections must both be non-empty")
    if args.reflection_minibatch_size <= 0:
        raise ValueError("--reflection-minibatch-size must be positive")
    if args.max_metric_calls is not None and args.max_metric_calls <= 0:
        raise ValueError("--max-metric-calls must be positive")
    if args.budget not in CONDITIONS_BY_BUDGET or condition not in CONDITIONS_BY_BUDGET[args.budget]:
        raise ValueError("The double budget supports only vanilla GEPA and full FOREST (react_v2)")
    training_epochs = TRAINING_EPOCHS_BY_BUDGET[args.budget]
    iterations_per_epoch = (len(trainset) + args.reflection_minibatch_size - 1) // args.reflection_minibatch_size
    sampled_tasks_per_epoch = iterations_per_epoch * args.reflection_minibatch_size
    candidate, _ = seed_candidate(args.student_model, resolved_family, args.experiment)
    operated = condition in FOREST_CONDITIONS
    reflection_level = args.reflection_level if operated else 0
    controller_selection = (
        "uniform_random"
        if condition == "react_v2_random"
        else "verbalized"
        if operated or condition == "action"
        else None
    )
    controller_policy = (
        UNIFORM_RANDOM_CONTROLLER_POLICY_CONTRACT if condition == "react_v2_random" else CONTROLLER_POLICY_CONTRACT
    )
    return {
        "schema_version": 13,
        "experiment": manifest.experiment,
        "optimization_target": "agent_text",
        "condition": condition,
        "budget": args.budget,
        "controller_selection": controller_selection,
        "component_kinds": manifest.component_kinds,
        "module_selector": "all",
        "document_bundle_version": BUNDLE_VERSION,
        "seed_document_digest": manifest.candidate_digest(candidate),
        "dataset": manifest.dataset,
        "split_policy": manifest.split_policy,
        "task_refs": manifest.task_refs,
        "test_task_ids": [task.task_id for task in manifest.tasks("test")],
        "edit_tool_set": args.edit_tool_set,
        "harbor_process_timeout_sec": args.harbor_process_timeout_sec,
        "manifest": str(manifest.path),
        "max_metric_calls": args.max_metric_calls,
        "evaluation_protocol": dict(EVALUATION_PROTOCOL),
        "reflection_feedback": deepcopy(REFLECTION_FEEDBACK_CONTRACT),
        "failure_policy": deepcopy(FAILURE_POLICY_CONTRACT),
        "optimization_budget": {
            "unit": "training_epochs",
            "reference": "https://arxiv.org/html/2608.23041v1#A2",
            "training_epochs": training_epochs,
            "iterations_per_epoch": iterations_per_epoch,
            "max_iterations": training_epochs * iterations_per_epoch,
            "sampled_training_tasks": training_epochs * sampled_tasks_per_epoch,
            "padding_tasks_per_epoch": sampled_tasks_per_epoch - len(trainset),
            "batch_sampler": "epoch_shuffled",
            "sampling_strategy": "single_mutation",
            "use_merge": False,
        },
        "n_concurrent": args.n_concurrent,
        "proposer_api_base": args.proposer_api_base,
        "proposer_backend": "react_v2" if operated else "stateless",
        "proposer_decoding": experiment_decoding(args.proposer_model),
        "proposer_model": args.proposer_model,
        "proposer_model_version": experiment_model_version(args.proposer_model),
        "proposer_request_overrides": experiment_request_overrides(args.proposer_model),
        "proposer_num_retries": EXPERIMENT_NUM_RETRIES,
        "reflection_level": reflection_level,
        "reflection_minibatch_size": args.reflection_minibatch_size,
        "max_proposer_model_calls": 8 if operated else None,
        "semantic_action_space": (
            deepcopy(SEMANTIC_ACTION_CATALOGS) if reflection_level == 2 or condition == "action" else None
        ),
        "semantic_controller_policy": deepcopy(controller_policy) if reflection_level == 2 else None,
        "stateless_selector_policy": (
            {**stateless_selector_policy_contract("verbalized"), "component_schedule": "per_component"}
            if condition == "action"
            else None
        ),
        "seed": args.seed,
        "student_api_base": args.student_api_base,
        "student_decoding": experiment_decoding(args.student_model),
        "student_model": args.student_model,
        "student_model_version": experiment_model_version(args.student_model),
        "student_request_overrides": experiment_request_overrides(args.student_model),
        "student_model_info": experiment_model_info(args.student_model),
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
    condition = args.condition
    try:
        contract = build_run_contract(args, manifest, trainset, valset, condition, resolved_family)
    except ValueError as exc:
        parser.error(str(exc))
    ensure_run_contract(args.run_dir, contract)

    student_agent_kwargs: dict[str, Any] = {
        "llm_kwargs": {
            "num_retries": EXPERIMENT_NUM_RETRIES,
            **experiment_decoding(args.student_model),
            **experiment_request_overrides(args.student_model),
        }
    }
    student_agent_kwargs["model_info"] = experiment_model_info(args.student_model)

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
        **experiment_request_overrides(args.proposer_model),
    }
    if args.proposer_api_base is not None:
        reflection_lm_kwargs["api_base"] = args.proposer_api_base

    reflection_strategy = None
    if condition in FOREST_CONDITIONS:
        reflection_strategy, _ = build_react_v2_strategy(
            reflection_model=args.proposer_model,
            task_model=args.student_model,
            lm_kwargs=reflection_lm_kwargs,
            level=args.reflection_level,
            edit_tool_set=args.edit_tool_set,
            template_family=resolved_family,
            component_kinds=manifest.component_kinds,
            controller_selection=contract["controller_selection"],
            rng=random.Random(args.seed),
        )
    elif condition == "action":
        reflection_strategy = ComponentActionReflectionLM(
            lm=LM(args.proposer_model, **reflection_lm_kwargs),
            selector_lm=LM(args.proposer_model, **reflection_lm_kwargs),
            component_kinds=manifest.component_kinds,
            template_family=resolved_family,
            rng=random.Random(args.seed),
        )
    optimize(
        seed_candidate=candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=args.proposer_model,
        reflection_lm_kwargs=reflection_lm_kwargs,
        reflection_strategy=reflection_strategy,
        max_metric_calls=args.max_metric_calls,
        stop_callbacks=MaxCandidateProposalsStopper(contract["optimization_budget"]["max_iterations"]),
        batch_sampler="epoch_shuffled",
        reflection_minibatch_size=args.reflection_minibatch_size,
        sampling_strategy=SingleMutationSampling(),
        module_selector=contract["module_selector"],
        use_merge=False,
        raise_on_exception=True,
        run_dir=str(args.run_dir),
        seed=args.seed,
        reflection_level=contract["reflection_level"],
        edit_tool_set=args.edit_tool_set,
        component_kinds=manifest.component_kinds,
        template_family=resolved_family,
        template_model=args.student_model,
    )


if __name__ == "__main__":
    main()
