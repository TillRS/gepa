"""Offline tests for the Terminal-Bench experiment CLI contract."""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from examples.common.experiment_models import (
    DEEPSEEK_V4_FLASH_MODEL,
    DEEPSEEK_V4_FLASH_MODEL_INFO,
    EXPERIMENT_NUM_RETRIES,
    QWEN3_8_27B_MODEL,
    QWEN3_8_27B_MODEL_INFO,
    experiment_decoding,
)
from examples.terminalbench import main as terminalbench_main
from examples.terminalbench.main import (
    EXPERIMENT_MANIFESTS,
    SYSTEM_PROMPT_SEED_PATH,
    build_parser,
    build_run_contract,
    ensure_run_contract,
    seed_candidate,
)
from gepa import optimize
from gepa.adapters.terminal_bench_adapter import load_terminalbench_manifest
from gepa.adapters.terminal_bench_adapter.documents import COMPONENT_KINDS
from gepa.core.adapter import EvaluationBatch
from gepa.strategies.document_template import TEMPLATE_FAMILIES
from gepa.strategies.intervention import CONTROLLER_POLICY_CONTRACT, SEMANTIC_ACTION_CATALOGS

MANIFEST_PATH = Path(__file__).parents[1] / "examples" / "terminalbench" / "terminalbench-v4-manifest.json"


def _model_args(tmp_path: Path, student_model: str, proposer_model: str) -> argparse.Namespace:
    """Parse a complete Terminal-Bench invocation for one model pair.

    Args:
        tmp_path: Pytest directory used for required output paths.
        student_model: Model assigned to Terminus.
        proposer_model: Model assigned to GEPA reflection.

    Returns:
        Parsed arguments ready for the run-contract builder.
    """
    return build_parser().parse_args(
        [
            "--experiment",
            "tb4-agent-text",
            "--condition",
            "react_v2",
            "--student-model",
            student_model,
            "--proposer-model",
            proposer_model,
            "--max-metric-calls",
            "400",
            "--manifest",
            str(MANIFEST_PATH),
            "--run-dir",
            str(tmp_path / "run"),
            "--harbor-work-dir",
            str(tmp_path / "harbor"),
        ]
    )


def test_qwen_student_uses_alibaba_user_prompt_template() -> None:
    """Render the Qwen seed as a sparse Alibaba user prompt."""
    candidate, family = seed_candidate(QWEN3_8_27B_MODEL, "auto", "tb4-agent-text")
    prompt = candidate["instruction_prompt"]
    bodies = TEMPLATE_FAMILIES[family]["user_prompt"].parse(prompt)

    assert family == "alibaba"
    assert "assigned command-line task" in bodies["Objective"]
    assert bodies["Context"] == ""
    assert [line for line in prompt.splitlines() if line.startswith("## ")] == ["## Objective"]


def test_deepseek_student_uses_generic_user_prompt_template() -> None:
    """Render the DeepSeek seed as a sparse generic user prompt."""
    candidate, family = seed_candidate(DEEPSEEK_V4_FLASH_MODEL, "auto", "tb4-agent-text")
    prompt = candidate["instruction_prompt"]
    bodies = TEMPLATE_FAMILIES[family]["user_prompt"].parse(prompt)

    assert family == "generic"
    assert "assigned command-line task" in bodies["Task"]
    assert all(not body for section, body in bodies.items() if section != "Task")
    assert [line for line in prompt.splitlines() if line.startswith("## ")] == ["## Task"]


def test_parser_exposes_react_v2_condition_and_ablation_axes() -> None:
    """Expose the ReAct V2 condition, reflection level, tools, and templates."""
    help_text = build_parser().format_help()

    assert "react_v2" in help_text
    assert "--reflection-level" in help_text
    assert "--edit-tool-set" in help_text
    assert "--template-family" in help_text


def test_parser_defaults_both_roles_to_qwen3_8_27b(tmp_path: Path) -> None:
    """Use the homogeneous Qwen condition when model flags are omitted.

    Args:
        tmp_path: Pytest directory used for required CLI paths.
    """
    args = build_parser().parse_args(
        [
            "--experiment",
            "tb4-agent-text",
            "--condition",
            "react_v2",
            "--max-metric-calls",
            "400",
            "--run-dir",
            str(tmp_path / "run"),
            "--harbor-work-dir",
            str(tmp_path / "harbor"),
        ]
    )

    assert args.student_model == QWEN3_8_27B_MODEL
    assert args.proposer_model == QWEN3_8_27B_MODEL


def test_run_contract_allows_exact_resume_and_rejects_drift(tmp_path: Path) -> None:
    """Accept an exact run contract while rejecting changed resume settings.

    Args:
        tmp_path: Pytest directory used for the isolated run contract.
    """
    contract = {"condition": "react_v2", "student_model": QWEN3_8_27B_MODEL, "edit_tool_set": "broad"}
    path = ensure_run_contract(tmp_path, contract)

    assert ensure_run_contract(tmp_path, contract) == path
    with pytest.raises(ValueError, match="different Terminal-Bench configuration"):
        ensure_run_contract(tmp_path, {**contract, "edit_tool_set": "minimal"})


def test_generated_run_contract_records_metric_call_budget(tmp_path: Path) -> None:
    """Record metric budget and semantic Controller policy in generated state.

    Args:
        tmp_path: Pytest directory used for parsed output paths.
    """
    args = build_parser().parse_args(
        [
            "--experiment",
            "tb4-agent-text",
            "--condition",
            "react_v2",
            "--max-metric-calls",
            "400",
            "--manifest",
            str(MANIFEST_PATH),
            "--run-dir",
            str(tmp_path / "run"),
            "--harbor-work-dir",
            str(tmp_path / "harbor"),
        ]
    )
    manifest = load_terminalbench_manifest(MANIFEST_PATH)

    contract = build_run_contract(
        args,
        manifest,
        manifest.tasks("train", 1),
        manifest.tasks("val", 1),
        "react_v2",
        "alibaba",
    )

    assert contract["max_metric_calls"] == 400
    assert contract["schema_version"] == 9
    assert contract["component_kinds"] == COMPONENT_KINDS
    assert contract["student_model"] == QWEN3_8_27B_MODEL
    assert contract["proposer_model"] == QWEN3_8_27B_MODEL
    assert contract["student_decoding"] == experiment_decoding(QWEN3_8_27B_MODEL)
    assert contract["student_model_info"] == QWEN3_8_27B_MODEL_INFO
    assert contract["proposer_decoding"] == experiment_decoding(QWEN3_8_27B_MODEL)
    assert contract["student_num_retries"] == EXPERIMENT_NUM_RETRIES
    assert contract["proposer_num_retries"] == EXPERIMENT_NUM_RETRIES
    assert contract["semantic_action_space"] == SEMANTIC_ACTION_CATALOGS
    assert contract["semantic_controller_policy"] == CONTROLLER_POLICY_CONTRACT


def test_deepseek_run_contract_uses_the_separate_same_model_condition(tmp_path: Path) -> None:
    """Record DeepSeek V4 Flash in both roles with its fixed decoding.

    Args:
        tmp_path: Pytest directory used for parsed output paths.
    """
    args = _model_args(tmp_path, DEEPSEEK_V4_FLASH_MODEL, DEEPSEEK_V4_FLASH_MODEL)
    manifest = load_terminalbench_manifest(MANIFEST_PATH)

    contract = build_run_contract(
        args,
        manifest,
        manifest.tasks("train", 1),
        manifest.tasks("val", 1),
        "react_v2",
        "generic",
    )

    assert contract["student_model"] == DEEPSEEK_V4_FLASH_MODEL
    assert contract["proposer_model"] == DEEPSEEK_V4_FLASH_MODEL
    assert contract["student_decoding"] == experiment_decoding(DEEPSEEK_V4_FLASH_MODEL)
    assert contract["student_model_info"] == DEEPSEEK_V4_FLASH_MODEL_INFO
    assert contract["proposer_decoding"] == experiment_decoding(DEEPSEEK_V4_FLASH_MODEL)


@pytest.mark.parametrize("experiment", EXPERIMENT_MANIFESTS)
@pytest.mark.parametrize("condition", ["vanilla", "react_v2"])
def test_deepseek_settings_reach_both_runtime_roles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, experiment: str, condition: str
) -> None:
    """Forward DeepSeek thinking and context settings through both experiment CLIs."""
    requirements = Mock()
    monkeypatch.setattr(terminalbench_main.HarborCLI, "check_requirements", requirements)
    harbor_factory = Mock(wraps=terminalbench_main.HarborCLI)
    optimizer = Mock()
    monkeypatch.setattr(terminalbench_main, "HarborCLI", harbor_factory)
    monkeypatch.setattr(terminalbench_main, "optimize", optimizer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "terminalbench",
            "--experiment",
            experiment,
            "--condition",
            condition,
            "--student-model",
            DEEPSEEK_V4_FLASH_MODEL,
            "--proposer-model",
            DEEPSEEK_V4_FLASH_MODEL,
            "--student-api-base",
            "http://localhost:8000/v1",
            "--proposer-api-base",
            "http://localhost:8000/v1",
            "--max-metric-calls",
            "4",
            "--train-limit",
            "1",
            "--val-limit",
            "1",
            "--run-dir",
            str(tmp_path / "run"),
            "--harbor-work-dir",
            str(tmp_path / "harbor"),
        ],
    )

    terminalbench_main.main()

    requirements.assert_called_once_with()
    harbor_kwargs = harbor_factory.call_args.kwargs
    optimize_kwargs = optimizer.call_args.kwargs
    student_kwargs = harbor_kwargs["student_agent_kwargs"]
    expected_body = {"chat_template_kwargs": {"thinking": True, "reasoning_effort": "max"}}
    assert harbor_kwargs["student_model"] == optimize_kwargs["reflection_lm"] == DEEPSEEK_V4_FLASH_MODEL
    assert student_kwargs["model_info"] == DEEPSEEK_V4_FLASH_MODEL_INFO
    assert student_kwargs["llm_kwargs"]["extra_body"] == expected_body
    assert optimize_kwargs["reflection_lm_kwargs"]["extra_body"] == expected_body
    assert harbor_kwargs["student_api_base"] == optimize_kwargs["reflection_lm_kwargs"]["api_base"]
    assert len(optimize_kwargs["trainset"]) == len(optimize_kwargs["valset"]) == 1
    manifest = load_terminalbench_manifest(EXPERIMENT_MANIFESTS[experiment])
    assert set(optimize_kwargs["seed_candidate"]) == set(manifest.component_kinds)
    assert optimize_kwargs["reflection_level"] == (0 if condition == "vanilla" else 2)
    assert optimize_kwargs["stop_callbacks"].max_proposals == 4
    assert optimize_kwargs["max_metric_calls"] == 4
    assert optimize_kwargs["batch_sampler"] == "epoch_shuffled"
    assert optimize_kwargs["module_selector"] == ("all" if experiment == "tb4-agent-text" else "round_robin")
    assert optimize_kwargs["use_merge"] is False
    contract = json.loads((tmp_path / "run" / "terminalbench-run-contract.json").read_text())
    assert contract["experiment"] == experiment
    assert contract["module_selector"] == optimize_kwargs["module_selector"]
    assert (
        contract["student_model_version"]
        == contract["proposer_model_version"]
        == ("7872f01b1d1fe23eabc4c98b48bffcef5a386062")
    )
    assert (
        contract["student_request_overrides"] == contract["proposer_request_overrides"] == {"extra_body": expected_body}
    )


@pytest.mark.parametrize("experiment,iterations,padding", [("tb2-system-prompt", 40, 0), ("tb4-agent-text", 32, 1)])
@pytest.mark.parametrize("outcome", ["accepted", "rejected", "perfect"])
def test_four_epoch_cli_budget_stops_and_resumes_with_real_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, experiment: str, iterations: int, padding: int, outcome: str
) -> None:
    """Complete four covered epochs across resume regardless of proposal success."""
    run_dir = tmp_path / "run"
    stop_file = run_dir / "gepa.stop"
    parent_batches = []
    evaluations = []
    selected_components = []
    trainset = load_terminalbench_manifest(EXPERIMENT_MANIFESTS[experiment]).tasks("train")

    class SamplingRecorder:
        """Record each sampling step once and pause partway through an epoch."""

        def on_minibatch_sampled(self, event):
            """Pause after the fifth minibatch while letting its evaluations finish."""
            parent_batches.append([trainset[index].task_id for index in event["minibatch_ids"]])
            if len(parent_batches) == 5:
                stop_file.touch()

    class BudgetAdapter:
        """Exercise real sampling, stopping, validation, and checkpoints offline."""

        def evaluate(self, batch, candidate, capture_traces=False):
            """Return controlled rewards for training and validation tasks."""
            evaluations.append([task.task_id for task in batch])
            score = sum(text.count("budget_step") for text in candidate.values()) / (100 * len(candidate))
            if outcome != "accepted":
                score = 1.0 if outcome == "perfect" else 0.0
            return EvaluationBatch(
                outputs=[{} for _ in batch],
                scores=[score for _ in batch],
                trajectories=[{} for _ in batch] if capture_traces else None,
                num_metric_calls=len(batch),
            )

        def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
            """Supply deterministic feedback without invoking a model."""
            return {key: [{"feedback": "Try the next revision"}] for key in components_to_update}

        def propose_new_texts(self, candidate, reflective_dataset, components_to_update):
            """Append a revision marker inside each selected document."""
            selected_components.append(set(components_to_update))
            return {key: candidate[key] + "\nbudget_step" for key in components_to_update}

    results = []

    def optimize_offline(**kwargs):
        """Keep CLI budget wiring while replacing model work with the adapter."""
        kwargs["reflection_lm"] = None
        kwargs["callbacks"] = [SamplingRecorder()]
        results.append(optimize(**kwargs, display_progress_bar=False, raise_on_exception=True))

    monkeypatch.setattr(terminalbench_main.HarborCLI, "check_requirements", Mock())
    monkeypatch.setattr(terminalbench_main, "TerminalBenchAdapter", lambda *args: BudgetAdapter())
    monkeypatch.setattr(terminalbench_main, "optimize", optimize_offline)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "terminalbench",
            "--experiment",
            experiment,
            "--condition",
            "vanilla",
            "--run-dir",
            str(run_dir),
            "--harbor-work-dir",
            str(tmp_path / "harbor"),
        ],
    )

    terminalbench_main.main()
    assert len(parent_batches) == 5
    stop_file.unlink()
    terminalbench_main.main()
    assert len(parent_batches) == iterations
    terminalbench_main.main()
    assert len(parent_batches) == iterations

    contract = json.loads((run_dir / "terminalbench-run-contract.json").read_text())
    budget = contract["optimization_budget"]
    assert contract["max_metric_calls"] is None
    assert budget["training_epochs"] == 4
    assert budget["max_iterations"] == iterations
    assert budget["padding_tasks_per_epoch"] == padding
    assert budget["sampled_training_tasks"] == sum(map(len, parent_batches)) == iterations * 3
    for start in range(0, iterations, iterations // 4):
        epoch_ids = [task_id for batch in parent_batches[start : start + iterations // 4] for task_id in batch]
        assert set(epoch_ids) == set(contract["train_task_ids"])
        assert len(epoch_ids) == len(contract["train_task_ids"]) + padding
    assert not set(contract["test_task_ids"]).intersection(task_id for batch in evaluations for task_id in batch)
    per_iteration = 3 if outcome == "perfect" else 6
    if outcome == "accepted":
        per_iteration += len(contract["val_task_ids"])
    assert results[-1].total_metric_calls == len(contract["val_task_ids"]) + iterations * per_iteration
    assert selected_components == ([] if outcome == "perfect" else [set(contract["component_kinds"])] * iterations)
    if outcome == "accepted":
        assert len(results[-1].candidates) == iterations + 1
        assert all(text.count("budget_step") == iterations for text in results[-1].candidates[-1].values())


@pytest.mark.parametrize("field,value", [("reflection_minibatch_size", 0), ("max_metric_calls", -1)])
def test_run_contract_rejects_invalid_budget_inputs(tmp_path: Path, field: str, value: int) -> None:
    """Reject invalid budget arithmetic before starting any Harbor work."""
    args = _model_args(tmp_path, QWEN3_8_27B_MODEL, QWEN3_8_27B_MODEL)
    setattr(args, field, value)
    manifest = load_terminalbench_manifest(MANIFEST_PATH)
    with pytest.raises(ValueError, match="must be positive"):
        build_run_contract(args, manifest, manifest.tasks("train"), manifest.tasks("val"), "vanilla", "alibaba")


def test_run_contract_rejects_a_cross_model_pair(tmp_path: Path) -> None:
    """Reject a Qwen student paired with the DeepSeek proposer.

    Args:
        tmp_path: Pytest directory used for parsed output paths.
    """
    args = _model_args(tmp_path, QWEN3_8_27B_MODEL, DEEPSEEK_V4_FLASH_MODEL)
    manifest = load_terminalbench_manifest(MANIFEST_PATH)

    with pytest.raises(ValueError, match="same model"):
        build_run_contract(
            args,
            manifest,
            manifest.tasks("train", 1),
            manifest.tasks("val", 1),
            "react_v2",
            "alibaba",
        )


def test_run_contract_rejects_an_unknown_model_pair(tmp_path: Path) -> None:
    """Reject homogeneous models outside the two configured experiment arms.

    Args:
        tmp_path: Pytest directory used for parsed output paths.
    """
    args = _model_args(tmp_path, "provider/unknown", "provider/unknown")
    manifest = load_terminalbench_manifest(MANIFEST_PATH)

    with pytest.raises(ValueError, match="Unsupported experiment model"):
        build_run_contract(
            args,
            manifest,
            manifest.tasks("train", 1),
            manifest.tasks("val", 1),
            "react_v2",
            "generic",
        )


def test_legacy_state_without_contract_is_not_resumed(tmp_path: Path) -> None:
    """Reject legacy GEPA state that lacks a Terminal-Bench contract.

    Args:
        tmp_path: Pytest directory containing simulated legacy state.
    """
    (tmp_path / "gepa_state.bin").write_bytes(b"old-state")

    with pytest.raises(ValueError, match=r"no terminalbench-run-contract\.json"):
        ensure_run_contract(tmp_path, {"condition": "react_v2"})


def test_experiment_has_no_implicit_default(tmp_path: Path) -> None:
    """Require an explicit experiment so neither configuration becomes primary."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--condition",
                "vanilla",
                "--max-metric-calls",
                "1",
                "--run-dir",
                str(tmp_path),
                "--harbor-work-dir",
                str(tmp_path / "harbor"),
            ]
        )


@pytest.mark.parametrize("family", TEMPLATE_FAMILIES)
def test_tb2_seed_contains_only_the_unified_terminus_prompt(family: str) -> None:
    """Preserve native Terminus instructions inside one editable prompt component."""
    candidate, resolved = seed_candidate(QWEN3_8_27B_MODEL, family, "tb2-system-prompt")
    assert set(candidate) == {"system_prompt"}
    sections = TEMPLATE_FAMILIES[resolved]["system_prompt"].parse(candidate["system_prompt"])
    assert [body.strip() for body in sections.values() if body.strip()] == [SYSTEM_PROMPT_SEED_PATH.read_text().strip()]
    assert "{instruction}" not in candidate["system_prompt"]
    assert "{terminal_state}" not in candidate["system_prompt"]


def test_experiment_manifest_mismatch_and_cross_experiment_resume_are_rejected(tmp_path: Path) -> None:
    """Prevent the selected target, manifest, or resumable state from drifting apart."""
    args = _model_args(tmp_path, QWEN3_8_27B_MODEL, QWEN3_8_27B_MODEL)
    contracts = []
    for experiment, path in EXPERIMENT_MANIFESTS.items():
        manifest = load_terminalbench_manifest(path)
        args.experiment = experiment
        contracts.append(
            build_run_contract(args, manifest, manifest.tasks("train"), manifest.tasks("val"), "vanilla", "generic")
        )
    assert contracts[0]["seed_document_digest"] != contracts[1]["seed_document_digest"]
    ensure_run_contract(tmp_path / "resume", contracts[0])
    with pytest.raises(ValueError, match="different Terminal-Bench configuration"):
        ensure_run_contract(tmp_path / "resume", contracts[1])
    manifest = load_terminalbench_manifest(EXPERIMENT_MANIFESTS["tb2-system-prompt"])
    with pytest.raises(ValueError, match="must match"):
        build_run_contract(args, manifest, manifest.tasks("train"), manifest.tasks("val"), "vanilla", "generic")
