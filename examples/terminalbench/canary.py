"""Measure the initial harness on training tasks before freezing runtime settings."""

import argparse
import json
from pathlib import Path

from examples.common.experiment_models import (
    EXPERIMENT_MODELS,
    EXPERIMENT_NUM_RETRIES,
    QWEN3_8_27B_MODEL,
    experiment_model_version,
    experiment_request_overrides,
)
from examples.common.provider_retries import PROVIDER_RETRY_POLICY
from examples.terminalbench.main import EXPERIMENT_MANIFESTS, REPO_ROOT, seed_candidate
from examples.terminalbench.model_settings import (
    terminalbench_decoding,
    terminalbench_limits,
    terminalbench_model_info,
)
from examples.terminalbench.token_usage import TOKEN_USAGE_POLICY, summarize_usage
from gepa.adapters.terminal_bench_adapter import (
    TERMINUS_ADAPTER_CONTRACT,
    HarborCLI,
    TerminusAdapter,
    load_terminalbench_manifest,
)
from gepa.adapters.terminal_bench_adapter.documents import seed_documents
from gepa.adapters.terminal_bench_adapter.terminal_bench_adapter import TASK_CONTEXT_SETTINGS
from gepa.adapters.terminal_bench_adapter.text_scope import OPTIMIZATION_SCOPES, TerminalBenchTextScope
from gepa.strategies.text_limits import parse_text_limits, resolve_text_limits


def main() -> None:
    """Run a separate training-only pilot and retain usage even if its job fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=EXPERIMENT_MANIFESTS, default="tb2.1")
    parser.add_argument("--optimization-scope", choices=OPTIMIZATION_SCOPES, default="all_text")
    parser.add_argument("--model", choices=EXPERIMENT_MODELS, default=QWEN3_8_27B_MODEL)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=3)
    parser.add_argument("--n-concurrent", type=int, default=1, help="Maximum simultaneous training-task trials")
    parser.add_argument("--harbor-executable", default="harbor")
    parser.add_argument("--docker-executable", default="docker")
    parser.add_argument("--text-limits", type=parse_text_limits, default=None)
    args = parser.parse_args()
    text_limits = resolve_text_limits(args.text_limits)
    if args.train_limit <= 0:
        parser.error("--train-limit must be positive")
    if args.n_concurrent <= 0:
        parser.error("--n-concurrent must be positive")
    manifest = load_terminalbench_manifest(EXPERIMENT_MANIFESTS[args.experiment])
    tasks = manifest.tasks("train", args.train_limit)
    candidate, family = seed_candidate(args.model, "auto", args.experiment, args.optimization_scope)
    scope = TerminalBenchTextScope(args.optimization_scope, family)
    limits = terminalbench_limits(args.model)
    agent_kwargs = {
        "model_info": terminalbench_model_info(args.model),
        "token_limits": limits,
        "llm_kwargs": {
            "num_retries": EXPERIMENT_NUM_RETRIES,
            **terminalbench_decoding(args.model),
            **experiment_request_overrides(args.model, explicit_reasoning=True),
        },
    }
    harbor = HarborCLI(
        manifest=manifest,
        student_model=args.model,
        student_api_base=args.api_base,
        work_dir=args.output_dir / "harbor",
        agent_python_path=REPO_ROOT,
        n_concurrent=args.n_concurrent,
        harbor_executable=args.harbor_executable,
        docker_executable=args.docker_executable,
        student_agent_kwargs=agent_kwargs,
        text_limits=text_limits,
    )
    harbor.check_requirements()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "canary-config.json").write_text(
        json.dumps(
            {
                "schema_version": 7,
                "adapter": TERMINUS_ADAPTER_CONTRACT,
                "provider_retry_policy": PROVIDER_RETRY_POLICY,
                "experiment": args.experiment,
                "optimization_scope": scope.name,
                "text_scope": scope.contract(),
                "split": "train",
                "n_concurrent": args.n_concurrent,
                "task_context_settings": dict(TASK_CONTEXT_SETTINGS),
                "task_ids": [task.task_id for task in tasks],
                "task_refs": {task.task_id: manifest.task_refs[task.task_id] for task in tasks},
                "model": args.model,
                "model_version": experiment_model_version(args.model),
                "api_base": args.api_base,
                "template_family": family,
                "candidate_digest": manifest.candidate_digest(scope.materialize(candidate)),
                "reference_seed_digest": manifest.candidate_digest(seed_documents(family)),
                "student_agent_kwargs": agent_kwargs,
                "token_usage_policy": TOKEN_USAGE_POLICY,
                "text_limits": text_limits.to_dict(),
            },
            indent=2,
        )
        + "\n"
    )
    try:
        batch = TerminusAdapter(manifest, harbor, text_scope=scope).evaluate(tasks, candidate)
        (args.output_dir / "task-results.json").write_text(json.dumps(batch.outputs, indent=2) + "\n")
    finally:
        report = summarize_usage([args.output_dir / "harbor"])
        (args.output_dir / "token-usage-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Review training usage before freezing the cap: {args.output_dir / 'token-usage-summary.json'}")


if __name__ == "__main__":
    main()
