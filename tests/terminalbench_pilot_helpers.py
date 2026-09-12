"""Create offline pilot artifacts for campaign and checkpoint integration tests."""

import json
from pathlib import Path

from examples.terminalbench.pilot import (
    PILOT_PROTOCOL,
    PILOT_SCHEMA_VERSION,
    complete_pilot,
    load_completed_pilot,
    run_runtime,
)
from gepa.adapters.terminal_bench_adapter.text_scope import TerminalBenchTextScope


def write_pilot_fixture(root: Path, contract, manifest) -> Path:
    """Write both completed stages using a campaign's exact task runtime."""
    smoke = None
    scope = TerminalBenchTextScope("system_prompt", contract["template_family"])
    for stage, count in (("smoke", 3), ("full", 30)):
        directory = root / stage
        directory.mkdir(parents=True, exist_ok=True)
        ids = manifest.splits["train"][:count]
        config = {
            **run_runtime(contract),
            "schema_version": PILOT_SCHEMA_VERSION,
            "pilot_protocol": PILOT_PROTOCOL,
            "stage": stage,
            "split": "train",
            "optimization_scope": scope.name,
            "text_scope": scope.contract(),
            "candidate_digest": manifest.candidate_digest(scope.materialize(scope.seed_candidate())),
            "task_ids": ids,
            "task_refs": {task_id: manifest.task_refs[task_id] for task_id in ids},
            "smoke_evidence": smoke,
        }
        (directory / "canary-config.json").write_text(json.dumps(config))
        outputs = [{"task_id": task_id, "reward": 0.0, "errors": []} for task_id in ids]
        (directory / "task-results.json").write_text(json.dumps(outputs))
        (directory / "token-usage-summary.json").write_text(
            json.dumps({"schema_version": 1, "files": [], "models": {}})
        )
        complete_pilot(directory, 60.0)
        smoke = load_completed_pilot(directory, manifest, stage)
    return root / "full"
