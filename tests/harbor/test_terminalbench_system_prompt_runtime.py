"""Exercise the single-prompt experiment against real Harbor without paid trials."""

import asyncio
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("harbor.agents.terminus_2")

from harbor.agents.terminus_2 import Terminus2
from harbor.llms.base import LLMResponse
from harbor.models.agent.context import AgentContext
from harbor.models.job.config import JobConfig

from examples.terminalbench.main import SYSTEM_PROMPT_SEED_PATH, seed_candidate
from examples.terminalbench.terminus_agent import SystemPromptTerminus
from gepa.adapters.terminal_bench_adapter import HarborCLI, load_terminalbench_manifest
from gepa.adapters.terminal_bench_adapter.documents import escape_document

REPO_ROOT = Path(__file__).parents[2]


def test_tb2_seed_preserves_the_pinned_harbor_prompt() -> None:
    """Compare the vendored seed with the installed runtime's actual prompt."""
    import_path = Path(inspect.getfile(Terminus2))
    source = (import_path.parent / "templates/terminus-json-plain.txt").read_text()
    instructions = source.partition("\nTask Description:\n")[0].format().strip()
    assert SYSTEM_PROMPT_SEED_PATH.read_text().strip() == instructions


def test_tb2_job_schema_accepts_exact_git_sources(tmp_path: Path) -> None:
    """Check the generated TB2 job against Harbor's real task-source schema."""
    manifest = load_terminalbench_manifest(REPO_ROOT / "examples/terminalbench/terminalbench-v2-manifest.json")
    runner = HarborCLI(
        manifest=manifest, student_model="openai/gpt-4o-mini", work_dir=tmp_path, agent_python_path=REPO_ROOT
    )
    task = manifest.tasks("train", 1)[0]
    config = runner.build_job_config(
        [task.task_id],
        prompt_path=tmp_path / "prompt.txt",
        bundle_path=None,
        jobs_dir=tmp_path / "jobs",
        job_name="tb2-schema",
    )
    parsed = JobConfig.model_validate(config)
    assert len(parsed.tasks) == 1 and parsed.datasets == []
    assert str(parsed.tasks[0].path) == task.task_id
    assert parsed.tasks[0].git_commit_id == manifest.task_refs[task.task_id]
    assert parsed.agents[0].override_timeout_sec is None
    assert parsed.timeout_multiplier == 1.0


def test_tb2_real_agent_loop_uses_one_prompt_and_native_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run parsing, terminal execution, and completion with no document bundle or skills."""
    candidate, _ = seed_candidate("openai/gpt-4o-mini", "generic", "tb2-system-prompt")
    candidate["system_prompt"] += "\nTB2_SENTINEL {literal}"
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text(
        escape_document(candidate["system_prompt"])
        + "\nTask Description:\n{instruction}\nCurrent terminal state:\n{terminal_state}"
    )
    model = SimpleNamespace(
        call=AsyncMock(), get_model_context_limit=lambda: 32768, get_model_output_limit=lambda: 4096
    )
    monkeypatch.setattr(Terminus2, "_init_llm", Mock(return_value=model))
    agent = SystemPromptTerminus(
        logs_dir=tmp_path / "logs",
        prompt_template_path=str(prompt_path),
        model_name="openai/gpt-4o-mini",
        record_terminal_session=False,
    )
    environment = SimpleNamespace(exec=AsyncMock(), is_dir=AsyncMock(return_value=True), upload_dir=AsyncMock())
    session = SimpleNamespace(
        get_incremental_output=AsyncMock(return_value="REAL_TERMINAL_STATE"),
        capture_pane=AsyncMock(return_value="REAL_TERMINAL_STATE"),
        send_keys=AsyncMock(),
        is_session_alive=AsyncMock(return_value=True),
    )
    agent._session = session
    response = {"analysis": "verify", "plan": "finish", "commands": []}
    model.call.side_effect = [
        LLMResponse("invalid JSON"),
        LLMResponse(json.dumps({**response, "commands": [{"keystrokes": "pwd\n", "duration": 0.1}]})),
        LLMResponse(json.dumps({**response, "task_complete": True})),
        LLMResponse(json.dumps({**response, "task_complete": True})),
    ]
    asyncio.run(agent.run("REAL_TASK_INPUT", environment, AgentContext()))
    prompts = [call.kwargs["prompt"] for call in model.call.call_args_list]
    assert len(prompts) == 4
    assert candidate["system_prompt"] in prompts[0]
    assert "REAL_TASK_INPUT" in prompts[0] and "REAL_TERMINAL_STATE" in prompts[0]
    assert "available_skills" not in prompts[0]
    assert "JSON" in prompts[1]
    assert "task_complete" in prompts[3]
    environment.exec.assert_not_awaited()
    environment.upload_dir.assert_not_awaited()
    session.send_keys.assert_awaited_once()
    assert agent._summarize.__func__ is Terminus2._summarize
    assert agent._query_llm.__func__ is Terminus2._query_llm
    trace = (tmp_path / "logs/trajectory.json").read_text()
    assert "TB2_SENTINEL" in trace and "REAL_TASK_INPUT" in trace
