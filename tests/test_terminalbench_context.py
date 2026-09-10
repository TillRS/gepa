"""Ensure Terminal-Bench reflection retains decisions without duplicated telemetry."""

from copy import deepcopy

from gepa.adapters.terminal_bench_adapter.context import reflection_trajectories


def _trajectory(steps: list[dict]) -> dict:
    """Build an ATIF trace containing bulky operational metadata."""
    return {
        "schema_version": "ATIF-v1.7",
        "agent": {"name": "terminus", "version": "2", "extra": {"configuration": "CONFIGURATION" * 1000}},
        "final_metrics": {"logprobs": [0.1] * 1000},
        "steps": steps,
    }


def test_projection_preserves_execution_and_references_copied_context() -> None:
    """Keep original steps, new summary reasoning, and copied-step positions once."""
    original = {
        "step_id": 1,
        "source": "agent",
        "message": "Run the failing test",
        "reasoning_content": "Need to verify the output",
        "tool_calls": [{"tool_call_id": "call-1", "function_name": "terminal", "arguments": {"command": "pytest"}}],
        "observation": {"results": [{"content": "FAILED: important error", "source_call_id": "call-1"}]},
        "metrics": {"logprobs": [0.0] * 1000},
    }
    copied = {**deepcopy(original), "is_copied_context": True, "metrics": None}
    traces = [
        _trajectory([copied, {"step_id": 2, "source": "agent", "message": "New summary reasoning"}]),
        _trajectory([original]),
    ]
    untouched = deepcopy(traces)
    projected = reflection_trajectories(traces)
    assert traces == untouched
    assert projected[0]["steps"][0] == {"step_id": 1, "copied_context_from": "Trajectory 2 / Step 1"}
    assert projected[0]["steps"][1]["message"] == "New summary reasoning"
    kept = projected[1]["steps"][0]
    assert kept["message"] == original["message"]
    assert kept["reasoning_content"] == original["reasoning_content"]
    assert kept["tool_calls"] == original["tool_calls"]
    assert kept["observation"] == original["observation"]
    assert "logprobs" not in str(projected) and "CONFIGURATION" not in str(projected)


def test_copied_content_without_a_visible_original_is_kept() -> None:
    """Avoid dangling references or lost evidence when the original trace is absent."""
    text = "\n".join(f"Unique evidence {index}" for index in range(1000))
    traces = [_trajectory([{"step_id": 1, "source": "user", "message": text, "is_copied_context": True}])]
    assert reflection_trajectories(traces)[0]["steps"][0]["message"] == text


def test_real_repeated_actions_remain_separate_steps() -> None:
    """Keep repeated attempts and their outcomes when they are not copied history."""
    traces = [_trajectory([{"step_id": index, "source": "agent", "message": "retry"} for index in range(1, 4)])]
    steps = reflection_trajectories(traces)[0]["steps"]
    assert [step["step_id"] for step in steps] == [1, 2, 3]
    assert all(step["message"] == "retry" for step in steps)


def test_embedded_subagent_evidence_is_kept_and_linked() -> None:
    """Project embedded children without repeating their copied parent history."""
    parent_step = {"step_id": 1, "source": "user", "message": "Task instruction"}
    parent = _trajectory([parent_step])
    parent["subagent_trajectories"] = [
        {**_trajectory([{**parent_step, "is_copied_context": True}]), "trajectory_id": "child-1"}
    ]
    projected = reflection_trajectories([parent])
    assert projected[1]["trajectory_id"] == "child-1"
    assert projected[1]["label"] == "Trajectory 1 / Trajectory 1"
    assert projected[1]["steps"][0]["copied_context_from"] == "Trajectory 1 / Step 1"
