"""Project Harbor traces into execution evidence without telemetry or copied histories."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

STEP_FIELDS = ("source", "message", "reasoning_content", "tool_calls", "observation")


def reflection_trajectories(trajectories: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Retain execution content and reference copied steps whose originals are present.

    Args:
        trajectories: Complete ATIF documents retained in the evaluation artifacts.

    Returns:
        New trace documents containing agent identity, messages, reasoning,
        commands, observations, and subagent relationships. Telemetry and raw
        model configuration stay in the original artifacts. Copied steps retain
        their position and refer to an identical visible source when possible.
    """
    located: list[tuple[str, Mapping[str, Any]]] = []

    def locate(items: Sequence[Mapping[str, Any]], prefix: str = "") -> None:
        """Assign stable labels to top-level and embedded subagent trajectories."""
        for index, trajectory in enumerate(items, 1):
            label = f"{prefix}Trajectory {index}"
            located.append((label, trajectory))
            locate(trajectory.get("subagent_trajectories") or [], prefix=f"{label} / ")

    locate(trajectories)
    originals: dict[str, str] = {}
    contents: dict[str, tuple[dict[str, Any], str]] = {}
    for label, trajectory in located:
        for step in trajectory["steps"]:
            location = f"{label} / Step {step['step_id']}"
            content = {key: deepcopy(step[key]) for key in STEP_FIELDS if step.get(key) is not None}
            fingerprint = json.dumps(content, sort_keys=True, ensure_ascii=False)
            contents[location] = content, fingerprint
            if not step.get("is_copied_context"):
                originals.setdefault(fingerprint, location)

    result = []
    for label, trajectory in located:
        agent = trajectory.get("agent", {})
        projected = {
            "label": label,
            "schema_version": trajectory.get("schema_version", "ATIF-v1.7"),
            "agent": {key: agent[key] for key in ("name", "version") if key in agent},
            **{key: trajectory[key] for key in ("trajectory_id", "session_id") if key in trajectory},
            "steps": [],
        }
        for step in trajectory["steps"]:
            location = f"{label} / Step {step['step_id']}"
            content, fingerprint = contents[location]
            if step.get("is_copied_context") and fingerprint in originals:
                projected["steps"].append({"step_id": step["step_id"], "copied_context_from": originals[fingerprint]})
            else:
                projected["steps"].append(
                    {
                        "step_id": step["step_id"],
                        **content,
                        **({"is_copied_context": True} if step.get("is_copied_context") else {}),
                    }
                )
        result.append(projected)
    return result
