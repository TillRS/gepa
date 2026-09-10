### Terminal-Bench agent documents

The adapter evaluates a reusable document bundle with Harbor 0.22.0's Terminus
agent on the pinned Terminal-Bench v3 manifest. Vanilla GEPA and FOREST
(`react_v2`) optimize the same 15 documents:

| Surface | Components |
| --- | --- |
| Initial instructions | `instruction_prompt`, `terminal_tool`, `skill_discovery` |
| Context management | `summary`, `summary_questions`, `summary_answers`, `handoff`, `short_summary`, `context_recovery` |
| Completion and recovery | `completion`, `timeout`, `parse_error`, `output_limit` |
| Reusable skills | `skill_debugging`, `skill_verification` |

Prompts use the selected provider's `user_prompt` template. Skills use the
`skill` template: Name, Description, Instructions, and Examples. Their metadata
appears in the initial context; the agent reads each full `SKILL.md` through the
terminal when needed. Both methods can rewrite skill names, descriptions, and
bodies within these two slots.

The JSON command API, tool execution, runtime task and terminal inputs, retry
policy, task splits, and official verifier remain fixed. Parser errors and
observed outputs accompany the editable recovery guidance. Each evaluation
records the full bundle and its content hash alongside Harbor's configuration
and ATIF trajectories. Reflection receives the selected document and complete
execution evidence.

Run from the repository root. Harbor needs a separate Python 3.12 environment
because its LiteLLM requirement differs from GEPA's optional dependencies:

```bash
uv sync --extra dev
uv tool install --python 3.12 harbor==0.22.0
uv run python examples/terminalbench/main.py \
  --condition react_v2 \
  --max-metric-calls 400 \
  --run-dir runs/terminalbench-documents/react_v2 \
  --harbor-work-dir runs/terminalbench-documents/react_v2/harbor
```

The example requires a running Docker daemon and credentials for the selected
student/proposer model. Use `--condition vanilla` with separate output directories
for the baseline. The held-out test split is never evaluated automatically.
Existing single-prompt checkpoints cannot resume as document-bundle runs; use a
fresh run directory. Runtime tests in `tests/harbor/` exercise the pinned agent
without model API calls or Docker.
