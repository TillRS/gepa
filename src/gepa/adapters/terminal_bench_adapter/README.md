### Terminal-Bench experiments

Select one of two independent experiments with `--experiment`. Neither has an
implicit default or primary status. Both support vanilla GEPA (`vanilla`) and
FOREST (`react_v2`) with identical seeds, editable components, task splits,
student/proposer models, and metric-call budgets within an experiment.

| Experiment | Dataset | Editable target | Train / validation / test |
| --- | --- | --- | --- |
| `tb2-system-prompt` | Terminal-Bench 2.0, 89 tasks | One unified `system_prompt` | 30 / 19 / 40 |
| `tb4-agent-text` | Terminal-Bench 4.0.0, 66 tasks | 13 prompts and two skills | 26 / 20 / 20 |

#### TB2: the published optimization surface

The GEPA comparison in [AutoSaddler, Appendix B](https://arxiv.org/html/2608.23041v1#A2)
optimizes Terminus 2's unified system prompt on TB2, with 30/19/40 task splits.
This experiment exposes that same kind of artifact as one component. Tool-use
guidance and response-format instructions inside the prompt are editable; the
parser, tools, agent loop, auxiliary prompts, and verifier remain fixed.

`SystemPromptTerminus` inherits Harbor 0.22.0's recovery, summarization, and
completion behavior. It replaces only the initial prompt and disables additional
skill discovery. Terminus delivers this unified prompt in its initial **user**
message; the component name follows the literature's terminology and does not
change the message role.

The seed in `examples/terminalbench/terminus-system-prompt.txt` preserves the
instructions from Harbor 0.22.0's `terminus-json-plain.txt`. Both optimizers receive
the same provider-specific section wrapper. Actual task instructions and terminal
state are appended separately, so they are never optimized or interpreted as
candidate placeholders. Candidate JSON braces remain literal.

This matches the **optimization surface**, not an exact reproduction of a paper's
reported score. The checked-in task identities use our deterministic hash split
with AutoSaddler's split sizes; the paper's exact identities and harness revision
were not established from its released artifacts. The existing homogeneous model
arms, provider section wrapper, and caller-selected budget are our configuration.
In particular, this is not the ReASearch paper's GPT-5/Bash-prompt setup.

#### TB4: full agent text and skills

`PromptedTerminus` exposes the existing 15-component document bundle:

| Surface | Components |
| --- | --- |
| Initial instructions | `instruction_prompt`, `terminal_tool`, `skill_discovery` |
| Context management | `summary`, `summary_questions`, `summary_answers`, `handoff`, `short_summary`, `context_recovery` |
| Completion and recovery | `completion`, `timeout`, `parse_error`, `output_limit` |
| Reusable skills | `skill_debugging`, `skill_verification` |

Prompts use the selected provider's `user_prompt` template. Skills use the `skill`
template: Name, Description, Instructions, and Examples. Their metadata appears
in the initial context, and the agent reads each full `SKILL.md` through the
terminal when needed. Both optimizers can rewrite skill metadata and bodies.

The JSON command interface, execution policy, task inputs, runtime observations,
and official verifier remain fixed. TB4's resource limits and agent timeouts come
from the [official 4.0.0 release](https://www.tbench.ai/news/terminal-bench-4-0),
without local timeout overrides. A caller may set `--harbor-process-timeout-sec`
as a whole-job operational limit; it is recorded in the run contract.

#### Dataset pins and run identity

Both experiments use Harbor **0.22.0** in a separate Python 3.12 environment.

- TB2 uses the official legacy registry's `terminal-bench@2.0` task list from
  [terminal-bench-2 at `69671fba`](https://github.com/laude-institute/terminal-bench-2/tree/69671fbaac6d67a7ef0dfec016cc38a64ef7a77c).
  Every Harbor job contains explicit Git paths and commits; it does not resolve
  a mutable registry version during evaluation.
- TB4 uses `terminal-bench/terminal-bench@4.0.0`, registry content hash
  `sha256:39d9f44b40420cde8fdcc087579c0d72a7e14fa3656d603c3f0d22fb35e27732`,
  and [source tag `v4.0.0`](https://github.com/harbor-framework/terminal-bench/tree/v4.0.0).
  Its checked-in manifest includes all 66 task content hashes.

Manifest validation checks the exact source metadata, task-reference digest,
split sizes, deterministic ordering, and disjoint coverage. Each evaluation
retains its candidate, experiment identity, Harbor job configuration, verifier
results, and ATIF trajectories. Reflection identifies the selected dataset and
only exposes components belonging to that experiment.

The resume contract records the experiment, dataset, complete task refs and
splits, target, seed digest, models, decoding, and budget. A different experiment,
manifest, or configuration requires a fresh run directory. Old TB3 checkpoints
cannot resume as TB4, and single-prompt candidates cannot enter the full bundle
experiment. The held-out test split is never evaluated automatically.

#### Run

From the repository root, with Docker and the selected model endpoint available:

```bash
uv sync --extra dev
uv tool install --python 3.12 harbor==0.22.0

uv run python -m examples.terminalbench.main \
  --experiment tb2-system-prompt \
  --condition vanilla \
  --max-metric-calls 400 \
  --run-dir runs/tb2-system-prompt/vanilla \
  --harbor-work-dir runs/tb2-system-prompt/vanilla/harbor

uv run python -m examples.terminalbench.main \
  --experiment tb4-agent-text \
  --condition vanilla \
  --max-metric-calls 400 \
  --run-dir runs/tb4-agent-text/vanilla \
  --harbor-work-dir runs/tb4-agent-text/vanilla/harbor
```

Use `--condition react_v2` and matching separate output directories for FOREST.
The displayed 400-call budget is an example, not a paper reproduction preset.
An optional `--manifest` must match the explicitly selected experiment.

Both experiments support two separate model arms: Qwen3.8-27B with Qwen3.8-27B
(the model default), and DeepSeek V4 Flash with DeepSeek V4 Flash. Student,
proposer, and Controller use the same model within an arm. Both are served through
local vLLM. DeepSeek uses the pinned July 31 checkpoint, maximum thinking, and the
native `deepseek_v4` tokenizer and parsers on vLLM 0.25.0 or newer; see the
[serving configuration](../../../../scripts/della/README.md).

For the DeepSeek arm, point both roles at the prepared endpoint and use separate
output directories:

```bash
uv run python -m examples.terminalbench.main \
  --experiment tb2-system-prompt \
  --condition vanilla \
  --student-model hosted_vllm/deepseek-ai/DeepSeek-V4-Flash-0731 \
  --proposer-model hosted_vllm/deepseek-ai/DeepSeek-V4-Flash-0731 \
  --student-api-base http://localhost:8000/v1 \
  --proposer-api-base http://localhost:8000/v1 \
  --max-metric-calls 400 \
  --run-dir runs/tb2-system-prompt/deepseek/vanilla \
  --harbor-work-dir runs/tb2-system-prompt/deepseek/vanilla/harbor
```

Use the same model and endpoint flags for `tb4-agent-text`, with matching separate
output paths. Model identity, checkpoint revision, and thinking settings are
recorded in the resume contract; changing any of them requires a fresh run.

Offline tests in `tests/harbor/` exercise both actual agent loops and Harbor job
schemas with simulated model and terminal boundaries. They make no paid model
calls and do not require Docker. The upstream prompt and adapted methods are
Apache-2.0; see `examples/terminalbench/HARBOR_LICENSE`.
