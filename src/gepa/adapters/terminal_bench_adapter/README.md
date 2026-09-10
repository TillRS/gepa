### Terminal-Bench experiments

Select one of two independent experiments with `--experiment`. Neither has an
implicit default or primary status. Both support vanilla GEPA (`vanilla`) and
FOREST (`react_v2`) with identical seeds, editable components, task splits,
student/proposer models, and four-epoch training budgets within an experiment.

| Experiment | Dataset | Editable target | Train / validation / test |
| --- | --- | --- | --- |
| `tb2-system-prompt` | Terminal-Bench 2.0, 89 tasks | One unified `system_prompt` | 30 / 19 / 40 |
| `tb4-agent-text` | Terminal-Bench 4.0.0, 66 tasks | 13 prompts and two skills | 23 / 23 / 20 |

#### TB2: the published optimization surface

The GEPA comparison in [AutoSaddler, Appendix B](https://arxiv.org/html/2608.23041v1#A2)
optimizes Terminus 2's unified system prompt on TB2, with 30/19/40 task splits.
This experiment exposes that same kind of artifact as one component. Tool-use
guidance and response-format instructions inside the prompt are editable; the
parser, tools, agent loop, auxiliary prompts, and verifier remain fixed.

This restriction describes the paper's **GEPA baseline**. AutoSaddler itself
edits a broader harness: its [TB2 patch catalog](https://arxiv.org/html/2608.23041v1#A15)
includes prompt changes, completion reminders, output limits, and startup code.

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
arms and provider section wrapper are our configuration. The four-epoch budget
follows the paper; minibatch size three and the sampler are our GEPA settings,
pending verification against the authors' unreleased TB2 configuration.
In particular, this is not the ReASearch paper's GPT-5/Bash-prompt setup.

The intended TB2 protocol follows AutoSaddler's GEPA baseline. The checked-in
30/19/40 partition remains provisional until the authors' exact task assignments
can be verified; matching the published counts does not establish an exact split.
On September 10, 2026, the [released AutoSaddler repository](https://github.com/microsoft/AutoSaddler/tree/9df6d2e3e1d3946057243690bca28e136fa81179)
contains GAIA2 split manifests and lists Terminal-Bench integration as forthcoming.
The approved Qwen and DeepSeek model arms remain our explicit experimental choice.

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

Both methods explicitly use `module_selector="all"`: each proposal selects all
15 documents, revises them separately using the same minibatch evidence, and
evaluates the combined harness as one child candidate. This applies the
[GEPA FAQ's multi-module efficiency guidance](https://gepa-ai.github.io/gepa/guides/faq/#how-do-i-optimize-multi-module-dspy-programs-efficiently)
to the TB4 text bundle. It increases optimizer-side editing work without adding
separate task evaluations for each document. The four-epoch budget stays fixed;
minibatches already scoring perfectly still skip mutation. TB2 explicitly keeps
`module_selector="round_robin"`, which always selects its sole prompt. The run
contract pins this policy for resume and final-test comparisons, so older runs
using implicit component selection require a fresh run directory.

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

TB4 retains its 20 held-out test tasks and splits the remaining 46 tasks equally:
23 for reflection and 23 for validation-based selection. This follows the
[GEPA FAQ's small-dataset guidance](https://gepa-ai.github.io/gepa/guides/faq/#whats-the-recommended-trainvalidation-split)
to use a 50/50 train/validation split below 200 examples. The 20-task test holdout
is our choice, not a test fraction prescribed by GEPA. Task assignments use the
existing deterministic hash order and are identical across models and optimizers.
This replaces the earlier 26/20/20 allocation without moving any test tasks.

The resume contract records the experiment, dataset, complete task refs and
splits, target, seed digest, models, decoding, and budget. A different experiment,
manifest, or configuration requires a fresh run directory. Old TB3 checkpoints
cannot resume as TB4, and single-prompt candidates cannot enter the full bundle
experiment. The held-out test split is never evaluated automatically.

#### Optimization budget

Both experiments use **four training epochs**, following the TB2 GEPA budget in
[AutoSaddler, Appendix B](https://arxiv.org/html/2608.23041v1#A2). TB4 receives the
same number of passes through its own training split. With the default minibatch
size of three, the stopping rule is `4 * ceil(train_tasks / 3)` iterations:

| Experiment | Training tasks | Iterations per epoch | Total iterations | Training task draws |
| --- | --- | --- | --- | --- |
| TB2 | 30 | 10 | 40 | 120 |
| TB4 | 23 | 8 | 32 | 96 |

GEPA's epoch sampler pads each final minibatch to the configured size. For TB4,
each epoch covers all 23 training tasks and repeats one, giving 92 unpadded draws
plus four padding draws over the run. A training limit or different minibatch
size changes the iteration count using the same rule. These counts describe
sampled training tasks, not total task executions or model calls.

Each iteration samples one minibatch for one mutation attempt; merging is off.
Perfect minibatches or unsuccessful proposals still consume their iteration.
Parent and proposed-candidate evaluations, plus initial and conditional full
validation, contribute to the measured `total_metric_calls`. Validation is
allowed to finish and does not reduce the number of training epochs. Thus the
methods have equal training-pass budgets, not necessarily equal task-execution,
token, or wall-time costs. The run contract records the epoch rule, iteration
limit, sampler, and padding. Resuming continues the original budget rather than
granting four additional epochs.

`--max-metric-calls` is an optional additional early-stop cap for pilot or
operational runs. It is checked at iteration boundaries and can be exceeded by
the final iteration's evaluations. A run stopped by that cap before four epochs
does not complete the standard protocol. The normal commands omit this cap.
Final held-out test evaluation remains separate.

#### Repetitions and final testing

Each benchmark/model arm uses one optimization run per method, followed by
three test repetitions of each frozen harness. This follows
[AutoSaddler, section 5.1 and Table 3](https://arxiv.org/html/2608.23041v1): one
evolution run and three test executions, reporting mean and standard deviation
of Pass@1. Both TB2 and TB4 use this repetition protocol.

The final evaluation command requires a completed vanilla GEPA run and a
completed FOREST run with matching benchmark, model, decoding, optimization seed,
splits, and budget. It rejects partial training/validation selections and runs
that stopped before completing four epochs. It selects each winner by mean
validation reward, with GEPA's earliest-candidate tie break, and freezes both
winners and their common initial harness before running any test task.

Each repetition starts a distinct Harbor job over the entire test split with
`n_attempts=1` and fresh task environments. Training and validation evaluations
remain single-attempt. All three test success rates contribute equally to the
reported mean; no best-of-three selection or Pass@3 aggregation is performed.
The output records sample standard deviation (`ddof=1`) explicitly. Test repeats
measure execution variability for the fixed harness, not optimization-seed
variability.

For each model, the three harnesses are initial, GEPA-selected, and FOREST-selected:

| Experiment | Test tasks | Repetitions per harness | Attempts per harness | Attempts across all three harnesses |
| --- | --- | --- | --- | --- |
| TB2 | 40 | 3 | 120 | 360 |
| TB4 | 20 | 3 | 60 | 180 |

Run final testing only after both matching optimization runs have completed:

```bash
uv run python -m examples.terminalbench.evaluate \
  --vanilla-run-dir runs/tb2-system-prompt/vanilla \
  --forest-run-dir runs/tb2-system-prompt/react_v2 \
  --output-dir runs/tb2-system-prompt/test
```

Use the corresponding directories for TB4 and for the separate DeepSeek arm.
The command reads student model, endpoint, decoding, and concurrency from the
optimization contracts. `--harbor-executable` and `--docker-executable` optionally
select installed binaries. Checkpoints must be trusted local optimization
artifacts because GEPA's checkpoint format uses Python pickle.

`frozen-comparison.json` contains all three harnesses and their source contracts.
Each completed repetition gets a JSON file with per-task verifier rewards and
its distinct Harbor job identity. Rerunning the same command reuses completed
repetitions and runs only missing ones; an interrupted, unrecorded repetition
starts again in fresh environments. Frozen harness or configuration changes
are rejected. The CLI locks the output directory against concurrent writers.
`summary.json` is written only after all nine repetitions finish and contains
the three Pass@1 values, their mean and sample standard deviation, and the
completed task-attempt count for each harness. Scores are fractions in JSON
and percentages in console output. Failed or incomplete Harbor jobs stop the
evaluation instead of becoming fabricated zero scores.

This aligns the repetition protocol with the paper; the previously documented
TB2 task-identity, harness-revision, and model differences still apply.

#### Run

From the repository root, with Docker and the selected model endpoint available:

```bash
uv sync --extra dev
uv tool install --python 3.12 harbor==0.22.0

uv run python -m examples.terminalbench.main \
  --experiment tb2-system-prompt \
  --condition vanilla \
  --run-dir runs/tb2-system-prompt/vanilla \
  --harbor-work-dir runs/tb2-system-prompt/vanilla/harbor

uv run python -m examples.terminalbench.main \
  --experiment tb4-agent-text \
  --condition vanilla \
  --run-dir runs/tb4-agent-text/vanilla \
  --harbor-work-dir runs/tb4-agent-text/vanilla/harbor
```

Use `--condition react_v2` and matching separate output directories for FOREST.
Both commands use the four-epoch budget above.
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
