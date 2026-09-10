### Terminal-Bench experiments

Select `--experiment tb2` or `--experiment tb4`. Both benchmarks optimize the
same full agent text and skills. Vanilla GEPA (`vanilla`) and FOREST (`react_v2`)
receive identical initial documents, editable components, task splits,
student/proposer models, and four-epoch training budgets within an experiment.
Neither benchmark is the default or designated primary.

| Experiment | Dataset | Editable target | Train / validation / test |
| --- | --- | --- | --- |
| `tb2` | Terminal-Bench 2.0, 89 tasks | 14 prompts and two skills | 30 / 19 / 40 |
| `tb4` | Terminal-Bench 4.0.0, 66 tasks | 14 prompts and two skills | 23 / 23 / 20 |

#### Editable agent text and skills

`PromptedTerminus` runs the same 16-component document bundle for both benchmarks:

| Surface | Components |
| --- | --- |
| Initial and tool instructions | `instruction_prompt`, `terminal_tool`, `command_format`, `skill_discovery` |
| Context management | `summary`, `summary_questions`, `summary_answers`, `handoff`, `short_summary`, `context_recovery` |
| Completion and recovery | `completion`, `timeout`, `parse_error`, `output_limit` |
| Reusable skills | `skill_debugging`, `skill_verification` |

Prompts use the selected provider's `user_prompt` template. Skills use the `skill`
template: Name, Description, Instructions, and Examples. Their metadata appears
in the initial context, and the agent reads each full `SKILL.md` through the
terminal when needed. Both optimizers can rewrite skill metadata and bodies.
Command-format guidance and completion instructions are editable text too.

Both methods explicitly use `module_selector="all"`: each proposal selects all
16 documents, revises them separately using the same minibatch evidence, and
evaluates the combined harness as one child candidate. This applies the
[GEPA FAQ's multi-module efficiency guidance](https://gepa-ai.github.io/gepa/guides/faq/#how-do-i-optimize-multi-module-dspy-programs-efficiently)
to both benchmarks. Optimizer-side editing work is measured separately; selecting
all documents does not require a separate task evaluation for each document.
The four-epoch budget stays fixed, and perfectly scored minibatches still skip
mutation. Run contracts pin the full component set, document bundle version,
and selection policy for resume and final-test comparisons.

The optimization target is **model-facing text and skills**. Python agent logic,
tool implementations, the actual JSON parser/command interface, task inputs,
runtime observations, and the official verifier stay fixed. Rewriting the text
that describes a tool does not change its implementation. Task and terminal-state
fields are appended separately, and candidate braces remain literal. Both
benchmarks retain their official task resource limits and agent timeouts, without
local overrides. An optional `--harbor-process-timeout-sec` is a whole-job
operational limit recorded in the run contract.

#### Reference protocol and pending confirmation

The working decision is to optimize the full text surface on both benchmarks.
It supersedes the earlier TB2 restriction to one unified prompt. The
[AutoSaddler GEPA baseline](https://arxiv.org/html/2608.23041v1#A2) optimized only
that prompt, while AutoSaddler itself could also change executable harness code.
Our GEPA and FOREST comparison now shares the broader text-and-skill scope,
with execution code fixed for both methods.

TB2 retains the paper-inspired 30/19/40 split sizes, four-epoch budget, and
three repeated final evaluations. Its checked-in task assignments remain our
deterministic split; the authors' exact identities and harness revision were
not established from released artifacts. The broader editable surface, common
seed bundle, homogeneous Qwen/DeepSeek arms, and provider section wrappers are
our experiment configuration. This does not reproduce the paper's GEPA setup
or claim direct comparability to its reported scores. TB4 retains its approved
23/23/20 split and the same normalized training-budget rule.

- [ ] Ask Lakshya to confirm the full model-facing text and skill scope for both
  TB2 and TB4, with identical editable components for GEPA and FOREST and fixed
  execution code. This is a research follow-up; the current implementation uses
  the user's approved working decision.

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
manifest, or configuration requires a fresh run directory. Old prompt-only or
earlier document-bundle checkpoints cannot resume under the new scope.
Experiment IDs are now `tb2` and `tb4`; the former
`tb2-system-prompt` and `tb4-agent-text` IDs are rejected rather than silently
changing the optimization target. The held-out test split is never evaluated
automatically.

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
  --vanilla-run-dir runs/tb2/vanilla \
  --forest-run-dir runs/tb2/react_v2 \
  --output-dir runs/tb2/test
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
  --experiment tb2 \
  --condition vanilla \
  --run-dir runs/tb2/vanilla \
  --harbor-work-dir runs/tb2/vanilla/harbor

uv run python -m examples.terminalbench.main \
  --experiment tb4 \
  --condition vanilla \
  --run-dir runs/tb4/vanilla \
  --harbor-work-dir runs/tb4/vanilla/harbor
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
  --experiment tb2 \
  --condition vanilla \
  --student-model hosted_vllm/deepseek-ai/DeepSeek-V4-Flash-0731 \
  --proposer-model hosted_vllm/deepseek-ai/DeepSeek-V4-Flash-0731 \
  --student-api-base http://localhost:8000/v1 \
  --proposer-api-base http://localhost:8000/v1 \
  --run-dir runs/tb2/deepseek/vanilla \
  --harbor-work-dir runs/tb2/deepseek/vanilla/harbor
```

Use the same model and endpoint flags for `tb4`, with matching separate
output paths. Model identity, checkpoint revision, and thinking settings are
recorded in the resume contract; changing any of them requires a fresh run.

Offline tests in `tests/harbor/` exercise the shared actual agent loop for both benchmarks and Harbor job
schemas with simulated model and terminal boundaries. They make no paid model
calls and do not require Docker. The upstream prompt and adapted methods are
Apache-2.0; see `examples/terminalbench/HARBOR_LICENSE`.
