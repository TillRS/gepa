### Terminal-Bench experiments

Select `--experiment tb2` or `--experiment tb4`. Both benchmarks optimize the
same full agent text and skills. All methods receive identical initial documents,
editable components, task splits, and student/proposer models within an experiment.
Neither benchmark is the default or designated primary.

| Experiment | Dataset | Editable target | Train / validation / test |
| --- | --- | --- | --- |
| `tb2` | Terminal-Bench 2.0, 89 tasks | 14 prompts and two skills | 30 / 19 / 40 |
| `tb4` | Terminal-Bench 4.0.0, 66 tasks | 14 prompts and two skills | 23 / 23 / 20 |

#### Methods and campaign matrix

Each benchmark/model arm follows HotPotQA's six-configuration comparison:

| Condition | Method | Standard budget | Double budget |
| --- | --- | --- | --- |
| `vanilla` | Vanilla GEPA reflection | 4 epochs | 8 epochs |
| `react_v2` | Full FOREST: Controller, Manifestor, ReAct V2 | 4 epochs | 8 epochs |
| `react_v2_random` | FOREST with a uniformly random Controller | 4 epochs | — |
| `action` | Action-conditioned stateless GEPA | 4 epochs | — |

This gives **24 optimization runs**: six configurations, two models, and two
benchmarks. There is one optimization run per configuration. The shared initial
harness is an additional evaluation reference, not an optimization run.

Random-Controller FOREST preserves Manifestor steering, the ReAct V2 editor,
and branch-local edit history. Only Controller selection changes. The `action`
condition uses HotPotQA's `VerbalizedActionSelector` and `StatelessReflectionLM`:
select a semantic action and section, then rewrite that section once without
Manifestor or a tool loop. Every selected prompt and skill gets its own
single-component action job with its matching section template. All edits see
the same parent harness and combine into one child candidate. `action` is no
longer an alias for full FOREST. Controller randomness is seeded, separate from
task sampling, and restored on resume.

HotPotQA's budget levels are 6,871 and 13,742 metric calls. Terminal-Bench retains
the approved epoch-based rule: four and eight training epochs. Thus the method
matrix and 2× budget multiplier match HotPotQA; the budget unit differs.

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
terminal when needed. All methods can rewrite skill metadata and bodies.
Command-format guidance and completion instructions are editable text too.

All methods explicitly use `module_selector="all"`: each proposal selects all
16 documents, revises them separately using the same minibatch evidence, and
evaluates the combined harness as one child candidate. This applies the
[GEPA FAQ's multi-module efficiency guidance](https://gepa-ai.github.io/gepa/guides/faq/#how-do-i-optimize-multi-module-dspy-programs-efficiently)
to both benchmarks. Optimizer-side editing work is measured separately; selecting
all documents does not require a separate task evaluation for each document.
The selected epoch budget stays fixed, and perfectly scored minibatches still skip
mutation. Run contracts pin the full component set, document bundle version,
and selection policy for resume and final-test comparisons.

The optimization target is **model-facing text and skills**. Python agent logic,
tool implementations, the actual JSON parser/command interface, task inputs,
runtime observations, and the official verifier stay fixed. Rewriting the text
that describes a tool does not change its implementation. Task and terminal-state
fields are appended separately, and candidate braces remain literal. Both
benchmarks retain their official task resource limits and agent timeouts, without
local overrides. For each benchmark, all six configurations use the same task
limits, including the standard and double optimization budgets. The double
budget increases optimization opportunities while keeping per-task limits fixed.
An optional `--harbor-process-timeout-sec` is a whole-job operational limit
recorded in the run contract.

#### Textual feedback for reflection

Every method receives the same training-evidence format: task identity, ATIF
execution content, official rewards, and textual verifier diagnostics. The task
instructions, new reasoning, commands, observations, and subagent relationships
remain available. Raw trial/process metadata, token statistics, and model
configuration stay in the original artifacts rather than entering reflection.
The official verifier reward is the optimization score; diagnostic text supplies
evidence for reflection without changing that score.

`Feedback` includes the actual contents of Harbor's `verifier/test-stdout.txt`
and `verifier/test-stderr.txt` when present, plus corresponding logs under
`steps/*/verifier/` for multi-step trials. These are console outputs from the
verifier run, not the verifier implementation or benchmark solution files.
All 16 selected components receive the same feedback. Each optimizer retains
its existing reflection procedure.

Both Terminal-Bench experiments and HotPotQA remove the additional
8,000-character Manifestor trace cap and rely on the configured model's context
window. Exact repeated long strings and paragraphs are shown once per request
with references for later occurrences. Long identical-line runs retain one line
and their repetition count. Distinct text is not shortened. FOREST's Manifestor
and editor read feedback once in the per-example traces. Context overflow stops
the run through the provider error path instead of silently truncating evidence.

Harbor's copied-context steps refer to identical original steps when those
originals are present in the same reflection input. Unmatched copied context and
new subagent reasoning stay intact; repeated actual actions keep their separate
step identities. The adapter retains complete original ATIF and job results,
and it no longer repeats the editable document body in every example's metadata.
Context formatting and the Manifestor limit are pinned for resume and final
comparison alongside the feedback policy.

Each log contributes up to 8,192 source bytes. Longer logs retain equal portions
from the beginning and end, separated by an explicit omitted-byte count. Full
logs stay unchanged in the Harbor artifacts. Text uses UTF-8 with replacement
for invalid bytes. Missing logs are reported as unavailable and do not change a
valid reward; a present but unreadable log raises an evidence error.

Only training tasks may enter reflection. Validation still selects candidates,
and held-out test feedback cannot enter optimization. The run contract pins the
feedback version, log locations, byte limit, decoding, and missing-log policy.
Earlier runs without this feedback contract require fresh directories.

#### Task failures, timeouts, and recovery

All six configurations on both benchmarks use the same policy. A completed
task keeps its official verifier reward, including zero for unsuccessful work.
An `AgentTimeoutError` also keeps the official reward when valid verification
and a trajectory exist; the timeout remains visible in reflection feedback.
Reaching the agent time limit can still score one if the final work passes.
Harbor's job error counter must match exactly these verified trial timeouts.
For multi-step tasks, a timed-out step must have its own verifier rewards;
an aggregate reward cannot hide an unverified step. ATIF trajectories are read
from both `agent/` and Harbor's archived `steps/*/agent/` directories.

Provider, container, verifier, subprocess, and missing/invalid-evidence failures
stop optimization or final testing without a fabricated score. This follows
HotPotQA's distinction between task outcomes and systemic failures; HotPotQA
also scores its specific malformed task-output case as zero. Terminal-Bench's
timeout handling uses Harbor's verifier instead of assigning an automatic zero.

Every Harbor job explicitly uses `n_attempts=1` and `retry.max_retries=0`.
There are no automatic retries of failed jobs or extra attempts to improve a
completed score. Repair infrastructure before explicitly resuming. Completed
test repetitions remain reusable; an interrupted, unrecorded repetition starts
again as described below. Failed-job logs and any Harbor-recorded token/cost
counters remain in their original evaluation directories, separate from scored
GEPA evaluations. Those recovery costs must be reported separately rather than
inferred from `total_metric_calls`; unavailable usage is not zero cost.
Run contracts pin this policy and reject earlier or changed policies on resume
and when freezing final comparisons.

#### Reference protocol and pending confirmation

The working decision is to optimize the full text surface on both benchmarks.
It supersedes the earlier TB2 restriction to one unified prompt. The
[AutoSaddler GEPA baseline](https://arxiv.org/html/2608.23041v1#A2) optimized only
that prompt, while AutoSaddler itself could also change executable harness code.
Our GEPA and FOREST comparison now shares the broader text-and-skill scope,
with execution code fixed for both methods.

TB2 retains the paper-inspired 30/19/40 split sizes, four-epoch standard budget, and
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
- [ ] Ask Lakshya to confirm the train/validation/test splits for TB2 (30/19/40)
  and TB4 (23/23/20), including how tasks are assigned to each split.
- [ ] Gilad: review all implemented deduplication and redundant-context removal
  for HotPotQA, TB2, and TB4: exact-text and paragraph references, repeated log
  lines, Harbor copied history, excluded metadata, duplicate feedback/document
  text, and remaining limits. Check useful-evidence preservation and the final
  model prompts, including JSON-encoded verifier logs.

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
earlier document-bundle checkpoints cannot resume under the new scope. The
current schema also pins the condition, Controller policy, and standard/double
budget; earlier contracts must use fresh run directories.
Experiment IDs are now `tb2` and `tb4`; the former
`tb2-system-prompt` and `tb4-agent-text` IDs are rejected rather than silently
changing the optimization target. The held-out test split is never evaluated
automatically.

#### Optimization budget

Both experiments use **four training epochs at the standard budget**, following the TB2 GEPA budget in
[AutoSaddler, Appendix B](https://arxiv.org/html/2608.23041v1#A2). TB4 receives the
same number of passes through its own training split. `--budget double` gives
vanilla GEPA and full FOREST **eight epochs**, with all other settings fixed.
With the default minibatch size of three, the stopping rule is
`epochs * ceil(train_tasks / 3)` iterations:

| Experiment | Training tasks | Iterations per epoch | Standard / double iterations | Standard / double training draws |
| --- | --- | --- | --- | --- |
| TB2 | 30 | 10 | 40 / 80 | 120 / 240 |
| TB4 | 23 | 8 | 32 / 64 | 96 / 192 |

GEPA's epoch sampler pads each final minibatch to the configured size. For TB4,
each epoch covers all 23 training tasks and repeats one, giving 92 unpadded draws
plus four padding draws in the standard run, or 184 unpadded plus eight padding
draws in the double run. A training limit or different minibatch
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
granting additional epochs. Doubling the budget doubles proposal opportunities;
it does not guarantee twice as many accepted candidates or total metric calls.
The two larger-budget runs start independently from the shared initial harness.
They cannot extend a standard-budget checkpoint in place.

`--max-metric-calls` is an optional additional early-stop cap for pilot or
operational runs. It is checked at iteration boundaries and can be exceeded by
the final iteration's evaluations. A run stopped by that cap before its selected
epoch budget does not complete the protocol. The normal commands omit this cap.
Final held-out test evaluation remains separate.

#### Repetitions and final testing

Each benchmark/model arm uses one optimization run per method/budget configuration, followed by
three test repetitions of each frozen harness. This follows
[AutoSaddler, section 5.1 and Table 3](https://arxiv.org/html/2608.23041v1): one
evolution run and three test executions, reporting mean and standard deviation
of Pass@1. Both TB2 and TB4 use this repetition protocol.

The final evaluation command requires all six completed runs with matching
benchmark, model, decoding, optimization seed, and splits. Each must have the
correct method and budget for its campaign cell. It rejects partial
training/validation selections and runs that stopped before completing their
four or eight epochs. It selects each winner independently by mean validation
reward, with GEPA's earliest-candidate tie break, and freezes all six winners
and their common initial harness before running any test task. Standard and
double-budget results retain separate labels; no selection across budgets
uses test results.

Each repetition starts a distinct Harbor job over the entire test split with
`n_attempts=1` and fresh task environments. Training and validation evaluations
remain single-attempt. All three test success rates contribute equally to the
reported mean; no best-of-three selection or Pass@3 aggregation is performed.
The output records sample standard deviation (`ddof=1`) explicitly. Test repeats
measure execution variability for the fixed harness, not optimization-seed
variability.

For each model, the seven harnesses are the initial harness and the six
validation-selected winners:

| Experiment | Test tasks | Repetitions per harness | Attempts per harness | Attempts across all seven harnesses |
| --- | --- | --- | --- | --- |
| TB2 | 40 | 3 | 120 | 840 |
| TB4 | 20 | 3 | 60 | 420 |

Run final testing only after all six matching optimization runs have completed:

```bash
uv run python -m examples.terminalbench.evaluate \
  --run-dir vanilla=runs/tb2/vanilla \
  --run-dir react_v2=runs/tb2/react_v2 \
  --run-dir react_v2_random=runs/tb2/react_v2_random \
  --run-dir action=runs/tb2/action \
  --run-dir vanilla_2x=runs/tb2/vanilla_2x \
  --run-dir react_v2_2x=runs/tb2/react_v2_2x \
  --output-dir runs/tb2/test
```

Use the corresponding directories for TB4 and for the separate DeepSeek arm.
The command reads student model, endpoint, decoding, and concurrency from the
optimization contracts. `--harbor-executable` and `--docker-executable` optionally
select installed binaries. Checkpoints must be trusted local optimization
artifacts because GEPA's checkpoint format uses Python pickle.

`frozen-comparison.json` contains all seven harnesses and their source contracts.
Each completed repetition gets a JSON file with per-task verifier rewards and
its distinct Harbor job identity. Rerunning the same command reuses completed
repetitions and runs only missing ones; an interrupted, unrecorded repetition
starts again in fresh environments. Frozen harness or configuration changes
are rejected. The CLI locks the output directory against concurrent writers.
`summary.json` is written only after all 21 repetitions finish and contains
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

Use `--condition react_v2`, `--condition react_v2_random`, and `--condition action`
with separate output directories for the other standard-budget methods.
Both commands above default to `--budget standard` (four epochs).
Launch each larger-budget run in its own fresh directory, for example:

```bash
uv run python -m examples.terminalbench.main \
  --experiment tb2 \
  --condition vanilla --budget double \
  --run-dir runs/tb2/vanilla_2x \
  --harbor-work-dir runs/tb2/vanilla_2x/harbor
```

Use `--condition react_v2 --budget double` and `runs/tb2/react_v2_2x` for the
larger-budget FOREST run. Repeat the same six configurations for TB4 and both
model arms. The CLI rejects double-budget ablations outside the approved pair.
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

Temperatures follow the model author's applicable task/mode guidance, with the
general recommendation as the fallback. For both pinned thinking-mode models,
the current recommendation is 1.0 for task execution and every optimizer role,
including the Manifestor. The [provider source review](../../../../examples/common/temperature_policy.md)
records the HotPotQA, TB2, TB4, Controller, Manifestor, and proposer mappings.
The previous Manifestor-0.0 policy cannot resume or enter a final comparison
under the new contract. These values apply at both optimization budgets.

Offline tests in `tests/harbor/` exercise the shared actual agent loop for both benchmarks and Harbor job
schemas with simulated model and terminal boundaries. They make no paid model
calls and do not require Docker. The upstream prompt and adapted methods are
Apache-2.0; see `examples/terminalbench/HARBOR_LICENSE`.
