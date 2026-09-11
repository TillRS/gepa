# Provider sampling policy

Reviewed on 2026-09-10 for HotPotQA, Terminal-Bench 2, and Terminal-Bench 4.
Use an applicable task-specific recommendation from the model author for the
exact checkpoint and reasoning mode. Otherwise use that model/mode's general
recommended temperature. A role's name alone does not justify another value.

The current local-vLLM experiments use thinking mode. The source review found
no separate temperature recommendation for factual QA, retrieval-query
generation, evidence summarization, prompt/skill rewriting, Controller action
selection, or Manifestor guidance for these checkpoints. Their general
thinking-mode recommendation therefore applies to those tasks. DeepSeek also
reports code-agent evaluations at the same temperature. This is our application
of provider guidance, not a claim that the providers evaluated every optimizer
role or these exact benchmark versions.

| Work performed | Qwen3.8-27B | DeepSeek-V4-Flash-0731 | Basis |
| --- | ---: | ---: | --- |
| HotPotQA: summaries, retrieval queries, factual answers | 1.0 | 1.0 | General thinking-mode/local-deployment recommendation |
| TB2 and TB4: terminal-agent execution | 1.0 | 1.0 | General recommendation; DeepSeek code-agent evaluation also uses 1.0 |
| GEPA/stateless action proposer: prompt and skill rewriting | 1.0 | 1.0 | General recommendation |
| FOREST Controller: action selection | 1.0 | 1.0 | General recommendation; random Controller makes no model call |
| FOREST Manifestor: action-specific edit guidance | 1.0 | 1.0 | General recommendation; replaces the previous 0.0 override |
| FOREST ReAct editor: editing through tools | 1.0 | 1.0 | General recommendation; DeepSeek agentic recommendation is also 1.0 |

The same role settings apply at standard and double optimization budgets and
during final task evaluation. Temperatures are recorded in run identity; old
Manifestor-0.0 runs cannot resume or enter the new final comparison unchanged.
The shared builder retains its legacy default for callers outside these three
experiments; HotPotQA and Terminal-Bench pass the provider value explicitly.

## Sources and applicability

- [Qwen's pinned model card](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/README.md):
  the API Usage and Best Practices sections recommend temperature 1.0 for
  thinking mode and 0.7 for non-thinking mode. The 0.7 recommendation does not
  apply to the current thinking-mode configuration. The current public card
  was also checked and gives the same values.
- [DeepSeek's pinned July 31 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/README.md):
  How to Run Locally recommends temperature 1.0. The Code Agent benchmark
  notes also report 1.0 with maximum reasoning. The current public card agrees.
- [DeepSeek's hosted thinking-mode API](https://api-docs.deepseek.com/guides/thinking_mode/)
  ignores user-supplied temperature and top-p. That hosted-API behavior does
  not determine the local-vLLM experiment; the checkpoint's local-deployment
  guidance is the applicable source here.

## Top-p by role

Qwen's pinned card recommends top-p 0.95 in thinking mode. DeepSeek's pinned
July 31 card recommends 0.95 for agentic work and 1.0 otherwise. We classify
iterative tool use as agentic, and single-call text generation or action
selection under the general recommendation. This role classification is our
application of the guidance; neither provider names the FOREST roles.

| Work performed | Qwen3.8-27B | DeepSeek-V4-Flash-0731 |
| --- | ---: | ---: |
| HotPotQA: summaries, retrieval queries, factual answers | 0.95 | 1.0 |
| TB2 and TB4: terminal-agent execution | 0.95 | 0.95 |
| GEPA/stateless action proposer: prompt and skill rewriting | 0.95 | 1.0 |
| Stateless action selector and FOREST Controller | 0.95 | 1.0 |
| FOREST Manifestor: action-specific edit guidance | 0.95 | 1.0 |
| FOREST ReAct editor: editing through tools | 0.95 | 0.95 |

These settings apply to both budgets and to final task evaluation. Separate
DeepSeek Controller and ReAct clients preserve each role's sampling, cost
accounting, and response-journal replay. Contracts record the actual role
settings and reject old or mismatched profiles on resume and final comparison.
The shared decoding helper preserves its existing default for other callers;
these three benchmarks choose their role profiles explicitly.

## Reasoning effort

Thinking is explicitly enabled for every model role in HotPotQA, TB2, and TB4,
at both optimization budgets and during final task evaluation:

| Work performed | Qwen3.8-27B | DeepSeek-V4-Flash-0731 |
| --- | --- | --- |
| HotPotQA: summaries, retrieval queries, factual answers | `xhigh` | `max` |
| TB2 and TB4: terminal-agent execution | `xhigh` | `max` |
| GEPA/stateless action proposer and selector | `xhigh` | `max` |
| FOREST Controller, Manifestor, and ReAct editor | `xhigh` | `max` |

Qwen's pinned model card names `xhigh` as its default. DeepSeek's pinned card
reports `max` for code-agent evaluations. Extending DeepSeek `max` to factual QA
and optimizer roles is the approved performance-focused experiment choice;
the provider does not publish a separate recommendation for those roles. The
setting is fixed within each model arm across all six configurations. Provider
effort labels do not imply equal token use or compute across models.

Qwen now sends `enable_thinking=true` and `reasoning_effort=xhigh` through
`extra_body.chat_template_kwargs`. The
[pinned Qwen chat template](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/chat_template.jinja)
consumes these fields, and the [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)
documents that transport. DeepSeek retains `thinking=true` and
`reasoning_effort=max` through its existing local-vLLM template arguments.
These request fields are persisted in run contracts and response-journal
identity. Missing or changed settings cannot silently resume a Qwen run or
enter the final comparison. Other callers retain their existing defaults.

The [AutoSaddler paper](https://arxiv.org/html/2608.23041v1#S5.SS1) names Opus 4.6
with default endpoint settings; its public
[legacy optimizer code](https://github.com/microsoft/AutoSaddler/blob/9df6d2e3e1d3946057243690bca28e136fa81179/src/autosaddler/v1/sdk_session.py#L45)
defaults to `max`. Exact effort for the paper's TB2 task
agent and GEPA baseline was not verified. This policy follows our model arms'
provider guidance and approved choices, without claiming an exact reproduction
of those paper settings.

## Output budgets and context

HotPotQA keeps a **16,384-token output ceiling**. TB2 and TB4 use **32,768
output tokens per model call**, including reasoning and final output, for both
model arms, every optimizer role, both budgets, and final task evaluation.
These are ceilings: natural end-of-sequence stopping stays enabled, with no
minimum generation length. The approved reasoning efforts above are unchanged.

The Terminal-Bench ceiling is our practical starting budget, subject to a
training-only usage and cutoff review before freezing the experiment settings.
It is not a claim to use each provider's maximum-performance output budget.
Qwen's pinned card recommends much larger separate reasoning/final allowances
for agentic work within a 1M context, but reports `max_tokens=32,768` and a 256K
context for its QwenSWEBench coding evaluation. That different benchmark informs
this starting point; it does not establish the right cap for TB2 or TB4.
DeepSeek's pinned card recommends 384K output for `high`/`max`; our smaller cap
deliberately departs from that recommendation.

Context capacity remains separate and unchanged: the configured Qwen server
uses 262,144 tokens and DeepSeek uses 393,216. This change does not enable Qwen
YaRN or million-token serving. Review the pilot's actual output usage and
provider `finish_reason=length` evidence on training tasks, then keep or revise
the cap consistently across methods before the full comparison. Never adjust
it using validation or test outcomes. A changed cap requires fresh run contracts.
See the [Terminal-Bench pilot and usage commands](../../src/gepa/adapters/terminal_bench_adapter/README.md#output-budget-review).
