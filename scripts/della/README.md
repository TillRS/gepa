# Paired benchmark models

Every benchmark arm uses the same model for execution and optimization:

Every model role uses the shared [provider retry policy](../../examples/common/provider_retries.md):
at most three attempts for temporary provider failures, with per-attempt logs
and nested SDK retries disabled. Completed answers and benchmark tasks are not
retried by this policy.

Evaluation-result caching is disabled for HotPotQA and TB2.1 across all methods
and budgets. HotPotQA also disables DSPy's disk and memory response caches, so
each newly requested evaluation runs the task model again. Completed checkpoint
records and optimizer response journals remain available for recovery; they do
not serve unrelated new evaluations. Cache settings are part of run identity,
so HotPotQA's earlier cache-enabled checkpoints require a fresh campaign. The
6,871/13,742-call budgets remain unchanged and now fund uncached evaluations.

HotPotQA training minibatches use a separate random stream initialized with the
experiment seed. Every method and budget therefore sees the same sequence of
shuffled training epochs for the same ordered dataset and seed, independent of
parent selection or reflection draws. Larger budgets continue that sequence;
metric-call budgets do not guarantee the same number of optimization iterations.
Checkpoints save the permutation, cursor, and private RNG state. Run-contract
schema 24 records this policy and requires a fresh campaign for older checkpoints.

| Arm | Student and proposer | Serving |
| --- | --- | --- |
| Qwen | `Qwen/Qwen3.8-27B` | Local POSIT/vLLM |
| DeepSeek | `deepseek-ai/DeepSeek-V4-Flash-0731` | Local POSIT/vLLM |

DeepSeek is pinned to revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062`.
The July 31 release preserves the previous DeepSeek experiment's release identity.
Its local runtime requires vLLM 0.25.0 or newer, with the `deepseek_v4`
tokenizer, reasoning parser, and tool parser. The launcher uses eight H200 GPUs,
TP8/EP8, one API server, FP8 KV cache, and no speculative decoding.
The serving environment and checkpoint bytes are frozen before a campaign.

DeepSeek uses temperature 1.0, top-p 0.95 for iterative tool use and 1.0 for
single-call text generation or action selection, and maximum reasoning through
`chat_template_kwargs`. The context limit is 393,216 tokens; the shared
experiment output limit remains 16,384 tokens per call. This output budget is
smaller than the model author's recommendation for unrestricted maximum reasoning.
See the [model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
and [vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash).

HotPotQA uses provider-recommended thinking-mode temperatures for every role,
including the Manifestor: 1.0 for both pinned models. Qwen uses top-p 0.95
throughout. DeepSeek uses top-p 1.0 for HotPotQA execution, stateless rewriting,
the Controller, and the Manifestor; its ReAct editor uses 0.95. The sampling policy
prefers applicable task-specific provider guidance and otherwise uses the
model/mode default. The [source review](../../examples/common/temperature_policy.md)
records the factual-QA, terminal-agent, and optimizer-role decisions, with links
to the exact checkpoint model cards. The prior Manifestor temperature of 0.0
is superseded for HotPotQA, TB2, and TB4. Changed role settings require fresh
run contracts and campaign checkpoints.

Thinking mode and effort are explicit in every HotPotQA request: Qwen uses
`enable_thinking=true` with `reasoning_effort=xhigh`; DeepSeek uses
`thinking=true` with `reasoning_effort=max`. They reach vLLM through
`extra_body.chat_template_kwargs` and are recorded for resume validation.
Qwen follows its provider default; DeepSeek's code-agent setting is also our
chosen setting for QA and optimizer roles, as explained in the source review.
The same policy applies to TB2 and TB4. The per-call output ceiling remains
16,384 tokens, independently of the effort setting.

The FOREST ReAct editor has no assistant-turn or tool-call limit in HotPotQA,
TB2, or TB4. It may make multiple edits within the Controller-selected section,
all serving the same semantic action and Manifestor steering, then explicitly
emit `<finish>`. Each successful edit returns the latest section text. With the
minimal tool basis, each replacement or move must complete its delete/insert
pair before another operation or finish. The protocol is recorded in run
contracts; older checkpoints require a fresh run. The runtime canary retains
its small diagnostic budget to verify one literal edit followed by finish.

All optimizer character limits default to unlimited and are independently
configurable through the shared [text-limit settings](../../examples/common/text_limits.md).
This covers complete prompts and skills, total candidate size, complete optimizer
requests, selector targets, feedback, Manifestor traces and steering, and saved
diagnostic text. Set `HOTPOTQA_TEXT_LIMITS_JSON` for this launcher or pass
`--text-limits` to the Python CLI. Model context and per-call output limits still
apply. Resolved settings are recorded in run contracts for resume validation.

After configuring `scripts/della/.env` from `.env.example`, prepare and submit:

```bash
scripts/della/build_env.sh
MODEL_PROFILE=qwen3.8-27b scripts/della/submit_hotpotqa.sh
MODEL_PROFILE=deepseek-v4-flash scripts/della/submit_hotpotqa.sh
```

Each HotPotQA arm contains the existing six optimization cells. DeepSeek first
runs the multi-tool canary; a failed canary prevents its campaign from starting.
Use a fresh campaign ID after changing models or serving environments. Previous
GLM results and checkpoints cannot be resumed as DeepSeek runs.

HotPotQA reflection uses the configured model context window without an
additional 8,000-character Manifestor trace cap. Exact repeated long text and
paragraphs are shown once per request with references for later occurrences;
long identical-line runs retain one line and a repetition count. Distinct
passages, reasoning, task outcomes, and gold feedback remain available. FOREST
reads feedback once in the per-example traces. Original evaluation records are
unchanged. Context overflow remains a provider error that stops the run; it
does not silently truncate evidence. The reflection policy is part of run
identity, so earlier checkpoints require a fresh campaign.
