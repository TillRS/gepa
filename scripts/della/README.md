# Paired benchmark models

Every benchmark arm uses the same model for execution and optimization:

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

DeepSeek uses temperature 1.0, top-p 0.95, and maximum reasoning through
`chat_template_kwargs`. The context limit is 393,216 tokens; the shared
experiment output limit remains 16,384 tokens per call. This output budget is
smaller than the model author's recommendation for unrestricted maximum reasoning.
See the [model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
and [vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash).

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
