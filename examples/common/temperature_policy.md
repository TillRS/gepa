# Provider temperature policy

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

## Separate sampling settings

This decision changes temperature only. DeepSeek's July 31 card also recommends
top-p 0.95 for agentic work and 1.0 otherwise. The existing shared experiment
profile currently uses 0.95; applying that task distinction is a separate
sampling decision to review interactively. Output-token limits and reasoning
budgets are separate settings too.
