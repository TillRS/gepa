---
name: della
description: >-
  Run GEPA experiments on Princeton's della GPU/SLURM cluster from a laptop:
  SSH setup (key + password, ControlMaster), scripts/della/*.sh launchers,
  the pinned HotPotQA campaign runbook, monitoring/resuming Slurm jobs, and
  storage/quota rules. Use whenever a task mentions della, Slurm (sbatch,
  squeue, sacct), Della logs/results, vLLM serving on della, or
  "disk quota exceeded" there.
---

# Working with della (Princeton Research Computing)

Della is a SLURM GPU cluster. This repo drives it from a laptop through
`scripts/della/*.sh`, which read all connection and cluster config from the
gitignored `scripts/della/.env` (template: `scripts/della/.env.example`).
Refs: https://researchcomputing.princeton.edu/systems/della and
https://researchcomputing.princeton.edu/support/knowledge-base/data-storage

## The scripts (use these; never hand-roll ssh/rsync/sbatch)

Local launchers (laptop, repo root):
- `scripts/della/della_session.sh open|status|close`: open the persistent SSH
  master connections (see SSH below). Run `open` once per laptop session
  before anything else; every other script fails with "Permission denied
  (keyboard-interactive)" without it.
- `scripts/della/preflight_hotpotqa.sh`: runbook steps 1-4 (prereqs, exact
  commit + clean tree, `.env`, BatchMode SSH, `cudatoolkit/13.0`,
  writable `MODEL_STORAGE`, home quota, each serving venv vs. its lock). Read-only.
- `scripts/della/build_env.sh [model ...]`: syncs, then on `della-vis1`
  (internet) runs `scripts/della/remote/setup_env.sh` (GEPA venv at
  `$REMOTE_DIR/.venv` plus one hash-locked vLLM serving venv per model:
  `.serving-venv` for Qwen, `.serving-venv-deepseek-v4.1-flash`),
  `remote/download_dataset.sh` (Wiki-2017 BM25 index, HotpotQA split), and,
  **detached**, `remote/download_model.sh <model>` for each model: downloads and
  byte-verifies the pinned checkpoint into `$MODEL_STORAGE`
  (DeepSeek-V4.1-Flash is 510.3 GB / 475.3 GiB in 48 shards; have ~600 GB free).
  Each remote script also runs alone from the synced checkout with
  `SCRATCH_BASE`/`MODEL_STORAGE` exported. Never run hours-long steps attached to
  a laptop ssh session.
- `scripts/della/submit_deepseek_smoke.sh submit|fetch <job-id>`: one-time
  DeepSeek serving smoke test as a batch job
  (`scripts/della/smoke_deepseek_serving.sbatch`, which execs
  `verify_deepseek_serving.sh`). Serves DeepSeek-V4.1-Flash with the sbatch's
  exact `vllm serve` flags, records one simple prompt's request, server-rendered
  prompt, reasoning, and content (`examples/hotpotqa/smoke_serving.py`), then
  checks an ordinary completion, tool-result continuation, and all four ReAct V2
  edit tools. `fetch` copies the results to `outputs/deepseek-smoke/<job-id>/`.
  Not wired into any launcher; no lock or marker files.
- `scripts/della/submit_hotpotqa.sh`: stages `git archive HEAD` under
  `$REMOTE_DIR/sources/<commit>`, verifies every artifact, then submits the
  `afterok` chain. Refuses a dirty tree; records HEAD as the source commit.
- `scripts/della/sync_to_della.sh`: code sync (called by the two above).
- `scripts/della/fetch_hotpotqa_results.sh`: pull runs/logs/locks/analysis
  into `outputs/hotpotqa-campaigns/<campaign>/<commit>/`.

Remote pieces (never call directly): `examples/hotpotqa/run_hotpotqa.sbatch`
serves the model through this repo's hash-locked vLLM venv (Qwen: TP1/DP8;
DeepSeek-V4.1-Flash: one TP8/EP8 replica), waits for health, runs GEPA,
tears down.

## HotPotQA campaign (Gilad's runbook)

Full text: `examples/hotpotqa/DELLA_CAMPAIGN.md`. Essentials:
- Launch from **exactly** the pinned commit, detached
  (`git switch --detach <commit>`), with a clean tree. Tooling lives on a
  separate branch; switching back and forth is fine because `.env` is ignored.
- Order per model arm: standard `vanilla`, `react_v2`, `react_v2_random`,
  `action` (6,871 calls) then expanded `vanilla`, `react_v2` (13,742 calls).
  Each arm (`MODEL_PROFILE=qwen3.8-27b` or `deepseek-v4.1-flash`) submits 6 jobs;
  there is no canary job. Run the DeepSeek smoke test once before the first
  DeepSeek submission. Arms are independent. Concurrent examples are 12 (Qwen,
  eight single-sequence replicas) and 4 (DeepSeek, one replica); each request has
  a 3,600 s timeout and two retries.
- Job names: `gepa-hp-<profile>-<standard|expanded>-<condition>`. Logs:
  `$SCRATCH_BASE/logs/hotpotqa/<campaign>/<commit>/hotpotqa-<job>-<id>.log`
  plus `gen-<id>.log` for the model server.
- Resume = resubmit the same commit/campaign/model with
  `BUDGET_PROFILE=standard|expanded CONDITION=<cell>`; GEPA reuses saved state.
  Cancel orphaned `afterok` dependents of a failed parent first.
- Never include a run that failed a preflight/integrity check.

## SSH (learned the hard way)

- Della requires **publickey AND keyboard-interactive** (password; Duo when
  off-campus). A key alone yields "Authenticated using publickey with partial
  success" then "Permission denied (keyboard-interactive)".
- Gilad's scripts use `BatchMode=yes` + `StrictHostKeyChecking=yes`, so they
  rely on **ControlMaster multiplexing**: `~/.ssh/config` sets
  `ControlMaster auto`, `ControlPath ~/.ssh/cm/%r@%h:%p`, `ControlPersist yes`
  for both hosts, and `della_session.sh open` authenticates once per host.
  `ssh -O check <host>` shows whether a master is alive.
- Both host keys must already be in `~/.ssh/known_hosts`.
- Other projects' Della scripts may use `sshpass` + password instead; do not
  mix the two styles in this repo.

## Node types

- Login (`REMOTE_HOST`, della.princeton.edu): brief ops only (rsync, sbatch,
  squeue, scontrol, sacct, checkquota).
- Vis (`REMOTE_VIS_HOST`, della-vis1): internet + CPU/RAM; builds, downloads,
  large file moves.
- GPU compute (`ailab` = H200 141 GB, 8 per node): **no internet**; the sbatch
  forces `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.

## SLURM specifics

- Never `--partition=gpu` (rejected by the submit filter). Use
  `GPU_PARTITION=ailab`; the campaign requires it and 8 GPUs per job.
- Inspect: `squeue -u $USER -o "%.18i %.60j %.2t %.12M %.30R"`,
  `scontrol show job <id> | tr ' ' '\n' | grep -E '^(JobName|JobState|Reason|Dependency)='`,
  `sacct -u $USER --starttime YYYY-MM-DD --format=JobIDRaw,JobName%64,State,ExitCode,Elapsed,Timelimit`.
- Time limits are caps, not estimates: Qwen standard 72 h, Qwen expanded
  144 h, DeepSeek jobs 144 h. 144 h is Della's maximum.

## Storage and quota (the #1 source of failures)

- `/home` quota ~48.8 GiB and ~1.9M files; check with `checkquota`. Keep all
  caches, venvs, outputs on scratch: `/scratch/gpfs/BSTEWART/<netid>/gepa`
  (`REMOTE_DIR` = `SCRATCH_BASE`). The scripts already export
  `XDG_CACHE_HOME`, `HF_HOME`, `UV_CACHE_DIR`, `DSPY_CACHEDIR` under scratch.
- Checkpoints live in the shared, group-writable
  `/projects/BSTEWART/model_storage` (`MODEL_STORAGE`); reference them as
  `${MODEL_STORAGE}/<name>`. Only `remote/download_model.sh` may populate it (it writes the
  `.gepa-model-integrity.json` manifests the launcher demands).
- Scratch is not backed up and is purged periodically.

## Serving environments (self-contained, one per arm)

Each arm serves through a plain uv venv this repo builds itself (no containers,
no other project's venv): Qwen from `examples/hotpotqa/serving/requirements.in`
and its lock `requirements-x86_64-linux-py312.txt` into `$REMOTE_DIR/.serving-venv`;
DeepSeek from `requirements-deepseek-v4.1-flash.in` and its lock
`requirements-deepseek-v4.1-flash-x86_64-linux-py312.txt` into
`$REMOTE_DIR/.serving-venv-deepseek-v4.1-flash`. Regenerate a lock with
`MODEL_PROFILE=<arm> scripts/della/lock_serving_env.sh`. `remote/setup_env.sh` installs
each with `uv pip sync --require-hashes` (Python 3.12.7), applies the cutlass-DSL
reinstall-order fix, and freezes a manifest under
`$SCRATCH_BASE/.cache/gepa/serving-environments/<lock-sha256>.json`. The launcher
and the sbatch record the lock's sha256 (`HOTPOTQA_SERVING_LOCK_SHA256`) and the
realized manifest digest (`HOTPOTQA_SERVING_ENV_SHA256`), and refuse to run if
the venv was built from a different lock. `sync_to_della.sh` excludes both venv
directories, so its `rsync --delete` never removes them.

Qwen keeps vLLM 0.25.1 / torch 2.11. DeepSeek-V4.1-Flash (`DeepseekV41ForCausalLM`)
is newer than every vLLM release, so its lock pins the per-commit wheel for vLLM
main `e77daef89` (from wheels.vllm.ai; version string `0.1.1.dev5+ge77daef89`),
torch 2.13 (CUDA 13.0), and flashinfer 0.6.18.post1 plus its prebuilt
`flashinfer-cubin` and `flashinfer-jit-cache` wheels from flashinfer.ai: from vLLM
0.26 flashinfer otherwise downloads kernels at runtime, which fails on the
offline GPU nodes, and the job sets `FLASHINFER_NO_DOWNLOAD=1`. DeepSeek flags:
`--tokenizer-mode deepseek_v41`, `deepseek_v41` reasoning and tool parsers,
TP8/EP8, FP8 `fp8_ds_mla` KV cache, KV block size left to vLLM (64 on SM90),
`VLLM_ENGINE_READY_TIMEOUT_S=3600`. FP4 experts use vLLM's Marlin MoE backend on
H200. Chosen and shared with Qwen: `--max-num-seqs 1`, `--seed 0`, no prefix
caching, one API server, no speculative decoding (MTP/DSpark not loaded).
Thinking mode and `reasoning_effort=100` go through `chat_template_kwargs`
(`experiment_models.py`). The sbatch still fails closed by checking the
checkpoint's `architectures` against vLLM's `ModelRegistry`. The sbatch puts the
venv's `nvidia/cu13` headers first on `CPATH` because at least one ailab node's
local CUDA install lacks `cublasLt.h`, which FlashInfer's JIT builds include.
