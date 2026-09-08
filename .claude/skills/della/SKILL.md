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
  writable `MODEL_STORAGE`, home quota, serving venv vs. lock). Read-only.
- `scripts/della/build_env.sh`: one-time on `della-vis1` (internet). Builds
  the frozen GEPA venv at `$REMOTE_DIR/.venv`, builds the hash-locked vLLM
  serving venv at `$REMOTE_DIR/.serving-venv` (shared by both model arms),
  builds/verifies the Wiki-2017 BM25 index, caches the HotPotQA split, and
  **downloads and byte-verifies both pinned checkpoints into
  `$MODEL_STORAGE`** (DeepSeek-V4-Flash-0731 alone is ~167 GB / 155 GiB in
  48 shards; have ~200 GB free there first). Hours.
- `scripts/della/verify_deepseek_serving.sh`: manual, one-time, on an
  allocated eight-H200 node (`salloc --partition=ailab --gres=gpu:8 ...`,
  then `cd $REMOTE_DIR && scripts/della/verify_deepseek_serving.sh`). Serves
  DeepSeek-V4-Flash-0731 with the sbatch's exact `vllm serve` flags, waits
  for health, then checks an ordinary completion, tool-result continuation,
  and all four ReAct V2 edit tools; prints PASS/FAIL, exits non-zero on
  failure. Not wired into any launcher; no lock or marker files.
- `scripts/della/submit_hotpotqa.sh`: stages `git archive HEAD` under
  `$REMOTE_DIR/sources/<commit>`, verifies every artifact, then submits the
  `afterok` chain. Refuses a dirty tree; records HEAD as the source commit.
- `scripts/della/sync_to_della.sh`: code sync (called by the two above).
- `scripts/della/fetch_hotpotqa_results.sh`: pull runs/logs/locks/analysis
  into `outputs/hotpotqa-campaigns/<campaign>/<commit>/`.

Remote pieces (never call directly): `examples/hotpotqa/run_hotpotqa.sbatch`
serves the model through this repo's hash-locked vLLM venv (Qwen: TP1/DP8;
DeepSeek-V4-Flash-0731: one TP8/EP8 replica), waits for health, runs GEPA,
tears down.

## HotPotQA campaign (Gilad's runbook)

Full text: `examples/hotpotqa/DELLA_CAMPAIGN.md`. Essentials:
- Launch from **exactly** the pinned commit, detached
  (`git switch --detach <commit>`), with a clean tree. Tooling lives on a
  separate branch; switching back and forth is fine because `.env` is ignored.
- Order per model arm: standard `vanilla`, `react_v2`, `react_v2_random`,
  `action` (6,871 calls) then expanded `vanilla`, `react_v2` (13,742 calls).
  Each arm (`MODEL_PROFILE=qwen3.8-27b` or `deepseek-v4-flash`) submits 6 jobs;
  there is no canary job. Run `verify_deepseek_serving.sh` once by hand before
  the first DeepSeek submission. Arms are independent.
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
  `${MODEL_STORAGE}/<name>`. Only `build_env.sh` may populate it (it writes the
  `.gepa-model-integrity.json` manifests the launcher demands).
- Scratch is not backed up and is purged periodically.

## Serving environment (self-contained, both arms)

Both arms serve through a vLLM venv this repo builds itself:
`examples/hotpotqa/serving/requirements.in` (direct pins) and the hash-locked
`requirements-x86_64-linux-py312.txt` (every transitive package, generated by
`scripts/della/lock_serving_env.sh`). `build_env.sh` installs it with
`uv pip sync --require-hashes` into `$REMOTE_DIR/.serving-venv` (Python 3.12.7),
applies the cutlass-DSL reinstall-order fix, and freezes a manifest under
`$SCRATCH_BASE/.cache/gepa/serving-environments/<lock-sha256>.json`. The
launcher and the sbatch record the lock's sha256 (`HOTPOTQA_SERVING_LOCK_SHA256`)
and the realized manifest digest (`HOTPOTQA_SERVING_ENV_SHA256`), and refuse to
run if the venv was built from a different lock. Changing `requirements.in`
means re-running the lock script, committing the lock, and re-running
`build_env.sh`. Nothing outside this checkout (no other project's venv or git
state) is consulted.

The pinned vLLM 0.25.1 already registers DeepSeek V4 (`DeepseekV4ForCausalLM`,
the `deepseek_v4` reasoning/tool parsers, native prompt encoding), so the
DeepSeek arm needs no pin, FlashInfer, or torch change. DeepSeek serving facts
that are forced by vLLM, not chosen: FP8 `fp8_ds_mla` KV cache (the only layout
its sparse-MLA path supports), 256-token KV blocks, and no DeepGEMM MegaMoE on
H200 (SM100 only; the checkpoint's MXFP4 experts use vLLM's Hopper MXFP4 MoE
fallback). Chosen and shared with Qwen: `--max-num-seqs 1`, `--seed 0`, no
prefix caching, one API server, no speculative decoding (MTP/DSpark not
loaded). Thinking mode and `reasoning_effort=max` go through
`chat_template_kwargs` (`experiment_models.py`). The sbatch still fails closed
by checking the checkpoint's `architectures` against vLLM's `ModelRegistry`.
