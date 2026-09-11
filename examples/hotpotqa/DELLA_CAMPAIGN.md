<!-- Source: Gilad Morad, https://gist.github.com/gilad12-coder/b0142ad28de98c47487ea9847686206a (fetched 2026-09-02).
     Adapted in this checkout: the second arm is DeepSeek-V4.1-Flash (GLM-5.3-Flash is dropped); neither arm
     depends on an external POSIT checkout or any container; this repo builds one hash-locked vLLM serving venv per
     arm (section 4); and there is no canary job, only a one-time serving smoke test (section 5). Everything else
     follows the gist. -->

# Running the HotPotQA campaign on Della

Zach,

The code is in [PR #59](https://github.com/zbambergerNLP/gepa/pull/59), and [Graphite](https://app.graphite.dev/github/pr/zbambergerNLP/gepa/59) has the review view. Use this commit for the first campaign:

```text
169ddda125b1abe305c7714bbb5b3fc38b21b587
```

We are running six configurations with Qwen3.8-27B and six with DeepSeek-V4.1-Flash (`deepseek-ai/DeepSeek-V4.1-Flash`, released 2026-09-10). Each run uses one model for the student, proposer, and Controller.

## 1. Run the launcher locally

Run the launcher from your laptop or another machine that can SSH non-interactively to both Della hosts. Do not invoke `examples/hotpotqa/run_hotpotqa.sbatch` directly.

The launcher uses each machine as follows:

- Your local machine freezes and syncs the source commit, submits jobs, and fetches results.
- `della-vis1` downloads and verifies the internet-dependent artifacts.
- The Della login node calls `sbatch`.
- The allocated eight-H200 Slurm nodes serve the local models and run GEPA.

Model inference and GEPA both run on Della. The local script handles submission and file transfer.

Local prerequisites:

```bash
command -v git
command -v ssh
command -v rsync
command -v sha256sum
command -v uv
```

Your SSH agent or `~/.ssh/config` must already provide access. Visit both hosts once so their verified host keys are present in `~/.ssh/known_hosts`:

```bash
ssh YOUR_NETID@della.princeton.edu exit
ssh YOUR_NETID@della-vis1.princeton.edu exit
```

The scripts use `BatchMode=yes` and `StrictHostKeyChecking=yes`. A launch will fail if either host needs a password prompt or has an unknown host key.

## 2. Check out the experiment source

For a fresh checkout:

```bash
git clone https://github.com/zbambergerNLP/gepa.git gepa-hotpotqa
cd gepa-hotpotqa

git fetch origin codex/review-current-07-hotpotqa-slurm
git switch --detach 169ddda125b1abe305c7714bbb5b3fc38b21b587
```

For an existing checkout, check for local work before switching commits:

```bash
cd /path/to/gepa
git status --short
git fetch origin codex/review-current-07-hotpotqa-slurm
git switch --detach 169ddda125b1abe305c7714bbb5b3fc38b21b587
```

Check the commit and worktree:

```bash
test "$(git rev-parse HEAD)" = "169ddda125b1abe305c7714bbb5b3fc38b21b587"
test -z "$(git status --porcelain --untracked-files=normal)"
echo "Source is exact and clean."
```

The launcher rejects a dirty checkout.

## 3. Configure the Della paths

Create the ignored local configuration file:

```bash
cp scripts/della/.env.example scripts/della/.env
chmod 600 scripts/della/.env
```

Edit `scripts/della/.env` and replace every placeholder:

```bash
REMOTE_USER="YOUR_NETID"
REMOTE_HOST="della.princeton.edu"
REMOTE_VIS_HOST="della-vis1.princeton.edu"
REMOTE_DIR="/scratch/gpfs/YOUR_ALLOCATION/YOUR_NETID/gepa"
SCRATCH_BASE="/scratch/gpfs/YOUR_ALLOCATION/YOUR_NETID/gepa"
MODEL_STORAGE="/projects/YOUR_ALLOCATION/model_storage"
GPU_PARTITION="ailab"
```

For the existing BSTEWART shared model location, use:

```bash
MODEL_STORAGE="/projects/BSTEWART/model_storage"
```

`MODEL_STORAGE` must be writable while the artifacts are first prepared and readable from the visualization, login, and compute nodes.

Recheck the file mode:

```bash
test "$(stat -f '%Lp' scripts/della/.env 2>/dev/null || stat -c '%a' scripts/della/.env)" = "600"
```

## 4. Check the serving prerequisites

Each arm is served by its own vLLM environment that this repository builds itself, as a
plain uv venv with every package hash-locked (no container, no other project's checkout):

| Arm | Requirements | Lock | Venv |
|---|---|---|---|
| Qwen3.8-27B | `serving/requirements.in` | `serving/requirements-x86_64-linux-py312.txt` | `$REMOTE_DIR/.serving-venv` |
| DeepSeek-V4.1-Flash | `serving/requirements-deepseek-v4.1-flash.in` | `serving/requirements-deepseek-v4.1-flash-x86_64-linux-py312.txt` | `$REMOTE_DIR/.serving-venv-deepseek-v4.1-flash` |

(paths under `examples/hotpotqa/`). Regenerate a lock with
`MODEL_PROFILE=<arm> scripts/della/lock_serving_env.sh` after changing its `.in` file, and
commit the lock.

Qwen keeps the proven vLLM 0.25.1 / torch 2.11 stack. DeepSeek-V4.1-Flash
(`DeepseekV41ForCausalLM`) is newer than every vLLM release, so its environment pins the
wheel vLLM publishes for main commit `e77daef89` (the commit that added the model, its
`deepseek_v41` tokenizer mode, and its reasoning and tool-call parsers), with torch 2.13
(CUDA 13.0) and flashinfer 0.6.18.post1. Della's GPU nodes are offline, and from vLLM 0.26
flashinfer downloads missing GPU kernels at runtime, so the lock also installs flashinfer's
prebuilt `flashinfer-cubin` and `flashinfer-jit-cache` wheels from flashinfer.ai and the job
sets `FLASHINFER_NO_DOWNLOAD=1`. Every job still fails closed at the architecture check in
`run_hotpotqa.sbatch` if the frozen vLLM does not register the checkpoint's architecture.

The DeepSeek arm serves one TP8/EP8 replica per node with the same single-sequence
determinism contract as Qwen (`--max-num-seqs 1`, `--seed 0`, no prefix caching, no
batch-invariant mode, one API server, no speculative decoding: neither the MTP layers nor
the DSpark draft module is loaded). Prompts use the checkpoint's native encoding
(`--tokenizer-mode deepseek_v41`; the repository ships no Jinja template), with thinking on
and reasoning effort 100, the maximum. The KV cache is FP8 (`fp8_ds_mla`, the layout its
sparse-MLA path uses) and the KV block size is left to vLLM (its SM90 sparse-MLA kernel
needs 64-token blocks). The checkpoint ships FP8 dense weights, FP4 experts, and FP8 Engram
tables; on the H200 (SM90) the FP4 experts run on vLLM's Marlin MoE backend.

Concurrency: every replica decodes one sequence at a time, so each concurrent example
beyond the replica count only queues. The launcher runs 12 concurrent examples on Qwen's
eight replicas and 4 on DeepSeek's single replica, and each request has a 3,600-second
timeout with two retries of the identical seeded request.

Run the read-only preflight from your laptop; it covers steps 1-4:

```bash
scripts/della/preflight_hotpotqa.sh
```

It checks the local tools, the exact commit and clean tree, `scripts/della/.env`,
non-interactive SSH to both hosts, and on `della-vis1`: the `cudatoolkit/13.0` module, a
writable `MODEL_STORAGE`, the home quota, and (once built) that the serving venv matches
the committed lock, for each arm's serving venv.

Della requires an SSH key **and** a password step; the launcher scripts use `BatchMode=yes`,
so open the persistent master connections once per laptop session:

```bash
scripts/della/della_session.sh open
```

## 5. Build the shared artifacts

Run this once from the repository on your laptop:

```bash
scripts/della/build_env.sh
```

`build_env.sh` runs the downloads on `della-vis1`. It also:

- builds the frozen Python 3.11.13 and uv 0.9.13 GEPA environment;
- builds each arm's hash-locked vLLM serving venv (table above) and freezes its manifest;
- builds and verifies the frozen Wiki-2017 BM25 index;
- caches the exact HotPotQA 150/300/300 split; and
- downloads and byte-verifies both pinned model checkpoints.

Disk prerequisite: the DeepSeek-V4.1-Flash checkpoint is 48 safetensors shards
totalling 510.3 GB (475.3 GiB) at the pinned revision, on top of Qwen3.8-27B. Check
that `MODEL_STORAGE` has roughly 600 GB free before the first build (`checkquota` or
`df -h "$MODEL_STORAGE"` on `della-vis1`); the download also stages transfer metadata
under `.cache` inside the checkpoint directory.

The shared files will be at:

```text
$MODEL_STORAGE/Qwen3.8-27B
$MODEL_STORAGE/DeepSeek-V4.1-Flash
```

Model revisions:

```text
Qwen/Qwen3.8-27B
revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0

deepseek-ai/DeepSeek-V4.1-Flash
revision dba1be0a40aa45a94ad051997016db3960a90277
```

Use `build_env.sh` rather than `huggingface-cli download`. The launcher requires the `.gepa-model-integrity.json` manifests generated during this build. Existing valid files are reused and verified.

After the build succeeds, check that both manifests are present:

```bash
source scripts/della/.env

ssh "${REMOTE_USER}@${REMOTE_VIS_HOST}" \
  "test -s '${MODEL_STORAGE}/Qwen3.8-27B/.gepa-model-integrity.json' &&
   test -s '${MODEL_STORAGE}/DeepSeek-V4.1-Flash/.gepa-model-integrity.json' &&
   echo 'All pinned model artifacts are present.'"
```

The byte-level verification runs inside `build_env.sh`.

### Smoke-test the DeepSeek serving stack once

Before submitting the DeepSeek chain the first time, run the serving smoke test. It is not
part of any campaign job, is not a dependency of one, and writes no marker or lock files.
It serves the checkpoint with exactly the `vllm serve` invocation the sbatch uses, waits
for health, then sends one simple prompt with the campaign's request settings and records
the request, the prompt text the server rendered from it (via vLLM's `/tokenize` and
`/detokenize`), and the full response including the reasoning tokens. It then exercises an
ordinary completion, a native tool call plus its tool-result continuation, and one ReAct V2
proposal per edit tool (DELETE_TEXT, INSERT_TEXT, MOVE_TEXT, REPLACE_TEXT). From your laptop:

```bash
scripts/della/submit_deepseek_smoke.sh submit          # prints the job id
scripts/della/submit_deepseek_smoke.sh fetch <job-id>  # after it finishes
```

The fetch copies `outputs/deepseek-smoke/<job-id>/`: `transcript.md` (human-readable),
`transcript.json`, `verify_report.txt`, `serving-packages.txt`, `gpus.txt`, `vllm.log`, and
the Slurm log. The same check runs interactively on an allocated node with
`scripts/della/verify_deepseek_serving.sh`. `VERIFY_ATTEMPTS` (default 4, one per tool)
raises the number of edit attempts for a longer soak. Do not submit the DeepSeek chain
until it prints `PASS`.

## 6. Experiment matrix

The launcher enforces this order separately for Qwen and DeepSeek:

| Order | Tree | Code condition | Metric-call budget |
|---:|---|---|---:|
| 1 | Standard | `vanilla` | 6,871 |
| 2 | Standard | `react_v2` | 6,871 |
| 3 | Standard | `react_v2_random` | 6,871 |
| 4 | Standard | `action` | 6,871 |
| 5 | Expanded | `vanilla` | 13,742 |
| 6 | Expanded | `react_v2` | 13,742 |

Each job uses `afterok` on the previous job. Within a model arm, the four standard-tree runs finish before either expanded-tree run starts. The Qwen and DeepSeek arms are independent and can occupy two nodes at once.

Every job, Qwen or DeepSeek, verifies the frozen serving environment, checks that the pinned vLLM registers the checkpoint's architecture, and runs a native tool-call check before optimization. There is no separate canary job; the one-time manual serving check is described in section 5.

## 7. Submit the jobs

Launch the Qwen chain:

```bash
MODEL_PROFILE=qwen3.8-27b \
BUDGET_PROFILE=campaign \
CONDITION=all \
HOTPOTQA_CAMPAIGN_ID=hotpotqa-final-v1 \
scripts/della/submit_hotpotqa.sh
```

Launch the DeepSeek chain (after the one-time manual serving check in section 5 has passed):

```bash
MODEL_PROFILE=deepseek-v4.1-flash \
BUDGET_PROFILE=campaign \
CONDITION=all \
HOTPOTQA_CAMPAIGN_ID=hotpotqa-final-v1 \
scripts/della/submit_hotpotqa.sh
```

Each command submits six jobs. The campaign produces 12 result cells.

The launcher stages the source at:

```text
$REMOTE_DIR/sources/169ddda125b1abe305c7714bbb5b3fc38b21b587
```

The jobs load the checkpoints from Della storage and set Hugging Face and Transformers to offline mode. They do not use OpenRouter or another hosted model API.

The Slurm time limits are safety caps, not runtime estimates: Qwen standard jobs receive 72 hours, Qwen expanded jobs 144 hours, and DeepSeek campaign jobs 144 hours.

## 8. Monitor the jobs

From a Della shell:

```bash
squeue -u "$USER" -o "%.18i %.60j %.2t %.12M %.30R"
```

Qwen job order:

```text
gepa-hp-qwen3.8-27b-standard-vanilla
gepa-hp-qwen3.8-27b-standard-react_v2
gepa-hp-qwen3.8-27b-standard-react_v2_random
gepa-hp-qwen3.8-27b-standard-action
gepa-hp-qwen3.8-27b-expanded-vanilla
gepa-hp-qwen3.8-27b-expanded-react_v2
```

The six DeepSeek jobs use the same standard-first order with the `deepseek-v4.1-flash` profile.

Inspect the dependency and state of one job:

```bash
scontrol show job JOB_ID | tr ' ' '\n' | grep -E '^(JobName|JobState|Reason|Dependency)='
```

After jobs leave the queue, inspect accounting records. Replace the date with the actual submission date:

```bash
sacct -u "$USER" --starttime YYYY-MM-DD \
  --format=JobIDRaw,JobName%64,State,ExitCode,Elapsed,Timelimit
```

Logs are written under:

```text
$SCRATCH_BASE/logs/hotpotqa/hotpotqa-final-v1/169ddda125b1abe305c7714bbb5b3fc38b21b587/
```

List them:

```bash
source scripts/della/.env

find "${SCRATCH_BASE}/logs/hotpotqa/hotpotqa-final-v1/169ddda125b1abe305c7714bbb5b3fc38b21b587" \
  -maxdepth 1 -type f -print | sort
```

Each Slurm job has a `hotpotqa-<job-name>-<job-id>.log`; the corresponding local model server has `gen-<job-id>.log`.

## 9. Resume a failed or preempted run

Keep the existing output. Resubmit the same source commit, campaign ID, model, budget profile, and condition. GEPA will reuse its saved state, response journal, evaluation cache, and held-out checkpoints.

Example: resume Qwen standard ReAct V2:

```bash
MODEL_PROFILE=qwen3.8-27b \
BUDGET_PROFILE=standard \
CONDITION=react_v2 \
HOTPOTQA_CAMPAIGN_ID=hotpotqa-final-v1 \
scripts/della/submit_hotpotqa.sh
```

Example: resume DeepSeek expanded ReAct V2:

```bash
MODEL_PROFILE=deepseek-v4.1-flash \
BUDGET_PROFILE=expanded \
CONDITION=react_v2 \
HOTPOTQA_CAMPAIGN_ID=hotpotqa-final-v1 \
scripts/della/submit_hotpotqa.sh
```

When an `afterok` parent fails, its old dependent jobs cannot run. Cancel those job IDs and submit the missing runs explicitly. Do not submit the `.sbatch` file directly.

## 10. Fetch the results

Once all result cells have completed, run this from the same local checkout:

```bash
HOTPOTQA_SOURCE_COMMIT=169ddda125b1abe305c7714bbb5b3fc38b21b587 \
HOTPOTQA_CAMPAIGN_ID=hotpotqa-final-v1 \
scripts/della/fetch_hotpotqa_results.sh
```

The local result directory is:

```text
outputs/hotpotqa-campaigns/hotpotqa-final-v1/169ddda125b1abe305c7714bbb5b3fc38b21b587/
```

It contains:

```text
runs/
logs/
campaign-locks/
hotpotqa_analysis.json
```

The command prints validation EM, test EM, test F1, call and candidate counts, candidate and proposal Jaccard diversity, action usage and acceptance, action entropy, Controller entropy, fallback rate, and tail-sampling rate.

The analyzer accepts a completed subset. Check that all campaign cells are present before comparing results:

```bash
python3 - <<'PY'
import json
from pathlib import Path

commit = "169ddda125b1abe305c7714bbb5b3fc38b21b587"
analysis = (
    Path("outputs/hotpotqa-campaigns/hotpotqa-final-v1")
    / commit
    / "hotpotqa_analysis.json"
)
runs = json.loads(analysis.read_text())["runs"]

models = {
    "hosted_vllm/Qwen/Qwen3.8-27B",
    "hosted_vllm/deepseek-ai/DeepSeek-V4.1-Flash",
}
cells = {
    (6871, "vanilla"),
    (6871, "react_v2"),
    (6871, "react_v2_random"),
    (6871, "action"),
    (13742, "vanilla"),
    (13742, "react_v2"),
}
expected = {
    (model, budget, condition)
    for model in models
    for budget, condition in cells
}
actual = {
    (run["model"], run["max_metric_calls"], run["condition"])
    for run in runs
}

missing = sorted(expected - actual)
extra = sorted(actual - expected)
if missing or extra:
    raise SystemExit(
        f"Campaign is incomplete. Missing={missing}; extra={extra}"
    )
print("All 12 HotPotQA campaign cells completed.")
PY
```

Print a compact table:

```bash
jq -r '
  .runs[]
  | [.model_label, .budget_profile, .condition,
     .best_validation_exact_match, .test_exact_match, .test_f1]
  | @tsv
' \
outputs/hotpotqa-campaigns/hotpotqa-final-v1/169ddda125b1abe305c7714bbb5b3fc38b21b587/hotpotqa_analysis.json
```

Paired confidence intervals are not implemented. Ignore the absolute `path` fields in the JSON because they point to temporary fetch directories. Use the result directory shown above. The reported scores and mechanism statistics are valid.

## 11. Final check

Confirm these items before using the results:

- The source commit is exactly `169ddda125b1abe305c7714bbb5b3fc38b21b587`.
- `build_env.sh` completed without an integrity error.
- The Qwen chain produced six successful Slurm jobs.
- The DeepSeek chain produced six successful Slurm jobs.
- No relevant job ended in `FAILED`, `TIMEOUT`, `OUT_OF_MEMORY`, or `CANCELLED`.
- The completeness script prints `All 12 HotPotQA campaign cells completed.`
- `hotpotqa_analysis.json`, run artifacts, campaign locks, and logs were fetched locally.

If a preflight or integrity check fails, keep the error and logs and stop. The failed run should not be included in the comparison.
