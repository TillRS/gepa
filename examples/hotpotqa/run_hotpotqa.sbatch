#!/bin/bash
# Run the GEPA HotpotQA evaluation on della.
#
# Serves the selected local checkpoint through its pinned OpenAI-compatible
# runtime, then runs the GEPA optimization experiment against it.
#
# Prerequisites:
#   1. GEPA repo synced and all pinned artifacts built with build_env.sh
#   2. Eight H200 GPUs allocated by scripts/della/submit_hotpotqa.sh
#
# Submit through scripts/della/submit_hotpotqa.sh. It supplies the complete
# model-specific resource request and is the only supported production entry.
#
# The production path fixes every scientific setting. BUDGET_PROFILE identifies
# the standard 6,871-call or two-times 13,742-call search budget for one approved
# campaign cell. Throughput settings may change scheduling, but not prompts,
# data, retrieval, decoding, candidate budgets, or evaluation order.

#SBATCH --job-name=gepa-hotpotqa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=gepa-hotpotqa-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1

if [[ "${HOTPOTQA_PRODUCTION_LAUNCH:-}" != "1" ]]; then
    echo "ERROR: submit this production job through scripts/della/submit_hotpotqa.sh" >&2
    exit 1
fi

MODEL_STORAGE="${MODEL_STORAGE:-/projects/BSTEWART/model_storage}"
MODEL_PROFILE="${MODEL_PROFILE:-qwen3.8-27b}"
BUDGET_PROFILE="${BUDGET_PROFILE:-standard}"
CONDITION="${CONDITION:-}"
HOTPOTQA_CAMPAIGN_ID="${HOTPOTQA_CAMPAIGN_ID:-}"
if [[ ! "${HOTPOTQA_CAMPAIGN_ID}" =~ ^[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?$ ]]; then
    echo "ERROR: production launcher did not provide a valid HotPotQA campaign ID" >&2
    exit 1
fi
GEN_PORT="${GEN_PORT:-}"
MAX_WORKERS="${MAX_WORKERS:-32}"
GEN_GMU=0.92
GEN_MAX_LEN=262144
VLLM_DATA_PARALLEL_SIZE="${VLLM_DATA_PARALLEL_SIZE:-1}"
VLLM_API_SERVER_COUNT="${VLLM_API_SERVER_COUNT:-${VLLM_DATA_PARALLEL_SIZE}}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1800}"
# Self-contained vLLM serving environment built by scripts/della/build_env.sh. Each
# model profile has its own hash-locked environment (SERVING_LOCK, set per profile below).
SERVING_VENV_DIR="${SERVING_VENV_DIR:-}"
GEPA_VENV_DIR="${GEPA_VENV_DIR:-}"
SCRATCH_BASE="${SCRATCH_BASE:-/scratch/gpfs/BSTEWART/${USER}/gepa}"
EXPECTED_HOTPOTQA_PYTHON_VERSION="3.11.13"
HOTPOTQA_PYTHON_VERSION="${HOTPOTQA_PYTHON_VERSION:-}"
EXPECTED_HOTPOTQA_UV_VERSION="0.9.13"
HOTPOTQA_UV_VERSION="${HOTPOTQA_UV_VERSION:-}"
HOTPOTQA_UV_SHA256="${HOTPOTQA_UV_SHA256:-}"
GEPA_UV_BIN="${GEPA_UV_BIN:-}"
QWEN_REVISION="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
# deepseek-ai/DeepSeek-V4.1-Flash main as of 2026-09-10, the release commit.
DEEPSEEK_REVISION="dba1be0a40aa45a94ad051997016db3960a90277"
HOTPOTQA_DATA_REVISION="1908d6afbbead072334abe2965f91bd2709910ab"
WIKI17_REVISION="ef6a5e72a98b47cef31574a400fea8fe149559a3"
HOTPOTQA_DSPY_COMMIT="62dc3b634d7dc0c4889abcf905cb4c391ea6b396"
HOTPOTQA_SOURCE_COMMIT="${HOTPOTQA_SOURCE_COMMIT:-}"
HOTPOTQA_SOURCE_MANIFEST_SHA256="${HOTPOTQA_SOURCE_MANIFEST_SHA256:-}"
HOTPOTQA_ENV_SPEC_SHA256="${HOTPOTQA_ENV_SPEC_SHA256:-}"
HOTPOTQA_GEPA_ENV_SHA256="${HOTPOTQA_GEPA_ENV_SHA256:-}"
HOTPOTQA_SERVING_ENV_SHA256="${HOTPOTQA_SERVING_ENV_SHA256:-}"
if [[ ! "${HOTPOTQA_SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: production launcher did not provide an exact source commit" >&2
    exit 1
fi
if [[ ! -f .gepa-source-commit || "$(tr -d '\n' < .gepa-source-commit)" != "${HOTPOTQA_SOURCE_COMMIT}" ]]; then
    echo "ERROR: Slurm job source does not match ${HOTPOTQA_SOURCE_COMMIT}" >&2
    exit 1
fi
if [[ ! "${HOTPOTQA_SOURCE_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ \
    || ! -f .gepa-source-manifest.sha256 \
    || "$(tr -d '\n' < .gepa-source-manifest.sha256)" != "${HOTPOTQA_SOURCE_MANIFEST_SHA256}" \
    || ! -f .gepa-source-manifest.sha256sums \
    || "$(sha256sum .gepa-source-manifest.sha256sums | cut -d' ' -f1)" != "${HOTPOTQA_SOURCE_MANIFEST_SHA256}" ]]; then
    echo "ERROR: Slurm job source bytes do not match the submitted source manifest" >&2
    exit 1
fi
if ! sha256sum --check .gepa-source-manifest.sha256sums >/dev/null; then
    echo "ERROR: Slurm job source bytes do not match the submitted source manifest" >&2
    exit 1
fi
ARTIFACT_LOCK_PATH="${SCRATCH_BASE}/.cache/gepa/hotpotqa-artifacts.lock"
mkdir -p "$(dirname "${ARTIFACT_LOCK_PATH}")"
exec {ARTIFACT_LOCK_FD}>"${ARTIFACT_LOCK_PATH}"
if ! flock -s -n "${ARTIFACT_LOCK_FD}"; then
    echo "ERROR: HotPotQA artifacts are being rebuilt; submit again after the build finishes" >&2
    exit 1
fi
if [[ -z "${GEPA_VENV_DIR}" || "${GEPA_VENV_DIR}" != /* ]]; then
    echo "ERROR: production launcher did not provide an absolute GEPA environment path" >&2
    exit 1
fi
export UV_PROJECT_ENVIRONMENT="${GEPA_VENV_DIR}"
CURRENT_ENV_SPEC_SHA256="$(
    {
        sha256sum pyproject.toml uv.lock
        printf 'python=%s\n' "${EXPECTED_HOTPOTQA_PYTHON_VERSION}"
        printf 'uv=%s\n' "${EXPECTED_HOTPOTQA_UV_VERSION}"
    } | sha256sum | cut -d' ' -f1
)"
if [[ ! "${HOTPOTQA_ENV_SPEC_SHA256}" =~ ^[0-9a-f]{64}$ \
    || "${CURRENT_ENV_SPEC_SHA256}" != "${HOTPOTQA_ENV_SPEC_SHA256}" \
    || ! -f "${GEPA_VENV_DIR}/.gepa-env-spec.sha256" \
    || "$(tr -d '\n' < "${GEPA_VENV_DIR}/.gepa-env-spec.sha256")" != "${HOTPOTQA_ENV_SPEC_SHA256}" \
    || "${HOTPOTQA_PYTHON_VERSION}" != "${EXPECTED_HOTPOTQA_PYTHON_VERSION}" \
    || ! -f "${GEPA_VENV_DIR}/.gepa-python-version" \
    || "$(tr -d '\n' < "${GEPA_VENV_DIR}/.gepa-python-version")" != "${HOTPOTQA_PYTHON_VERSION}" \
    || "$("${GEPA_VENV_DIR}/bin/python" -c 'import platform; print(platform.python_version())')" != "${HOTPOTQA_PYTHON_VERSION}" \
    || "${HOTPOTQA_UV_VERSION}" != "${EXPECTED_HOTPOTQA_UV_VERSION}" \
    || ! "${HOTPOTQA_UV_SHA256}" =~ ^[0-9a-f]{64}$ \
    || ! -x "${GEPA_UV_BIN}" \
    || ! -f "${GEPA_VENV_DIR}/.gepa-uv-version" \
    || "$(tr -d '\n' < "${GEPA_VENV_DIR}/.gepa-uv-version")" != "${HOTPOTQA_UV_VERSION}" \
    || "$("${GEPA_UV_BIN}" --version)" != "uv ${HOTPOTQA_UV_VERSION}"* \
    || ! -f "${GEPA_VENV_DIR}/.gepa-uv-sha256" \
    || "$(tr -d '\n' < "${GEPA_VENV_DIR}/.gepa-uv-sha256")" != "${HOTPOTQA_UV_SHA256}" \
    || "$(sha256sum "${GEPA_UV_BIN}" | cut -d' ' -f1)" != "${HOTPOTQA_UV_SHA256}" ]]; then
    echo "ERROR: GEPA environment does not match the staged dependency lock" >&2
    exit 1
fi
if ! "${GEPA_UV_BIN}" sync --python "${HOTPOTQA_PYTHON_VERSION}" --frozen --check --no-install-project \
    --extra dev --extra wiki17 --group hotpotqa-task-program; then
    echo "ERROR: GEPA environment has drifted from uv.lock" >&2
    exit 1
fi
export PYTHONPATH="${PWD}/src:${PWD}"
GEPA_ENV_MANIFEST="${SCRATCH_BASE}/.cache/gepa/python-environments/gepa-${HOTPOTQA_ENV_SPEC_SHA256}.json"
VERIFIED_GEPA_ENV_SHA256="$(
    "${GEPA_VENV_DIR}/bin/python" -m examples.common.python_environment verify --path "${GEPA_ENV_MANIFEST}"
)"
if [[ ! "${HOTPOTQA_GEPA_ENV_SHA256}" =~ ^[0-9a-f]{64}$ \
    || "${VERIFIED_GEPA_ENV_SHA256}" != "${HOTPOTQA_GEPA_ENV_SHA256}" ]]; then
    echo "ERROR: realized GEPA environment differs from the submitted frozen environment" >&2
    exit 1
fi

case "${BUDGET_PROFILE}" in
    standard)
        EXPECTED_MAX_METRIC_CALLS=6871
        ;;
    expanded)
        EXPECTED_MAX_METRIC_CALLS=13742
        ;;
    *)
        echo "ERROR: BUDGET_PROFILE must be standard or expanded" >&2
        exit 1
        ;;
esac
if [[ -n "${MAX_METRIC_CALLS:-}" && "${MAX_METRIC_CALLS}" != "${EXPECTED_MAX_METRIC_CALLS}" ]]; then
    echo "ERROR: MAX_METRIC_CALLS must match BUDGET_PROFILE=${BUDGET_PROFILE} (${EXPECTED_MAX_METRIC_CALLS})" >&2
    exit 1
fi
MAX_METRIC_CALLS="${EXPECTED_MAX_METRIC_CALLS}"

case "${BUDGET_PROFILE}:${CONDITION}" in
    standard:vanilla|standard:react_v2|standard:react_v2_random|standard:action) ;;
    expanded:vanilla|expanded:react_v2) ;;
    *)
        echo "ERROR: the requested condition and budget are not an approved HotPotQA campaign cell" >&2
        exit 1
        ;;
esac

case "${MODEL_PROFILE}" in
    qwen3.8-27b)
        MODEL="Qwen3.8-27B"
        SOLVER_MODEL_PATH="${MODEL_STORAGE}/${MODEL}"
        SOLVER_SERVED_NAME="Qwen/Qwen3.8-27B"
        SOLVER_MODEL="hosted_vllm/Qwen/Qwen3.8-27B"
        HOTPOTQA_MODEL_REVISION="${QWEN_REVISION}"
        SOLVER_API_BASE=""
        MODEL_SNAPSHOT_PROFILE="qwen3.8-27b"
        SERVING_ENGINE="vllm"
        SERVING_LOCK="examples/hotpotqa/serving/requirements-x86_64-linux-py312.txt"
        ;;
    deepseek-v4.1-flash)
        MODEL="DeepSeek-V4.1-Flash"
        SOLVER_MODEL_PATH="${MODEL_STORAGE}/${MODEL}"
        SOLVER_SERVED_NAME="deepseek-ai/DeepSeek-V4.1-Flash"
        SOLVER_MODEL="hosted_vllm/deepseek-ai/DeepSeek-V4.1-Flash"
        HOTPOTQA_MODEL_REVISION="${DEEPSEEK_REVISION}"
        SOLVER_API_BASE=""
        MODEL_SNAPSHOT_PROFILE="deepseek-v4.1-flash"
        SERVING_ENGINE="vllm"
        SERVING_LOCK="examples/hotpotqa/serving/requirements-deepseek-v4.1-flash-x86_64-linux-py312.txt"
        ;;
    *)
        echo "ERROR: MODEL_PROFILE must be qwen3.8-27b or deepseek-v4.1-flash" >&2
        exit 1
        ;;
esac
REFLECTION_MODEL="${SOLVER_MODEL}"
REFLECTION_API_BASE="${SOLVER_API_BASE}"

# --- Keep all large I/O on scratch ----------------------------------------
WIKI17_DIR="${WIKI17_DIR:-${SCRATCH_BASE}/.cache/gepa/wiki17}"
export XDG_CACHE_HOME="${SCRATCH_BASE}/.cache"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"
export VLLM_CACHE_ROOT="${SCRATCH_BASE}/.cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_BASE}/.cache/torchinductor"
export TRITON_CACHE_DIR="${SCRATCH_BASE}/.cache/triton"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export JAX_PLATFORMS=cpu
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_USE_FLASHINFER_SAMPLER=0
export FLASHINFER_WORKSPACE_BASE="${SCRATCH_BASE}"

LOG_DIR="${SCRATCH_BASE}/logs/hotpotqa/${HOTPOTQA_CAMPAIGN_ID}/${HOTPOTQA_SOURCE_COMMIT}"
mkdir -p "${XDG_CACHE_HOME}" "${HF_HOME}" "${VLLM_CACHE_ROOT}" \
    "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${LOG_DIR}"

RUN_LOCK_DIR="${SCRATCH_BASE}/.cache/gepa/hotpotqa-runs/${HOTPOTQA_CAMPAIGN_ID}"
mkdir -p "${RUN_LOCK_DIR}"
RUN_LOCK_PATH="${RUN_LOCK_DIR}/${MODEL_PROFILE}-${BUDGET_PROFILE}-${CONDITION}.lock"
exec {RUN_LOCK_FD}>"${RUN_LOCK_PATH}"
if ! flock -n "${RUN_LOCK_FD}"; then
    echo "ERROR: another HotPotQA job is already writing ${MODEL_PROFILE}/${BUDGET_PROFILE}/${CONDITION}" >&2
    exit 1
fi
echo "==> acquired exclusive run lock for: ${CONDITION}"

# --- Verify immutable data and model artifacts ----------------------------
PY="${GEPA_VENV_DIR}/bin/python"
[[ -x "${PY}" ]] || { echo "ERROR: missing ${PY} -- run scripts/della/build_env.sh" >&2; exit 1; }
if ! "${PY}" -m examples.common.wiki17_bm25 verify --deep --root "${WIKI17_DIR}" >/dev/null; then
    echo "ERROR: Wiki-2017 is not prepared at ${WIKI17_DIR}; run scripts/della/build_env.sh first" >&2
    exit 1
fi
WIKI17_INTEGRITY_SHA256="$(sha256sum "${WIKI17_DIR}/integrity.json" | cut -d' ' -f1)"
HOTPOTQA_VERIFIED_WIKI17_INTEGRITY_SHA256="${WIKI17_INTEGRITY_SHA256}"
export HOTPOTQA_VERIFIED_WIKI17_INTEGRITY_SHA256

if [[ ! -d "${SOLVER_MODEL_PATH}" ]]; then
    echo "ERROR: missing ${SOLVER_MODEL_PATH}; run scripts/della/build_env.sh" >&2
    exit 1
fi
exec {MODEL_LOCK_FD}<"${SOLVER_MODEL_PATH}"
if ! flock -s -n "${MODEL_LOCK_FD}"; then
    echo "ERROR: ${MODEL} checkpoint is being rebuilt; submit again after it finishes" >&2
    exit 1
fi
if ! "${PY}" -m examples.common.model_snapshot verify \
    --model-profile "${MODEL_SNAPSHOT_PROFILE}" --root "${SOLVER_MODEL_PATH}" >/dev/null; then
    echo "ERROR: ${MODEL} does not match pinned revision ${HOTPOTQA_MODEL_REVISION}; run scripts/della/build_env.sh" >&2
    exit 1
fi
HOTPOTQA_MODEL_INTEGRITY_SHA256="$(sha256sum "${SOLVER_MODEL_PATH}/.gepa-model-integrity.json" | cut -d' ' -f1)"

GEN_PORT="$("${PY}" - "${GEN_PORT}" <<'PY'
import socket
import sys

requested_port = int(sys.argv[1]) if sys.argv[1] else 0
if not 0 <= requested_port <= 65535:
    raise SystemExit("GEN_PORT must be between 0 and 65535")
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
    listener.bind(("127.0.0.1", requested_port))
    print(listener.getsockname()[1])
PY
)"
SOLVER_API_BASE="http://127.0.0.1:${GEN_PORT}/v1"
REFLECTION_API_BASE="${SOLVER_API_BASE}"
GEN_LOG="${LOG_DIR}/gen-${SLURM_JOB_ID:-local}.log"

HOTPOTQA_VLLM_VERSION=""
HOTPOTQA_SERVING_LOCK_SHA256=""
HOTPOTQA_CUDA_MODULE=""
HOTPOTQA_VLLM_BATCH_INVARIANT=""
HOTPOTQA_VLLM_SINGLE_SEQUENCE_REPLICAS=""

# --- Frozen vLLM serving environment (one per profile) ---------------------
VLLM_PY="${SERVING_VENV_DIR}/bin/python"
VLLM_BIN="${SERVING_VENV_DIR}/bin/vllm"
if [[ -z "${SERVING_VENV_DIR}" || ! -x "${VLLM_PY}" || ! -x "${VLLM_BIN}" ]]; then
    echo "ERROR: missing frozen vLLM serving environment; run scripts/della/build_env.sh" >&2
    exit 1
fi
# Determinism contract for both profiles: no vLLM batch-invariant mode; instead every
# replica decodes exactly one sequence at a time, so no request's forward pass depends
# on which other requests happen to be in flight.
unset VLLM_BATCH_INVARIANT
HOTPOTQA_VLLM_BATCH_INVARIANT="false"
HOTPOTQA_VLLM_SINGLE_SEQUENCE_REPLICAS="true"
if [[ "${VLLM_MAX_NUM_SEQS}" != "1" ]]; then
    echo "ERROR: scientific runs require one sequence per replica (VLLM_MAX_NUM_SEQS=1)" >&2
    exit 1
fi
case "${MODEL_PROFILE}" in
    qwen3.8-27b)
        # Dense 27B: eight independent TP1 replicas, one API server each.
        if [[ "${VLLM_TENSOR_PARALLEL_SIZE}" != "1" \
            || "${VLLM_DATA_PARALLEL_SIZE}" != "8" \
            || "${VLLM_API_SERVER_COUNT}" != "8" ]]; then
            echo "ERROR: Qwen requires TP1/DP8, eight API servers, and one sequence per replica" >&2
            exit 1
        fi
        export HOTPOTQA_WEIGHT_DTYPE="bfloat16"
        export HOTPOTQA_KV_CACHE_DTYPE="auto"
        VLLM_PROFILE_ARGS=(
            --reasoning-parser qwen3
            --tool-call-parser qwen3_coder
            --tensor-parallel-size 1
            --data-parallel-size 8
            --api-server-count 8
            --kv-cache-dtype auto
        )
        HOTPOTQA_SERVE_ARGUMENTS="tp=1;dp=8;api_servers=8;gpu_memory_utilization=${GEN_GMU};max_model_len=${GEN_MAX_LEN};rope_scaling=none;max_num_seqs=1;max_num_batched_tokens=${VLLM_MAX_NUM_BATCHED_TOKENS};dtype=bfloat16;kv_cache_dtype=auto;prefix_caching=false;language_model_only=true;reasoning_parser=qwen3;auto_tool_choice=true;tool_parser=qwen3_coder;seed=0;batch_invariant=false;single_sequence_replicas=true"
        SERVE_TOPOLOGY="TP1/DP8"
        ;;
    deepseek-v4.1-flash)
        # DeepSeek-V4.1-Flash: a MoE (384 routed experts, 6 active per token) whose
        # checkpoint ships FP8 block-quantized dense weights (32x32 blocks, ue8m0 scales),
        # FP4 experts, and FP8 Engram lookup tables. One TP8 replica across the node with
        # expert parallelism (EP8), a single API server, no speculative decoding (neither
        # the MTP layers nor the DSpark draft module is loaded), and no data-parallel
        # attention. Weights stay in the checkpoint's quantized formats; the dtype flag
        # sets the activation dtype. Prompts use the checkpoint's native encoding
        # (tokenizer mode deepseek_v41; the repository ships no Jinja template). The KV
        # cache is fp8_ds_mla, the layout its sparse-MLA path uses; the KV block size is
        # left to vLLM, whose SM90 sparse-MLA kernel needs 64-token blocks. On the H200
        # (SM90) the FP4 experts run on vLLM's Marlin MoE backend.
        if [[ "${VLLM_TENSOR_PARALLEL_SIZE}" != "8" \
            || "${VLLM_DATA_PARALLEL_SIZE}" != "1" \
            || "${VLLM_API_SERVER_COUNT}" != "1" ]]; then
            echo "ERROR: DeepSeek requires one TP8/EP8 replica with one API server on the eight-H200 node" >&2
            exit 1
        fi
        export HOTPOTQA_WEIGHT_DTYPE="fp8"
        export HOTPOTQA_KV_CACHE_DTYPE="fp8"
        VLLM_PROFILE_ARGS=(
            --tokenizer-mode deepseek_v41
            --reasoning-parser deepseek_v41
            --tool-call-parser deepseek_v41
            --tensor-parallel-size 8
            --enable-expert-parallel
            --data-parallel-size 1
            --api-server-count 1
            --kv-cache-dtype fp8
        )
        # The prebuilt flashinfer kernel wheels are in the serving lock, so a missing
        # kernel must fail instead of reaching for the network from the offline node.
        # Loading 510 GB of weights and warming DeepGEMM can outlast vLLM's default
        # engine-ready timeout.
        export FLASHINFER_NO_DOWNLOAD=1
        export VLLM_ENGINE_READY_TIMEOUT_S=3600
        HOTPOTQA_SERVE_ARGUMENTS="tp=8;ep=8;dp=1;api_servers=1;dp_attention=false;speculative_decoding=false;gpu_memory_utilization=${GEN_GMU};max_model_len=${GEN_MAX_LEN};rope_scaling=none;max_num_seqs=1;max_num_batched_tokens=${VLLM_MAX_NUM_BATCHED_TOKENS};dtype=bfloat16;weight_quant=fp8;expert_dtype=fp4;kv_cache_dtype=fp8;block_size=auto;prefix_caching=false;language_model_only=true;tokenizer_mode=deepseek_v41;reasoning_parser=deepseek_v41;auto_tool_choice=true;tool_parser=deepseek_v41;seed=0;batch_invariant=false;single_sequence_replicas=true"
        SERVE_TOPOLOGY="TP8/EP8"
        ;;
esac
export PATH="${SERVING_VENV_DIR}/bin:${PATH}"
HOTPOTQA_VLLM_VERSION="$("${VLLM_PY}" -c 'from importlib.metadata import version; print(version("vllm"))')"
HOTPOTQA_TORCH_VERSION="$("${VLLM_PY}" -c 'import torch; print(torch.__version__)')"
HOTPOTQA_CUDA_VERSION="$("${VLLM_PY}" -c 'import torch; print(torch.version.cuda)')"
HOTPOTQA_TRANSFORMERS_VERSION="$("${VLLM_PY}" -c 'from importlib.metadata import version; print(version("transformers"))')"
HOTPOTQA_CUDA_MODULE="cudatoolkit/${HOTPOTQA_CUDA_VERSION}"
source /usr/share/Modules/init/bash
if ! module load "${HOTPOTQA_CUDA_MODULE}" || ! module is-loaded "${HOTPOTQA_CUDA_MODULE}"; then
    echo "ERROR: exact CUDA module ${HOTPOTQA_CUDA_MODULE} is unavailable" >&2
    exit 1
fi
# The module's /usr/local/cuda-* tree is a per-node local install, and at least one
# ailab node lacks the cuBLAS headers that FlashInfer's first-load JIT builds include.
# Put the locked CUDA wheels' headers and libraries first so those builds depend only
# on the hash-locked serving environment, not on what the node happens to have.
SERVING_CUDA_ROOT="$("${VLLM_PY}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/nvidia/cu13"
if [[ -d "${SERVING_CUDA_ROOT}/include" ]]; then
    export CPATH="${SERVING_CUDA_ROOT}/include${CPATH:+:${CPATH}}"
    export LIBRARY_PATH="${SERVING_CUDA_ROOT}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
fi
if [[ ! -f "${SERVING_LOCK}" ]]; then
    echo "ERROR: Slurm job source lacks ${SERVING_LOCK}" >&2
    exit 1
fi
HOTPOTQA_SERVING_LOCK_SHA256="$(sha256sum "${SERVING_LOCK}" | cut -d' ' -f1)"
if [[ ! -f "${SERVING_VENV_DIR}/.gepa-serving-lock.sha256" \
    || "$(tr -d '\n' < "${SERVING_VENV_DIR}/.gepa-serving-lock.sha256")" != "${HOTPOTQA_SERVING_LOCK_SHA256}" ]]; then
    echo "ERROR: serving environment was built from a different lockfile; run scripts/della/build_env.sh" >&2
    exit 1
fi
SERVING_ENV_MANIFEST="${SCRATCH_BASE}/.cache/gepa/serving-environments/${HOTPOTQA_SERVING_LOCK_SHA256}.json"
if ! "${GEPA_UV_BIN}" pip check --python "${VLLM_PY}"; then
    echo "ERROR: serving environment has inconsistent dependencies" >&2
    exit 1
fi
VERIFIED_SERVING_ENV_SHA256="$(
    "${VLLM_PY}" -m examples.common.python_environment verify --path "${SERVING_ENV_MANIFEST}"
)"
if [[ ! "${HOTPOTQA_SERVING_ENV_SHA256}" =~ ^[0-9a-f]{64}$ \
    || "${VERIFIED_SERVING_ENV_SHA256}" != "${HOTPOTQA_SERVING_ENV_SHA256}" ]]; then
    echo "ERROR: serving environment differs from the submitted frozen environment" >&2
    exit 1
fi
# Fail closed if the frozen vLLM does not register this checkpoint's architecture
# (for the DeepSeek arm, DeepseekV41ForCausalLM), so model support is checked against
# the actual build rather than a version number. The DeepSeek arm's vLLM is a main-branch
# wheel whose version string (0.1.1.dev5+ge77daef89) sorts below every release.
"${VLLM_PY}" - "${SOLVER_MODEL_PATH}" "${HOTPOTQA_VLLM_VERSION}" <<'PY'
import json
import sys
from pathlib import Path

from vllm import ModelRegistry

config = json.loads((Path(sys.argv[1]) / "config.json").read_text())
architectures = config.get("architectures") or []
supported = set(ModelRegistry.get_supported_archs())
unsupported = [architecture for architecture in architectures if architecture not in supported]
if not architectures or unsupported:
    raise SystemExit(
        f"vLLM {sys.argv[2]} does not register {unsupported or architectures!r}; "
        "the pinned serving environment cannot serve this checkpoint."
    )
PY
VLLM_HELP="$("${VLLM_BIN}" serve --help=all 2>&1)"
for required_option in \
    --api-server-count \
    --data-parallel-size \
    --enable-auto-tool-choice \
    --enable-expert-parallel \
    --kv-cache-dtype \
    --language-model-only \
    --no-enable-prefix-caching \
    --reasoning-parser \
    --tokenizer-mode \
    --tool-call-parser
do
    if [[ "${VLLM_HELP}" != *"${required_option}"* ]]; then
        echo "ERROR: ${VLLM_BIN} does not support ${required_option}" >&2
        exit 1
    fi
done
echo "==> serving ${MODEL} with vLLM ${SERVE_TOPOLOGY} on :${GEN_PORT}; log: ${GEN_LOG}"
env \
    -u OMP_NUM_THREADS \
    -u MKL_NUM_THREADS \
    -u OPENBLAS_NUM_THREADS \
    -u NUMEXPR_NUM_THREADS \
    -u TOKENIZERS_PARALLELISM \
    "${VLLM_BIN}" serve "${SOLVER_MODEL_PATH}" \
    --served-model-name "${SOLVER_SERVED_NAME}" \
    --enable-auto-tool-choice \
    --host 127.0.0.1 \
    --port "${GEN_PORT}" \
    --gpu-memory-utilization "${GEN_GMU}" \
    --max-model-len "${GEN_MAX_LEN}" \
    --max-num-seqs 1 \
    --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}" \
    --dtype bfloat16 \
    --seed 0 \
    --no-enable-prefix-caching \
    --language-model-only \
    "${VLLM_PROFILE_ARGS[@]}" \
    > "${GEN_LOG}" 2>&1 &
GEN_PID=$!

HOTPOTQA_GPU_RUNTIME="$("${VLLM_PY}" - <<'PY'
import json
import subprocess

import torch

device_count = torch.cuda.device_count()
if device_count != 8:
    raise SystemExit(f"Scientific runs require exactly eight visible H200 GPUs; found {device_count}.")
names = []
capabilities = []
for device_index in range(device_count):
    name = torch.cuda.get_device_name(device_index).strip()
    capability = ".".join(str(part) for part in torch.cuda.get_device_capability(device_index))
    if "H200" not in name.upper() or capability != "9.0":
        raise SystemExit(
            f"CUDA device {device_index} must be an H200 with compute capability 9.0; "
            f"found {name!r} with {capability}."
        )
    names.append(name)
    capabilities.append(capability)
driver_lines = subprocess.run(
    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
    check=True,
    capture_output=True,
    text=True,
).stdout.splitlines()
driver_versions = sorted({line.strip() for line in driver_lines if line.strip()})
if len(driver_versions) != 1:
    raise SystemExit(f"Expected one NVIDIA driver version; found {driver_versions!r}.")
print(
    json.dumps(
        {
            "compute_capabilities": capabilities,
            "count": device_count,
            "driver_version": driver_versions[0],
            "names": names,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
)
PY
)"

cleanup() {
    echo "==> tearing down local vLLM endpoint"
    kill "${GEN_PID}" 2>/dev/null || true
    wait "${GEN_PID}" 2>/dev/null || true
}
trap cleanup EXIT

generator_reports_expected_model() {
    curl -sf "http://127.0.0.1:${GEN_PORT}/v1/models" \
        | "${PY}" -c 'import json, sys; payload = json.load(sys.stdin); ids = [item.get("id") for item in payload.get("data", [])]; raise SystemExit(0 if ids == [sys.argv[1]] else 1)' "${SOLVER_SERVED_NAME}" \
            2>/dev/null
}

deadline=$((SECONDS + HEALTH_TIMEOUT))
while true; do
    if ! kill -0 "${GEN_PID}" 2>/dev/null; then
        echo "ERROR: vLLM exited before becoming healthy; see ${GEN_LOG}" >&2
        exit 1
    fi
    if generator_reports_expected_model && kill -0 "${GEN_PID}" 2>/dev/null; then
        break
    fi
    if (( SECONDS >= deadline )); then
        echo "ERROR: vLLM did not report ${SOLVER_SERVED_NAME} within ${HEALTH_TIMEOUT}s; see ${GEN_LOG}" >&2
        exit 1
    fi
    sleep 5
done
echo "==> vLLM endpoint ready on :${GEN_PORT}"

# --- Freeze the complete local scientific runtime ------------------------
export OPENAI_API_KEY="EMPTY"
export PYTHONUNBUFFERED=1
export HOTPOTQA_MODEL_REVISION
export HOTPOTQA_MODEL_INTEGRITY_SHA256
export HOTPOTQA_SERVING_ENGINE="${SERVING_ENGINE}"
export HOTPOTQA_SERVE_ARGUMENTS
export HOTPOTQA_VLLM_VERSION
export HOTPOTQA_TORCH_VERSION
export HOTPOTQA_CUDA_VERSION
export HOTPOTQA_CUDA_MODULE
export HOTPOTQA_TRANSFORMERS_VERSION
export HOTPOTQA_SERVING_LOCK_SHA256
export HOTPOTQA_SERVING_ENV_SHA256
export HOTPOTQA_GPU_RUNTIME
export HOTPOTQA_SOURCE_COMMIT
export HOTPOTQA_SOURCE_MANIFEST_SHA256
export HOTPOTQA_PYTHON_VERSION
export HOTPOTQA_UV_VERSION
export HOTPOTQA_UV_SHA256
export HOTPOTQA_ENV_SPEC_SHA256
export HOTPOTQA_GEPA_ENV_SHA256
export HOTPOTQA_VLLM_BATCH_INVARIANT
export HOTPOTQA_VLLM_SINGLE_SEQUENCE_REPLICAS
HOTPOTQA_LITELLM_VERSION="$("${PY}" -c 'from importlib.metadata import version; print(version("litellm"))')"
export HOTPOTQA_LITELLM_VERSION
RUNTIME_IDENTITY="python=${HOTPOTQA_PYTHON_VERSION};uv=${HOTPOTQA_UV_VERSION};uv_sha=${HOTPOTQA_UV_SHA256};gepa_env=${HOTPOTQA_GEPA_ENV_SHA256};model_manifest=${HOTPOTQA_MODEL_INTEGRITY_SHA256};engine=${HOTPOTQA_SERVING_ENGINE};vllm=${HOTPOTQA_VLLM_VERSION};torch=${HOTPOTQA_TORCH_VERSION};cuda=${HOTPOTQA_CUDA_VERSION};cuda_module=${HOTPOTQA_CUDA_MODULE};transformers=${HOTPOTQA_TRANSFORMERS_VERSION};serving_lock=${HOTPOTQA_SERVING_LOCK_SHA256};serving_env=${HOTPOTQA_SERVING_ENV_SHA256};gpu=${HOTPOTQA_GPU_RUNTIME};${HOTPOTQA_SERVE_ARGUMENTS}"
CAMPAIGN_QUALITY_IDENTITY="source=${HOTPOTQA_SOURCE_COMMIT};source_manifest=${HOTPOTQA_SOURCE_MANIFEST_SHA256};python=${HOTPOTQA_PYTHON_VERSION};uv=${HOTPOTQA_UV_VERSION};uv_sha=${HOTPOTQA_UV_SHA256};env_spec=${HOTPOTQA_ENV_SPEC_SHA256};gepa_env=${HOTPOTQA_GEPA_ENV_SHA256};model=${HOTPOTQA_MODEL_REVISION};model_manifest=${HOTPOTQA_MODEL_INTEGRITY_SHA256};engine=${HOTPOTQA_SERVING_ENGINE};vllm=${HOTPOTQA_VLLM_VERSION};torch=${HOTPOTQA_TORCH_VERSION};cuda=${HOTPOTQA_CUDA_VERSION};cuda_module=${HOTPOTQA_CUDA_MODULE};transformers=${HOTPOTQA_TRANSFORMERS_VERSION};litellm=${HOTPOTQA_LITELLM_VERSION};serving_lock=${HOTPOTQA_SERVING_LOCK_SHA256};serving_env=${HOTPOTQA_SERVING_ENV_SHA256};gpu=${HOTPOTQA_GPU_RUNTIME};serve=${HOTPOTQA_SERVE_ARGUMENTS};request_seed=0;dspy=${HOTPOTQA_DSPY_COMMIT};hotpot=${HOTPOTQA_DATA_REVISION};wiki17=${WIKI17_REVISION};index=${WIKI17_INTEGRITY_SHA256}"
CAMPAIGN_IDENTITY_SHA256="$(printf '%s' "${CAMPAIGN_QUALITY_IDENTITY}" | sha256sum | cut -d' ' -f1)"
CAMPAIGN_LOCK_DIR="${SCRATCH_BASE}/.cache/gepa/hotpotqa-campaign/${HOTPOTQA_CAMPAIGN_ID}"
CAMPAIGN_LOCK_PATH="${CAMPAIGN_LOCK_DIR}/${MODEL_PROFILE}.sha256"
mkdir -p "${CAMPAIGN_LOCK_DIR}"
DATA_CAMPAIGN_IDENTITY="source=${HOTPOTQA_SOURCE_COMMIT};source_manifest=${HOTPOTQA_SOURCE_MANIFEST_SHA256};python=${HOTPOTQA_PYTHON_VERSION};uv=${HOTPOTQA_UV_VERSION};uv_sha=${HOTPOTQA_UV_SHA256};env_spec=${HOTPOTQA_ENV_SPEC_SHA256};gepa_env=${HOTPOTQA_GEPA_ENV_SHA256};dspy=${HOTPOTQA_DSPY_COMMIT};hotpot=${HOTPOTQA_DATA_REVISION};wiki17=${WIKI17_REVISION};index=${WIKI17_INTEGRITY_SHA256}"
DATA_CAMPAIGN_SHA256="$(printf '%s' "${DATA_CAMPAIGN_IDENTITY}" | sha256sum | cut -d' ' -f1)"
DATA_CAMPAIGN_LOCK_PATH="${CAMPAIGN_LOCK_DIR}/data-and-source.sha256"
DATA_CAMPAIGN_LOCK_TEMP="$(mktemp "${CAMPAIGN_LOCK_DIR}/.data-and-source.XXXXXX")"
printf '%s\n' "${DATA_CAMPAIGN_SHA256}" > "${DATA_CAMPAIGN_LOCK_TEMP}"
if ln "${DATA_CAMPAIGN_LOCK_TEMP}" "${DATA_CAMPAIGN_LOCK_PATH}" 2>/dev/null; then
    echo "==> froze shared HotPotQA data and source: ${DATA_CAMPAIGN_SHA256}"
elif [[ "$(tr -d '\n' < "${DATA_CAMPAIGN_LOCK_PATH}")" != "${DATA_CAMPAIGN_SHA256}" ]]; then
    echo "ERROR: source, dependency lock, HotPotQA data, DSPy, or Wiki-2017 index differs from the existing campaign lock" >&2
    rm -f -- "${DATA_CAMPAIGN_LOCK_TEMP}"
    exit 1
fi
rm -f -- "${DATA_CAMPAIGN_LOCK_TEMP}"
CAMPAIGN_LOCK_TEMP="$(mktemp "${CAMPAIGN_LOCK_DIR}/.${MODEL_PROFILE}.XXXXXX")"
printf '%s\n' "${CAMPAIGN_IDENTITY_SHA256}" > "${CAMPAIGN_LOCK_TEMP}"
if ln "${CAMPAIGN_LOCK_TEMP}" "${CAMPAIGN_LOCK_PATH}" 2>/dev/null; then
    echo "==> froze ${MODEL_PROFILE} campaign runtime: ${CAMPAIGN_IDENTITY_SHA256}"
elif [[ "$(tr -d '\n' < "${CAMPAIGN_LOCK_PATH}")" != "${CAMPAIGN_IDENTITY_SHA256}" ]]; then
    echo "ERROR: ${MODEL_PROFILE} source or quality-relevant runtime differs from the existing campaign lock" >&2
    rm -f -- "${CAMPAIGN_LOCK_TEMP}"
    exit 1
fi
rm -f -- "${CAMPAIGN_LOCK_TEMP}"
RUNTIME_IDENTITY_SHA256="$(printf '%s' "${RUNTIME_IDENTITY}" | sha256sum | cut -d' ' -f1)"
CACHE_IDENTITY="source=${HOTPOTQA_SOURCE_COMMIT};source_manifest=${HOTPOTQA_SOURCE_MANIFEST_SHA256};python=${HOTPOTQA_PYTHON_VERSION};uv=${HOTPOTQA_UV_VERSION};uv_sha=${HOTPOTQA_UV_SHA256};env_spec=${HOTPOTQA_ENV_SPEC_SHA256};gepa_env=${HOTPOTQA_GEPA_ENV_SHA256};model=${HOTPOTQA_MODEL_REVISION};model_manifest=${HOTPOTQA_MODEL_INTEGRITY_SHA256};budget_profile=${BUDGET_PROFILE};runtime=${RUNTIME_IDENTITY_SHA256};litellm=${HOTPOTQA_LITELLM_VERSION};dspy=${HOTPOTQA_DSPY_COMMIT};hotpot=${HOTPOTQA_DATA_REVISION};wiki17=${WIKI17_REVISION};index=${WIKI17_INTEGRITY_SHA256}"
CACHE_IDENTITY_SHA256="$(printf '%s' "${CACHE_IDENTITY}" | sha256sum | cut -d' ' -f1)"
RUNTIME_CACHE_KEY="${MODEL_PROFILE}-${CACHE_IDENTITY_SHA256}"
export DSPY_CACHEDIR="${SCRATCH_BASE}/.cache/dspy/hotpotqa/${RUNTIME_CACHE_KEY}"
mkdir -p "${DSPY_CACHEDIR}"
echo "==> DSPy cache identity: ${CACHE_IDENTITY_SHA256}"
# The serving process is already running, so these limits apply only to the
# evaluator and prevent its per-example threads from spawning nested CPU pools.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
SOLVER_API_ARG=()
[[ -n "${SOLVER_API_BASE}" ]] && SOLVER_API_ARG=(--solver-api-base "${SOLVER_API_BASE}")
REFLECTION_API_ARG=()
[[ -n "${REFLECTION_API_BASE}" ]] && REFLECTION_API_ARG=(--reflection-api-base "${REFLECTION_API_BASE}")

echo "==> checking native tool-call compatibility"
"${PY}" - "${SOLVER_MODEL}" "${SOLVER_API_BASE}" <<'PY'
import json
import sys

from examples.hotpotqa.utils import resolve_hotpotqa_lm_kwargs
from gepa.lm import LM

model = sys.argv[1]
api_base = sys.argv[2] or None
kwargs = resolve_hotpotqa_lm_kwargs(model, api_base)
completion = LM(model, **kwargs).complete_with_tools(
    [{"role": "user", "content": "Call the echo function exactly once with value ready."}],
    [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Return a supplied readiness value.",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            },
        }
    ],
    tool_choice="auto",
)
if len(completion.tool_calls) != 1 or completion.tool_calls[0].name != "echo":
    raise SystemExit("Native tool-call preflight did not return the required echo call.")
arguments = json.loads(completion.tool_calls[0].arguments)
if arguments.get("value") != "ready":
    raise SystemExit(f"Native tool-call preflight returned unexpected arguments: {arguments!r}")
print("Native tool-call preflight passed.")
PY

echo "==> running GEPA experiment: condition=${CONDITION} budget_profile=${BUDGET_PROFILE} max_metric_calls=${MAX_METRIC_CALLS}"
echo "==> solver=${SOLVER_MODEL} local_api_base=${SOLVER_API_BASE}"
echo "==> reflection=${REFLECTION_MODEL} local_api_base=${REFLECTION_API_BASE}"
echo "==> retrieval=Wiki-2017/BM25 k=7 concurrent_examples=${MAX_WORKERS} root=${WIKI17_DIR}"

"${PY}" -m examples.hotpotqa.main \
    --solver-model "${SOLVER_MODEL}" \
    --reflection-model "${REFLECTION_MODEL}" \
    "${SOLVER_API_ARG[@]}" \
    "${REFLECTION_API_ARG[@]}" \
    --max-metric-calls "${MAX_METRIC_CALLS}" \
    --condition "${CONDITION}" \
    --program 2stage \
    --seed-style structured \
    --seed 0 \
    --max-workers "${MAX_WORKERS}" \
    --wiki17-dir "${WIKI17_DIR}" \
    --retrieval-k 7 \
    --reflection-level 2 \
    --edit-tool-set broad \
    --template-family auto \
    --enforce-scientific-contract \
    --tag "${BUDGET_PROFILE}"
