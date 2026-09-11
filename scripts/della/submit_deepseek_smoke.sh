#!/bin/bash
# Submit the DeepSeek-V4.1-Flash serving smoke test from the laptop, or fetch its results.
#
# Usage:
#   scripts/della/submit_deepseek_smoke.sh submit          # sync the checkout, submit, print the job id
#   scripts/della/submit_deepseek_smoke.sh fetch <job-id>  # copy the run directory and Slurm log here
#
# Requires scripts/della/build_env.sh to have built the DeepSeek serving environment and
# staged the checkpoint. Results land in outputs/deepseek-smoke/<job-id>/: transcript.md
# (request, server-rendered prompt, reasoning, and content), transcript.json,
# verify_report.txt, serving-packages.txt, gpus.txt, vllm.log, and the Slurm log.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
if [[ ! -f "${ENV_FILE}" ]]; then
    echo "ERROR: ${ENV_FILE} not found." >&2
    exit 1
fi
source "${ENV_FILE}"

SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=yes)
VERIFY_LOG_DIR="${SCRATCH_BASE}/logs/hotpotqa/verify"

case "${1:-}" in
    submit)
        echo "==> syncing the checkout to ${REMOTE_DIR}"
        "${SCRIPT_DIR}/sync_to_della.sh"
        JOB_ID="$(
            ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
                "mkdir -p '${VERIFY_LOG_DIR}' && cd '${REMOTE_DIR}' && sbatch --parsable --output='${VERIFY_LOG_DIR}/smoke-%j.out' scripts/della/smoke_deepseek_serving.sbatch"
        )"
        JOB_ID="${JOB_ID%%;*}"
        if [[ ! "${JOB_ID}" =~ ^[0-9]+$ ]]; then
            echo "ERROR: sbatch returned an invalid job id: ${JOB_ID}" >&2
            exit 1
        fi
        echo "==> submitted DeepSeek-V4.1-Flash smoke test: job ${JOB_ID}"
        echo "    fetch when it finishes: scripts/della/submit_deepseek_smoke.sh fetch ${JOB_ID}"
        ;;
    fetch)
        JOB_ID="${2:?usage: $0 fetch <job-id>}"
        DESTINATION="${REPO_ROOT}/outputs/deepseek-smoke/${JOB_ID}"
        mkdir -p "${DESTINATION}"
        rsync -a -e "ssh ${SSH_OPTS[*]}" \
            "${REMOTE_USER}@${REMOTE_HOST}:${VERIFY_LOG_DIR}/${JOB_ID}/" "${DESTINATION}/"
        rsync -a -e "ssh ${SSH_OPTS[*]}" \
            "${REMOTE_USER}@${REMOTE_HOST}:${VERIFY_LOG_DIR}/smoke-${JOB_ID}.out" "${DESTINATION}/"
        echo "==> fetched into ${DESTINATION}"
        ls -la "${DESTINATION}"
        ;;
    *)
        echo "usage: $0 submit | fetch <job-id>" >&2
        exit 2
        ;;
esac
