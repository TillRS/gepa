"""Model identities and request settings for the paired benchmark runs."""

from copy import deepcopy

from packaging.version import Version

QWEN3_8_27B_REPO = "Qwen/Qwen3.8-27B"
QWEN3_8_27B_MODEL = f"hosted_vllm/{QWEN3_8_27B_REPO}"
QWEN3_8_27B_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
QWEN3_8_27B_MODEL_INFO = {
    "max_input_tokens": 262_144,
    "max_output_tokens": 16_384,
    "input_cost_per_token": 0.0,
    "output_cost_per_token": 0.0,
}
DEEPSEEK_V4_FLASH_REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
DEEPSEEK_V4_FLASH_MODEL = f"hosted_vllm/{DEEPSEEK_V4_FLASH_REPO}"
DEEPSEEK_V4_FLASH_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
DEEPSEEK_V4_FLASH_MODEL_INFO = {
    "max_input_tokens": 393_216,
    "max_output_tokens": 16_384,
    "input_cost_per_token": 0.0,
    "output_cost_per_token": 0.0,
}
EXPERIMENT_MODELS = (QWEN3_8_27B_MODEL, DEEPSEEK_V4_FLASH_MODEL)
EXPERIMENT_NUM_RETRIES = 0

_EXPERIMENT_MODEL_VERSIONS = {
    QWEN3_8_27B_MODEL: QWEN3_8_27B_REVISION,
    DEEPSEEK_V4_FLASH_MODEL: DEEPSEEK_V4_FLASH_REVISION,
}

# These settings follow each checkpoint's published generation configuration;
# the lower output limit is the fixed experiment contract for both model arms.
# Sources: https://huggingface.co/Qwen/Qwen3.8-27B
#          https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731
_EXPERIMENT_DECODING = {
    QWEN3_8_27B_MODEL: {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": 16_384,
    },
    DEEPSEEK_V4_FLASH_MODEL: {
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 16_384,
    },
}

_EXPERIMENT_REQUEST_OVERRIDES: dict[str, dict[str, object]] = {
    DEEPSEEK_V4_FLASH_MODEL: {
        "extra_body": {
            "chat_template_kwargs": {
                "reasoning_effort": "max",
                "thinking": True,
            },
        }
    },
}


def experiment_decoding(model: str, *, agentic: bool = True) -> dict[str, int | float | str]:
    """Return provider decoding settings for the model and kind of work.

    Qwen3.8-27B and DeepSeek-V4-Flash-0731 use their published thinking-mode sampling
    parameters. Maximum DeepSeek reasoning is carried separately in its request
    override so the local serving runtime applies it through the checkpoint's
    template.

    Args:
        model: Exact LiteLLM model identifier used by a benchmark run.
        agentic: Whether the role iteratively uses tools. DeepSeek recommends
            top-p 0.95 for agentic work and 1.0 otherwise. The default preserves
            existing callers; reviewed benchmarks classify each role explicitly.

    Returns:
        Independent decoding-parameter mapping for the requested model.

    Raises:
        ValueError: The model is not a supported experiment runtime.
    """
    try:
        decoding = dict(_EXPERIMENT_DECODING[model])
    except KeyError as exc:
        supported = ", ".join(_EXPERIMENT_DECODING)
        raise ValueError(f"Unsupported experiment model {model!r}; expected one of: {supported}") from exc
    if model == DEEPSEEK_V4_FLASH_MODEL and not agentic:
        decoding["top_p"] = 1.0
    return decoding


def experiment_model_version(model: str) -> str:
    """Return the exact checkpoint revision for one model.

    Args:
        model: Canonical experiment model identifier.

    Returns:
        Exact Hugging Face revision for the local checkpoint.

    Raises:
        ValueError: The model is not part of the experiment matrix.
    """
    if model not in _EXPERIMENT_MODEL_VERSIONS:
        supported = ", ".join(_EXPERIMENT_MODEL_VERSIONS)
        raise ValueError(f"Unsupported experiment model {model!r}; expected one of: {supported}")
    version = _EXPERIMENT_MODEL_VERSIONS[model]
    return version


def experiment_request_overrides(model: str) -> dict[str, object]:
    """Return provider-specific request fields for one runtime model.

    Self-hosted DeepSeek requests set maximum reasoning through the checkpoint's
    chat-template arguments. A deep copy keeps one client from mutating the
    policy used by later calls.

    Args:
        model: Exact LiteLLM model identifier used by a benchmark run.

    Returns:
        Independent provider-request mapping, or an empty mapping when the
        selected runtime does not need a transport override.

    Raises:
        ValueError: The model is not a supported experiment runtime.
    """
    if model not in _EXPERIMENT_DECODING:
        supported = ", ".join(_EXPERIMENT_DECODING)
        raise ValueError(f"Unsupported experiment model {model!r}; expected one of: {supported}")
    return deepcopy(_EXPERIMENT_REQUEST_OVERRIDES.get(model, {}))


def validate_experiment_model_pair(student_model: str, proposer_model: str) -> None:
    """Require a homogeneous student/proposer experiment profile.

    Args:
        student_model: Model that executes the benchmark program.
        proposer_model: Model that reflects on traces and proposes revisions.

    Raises:
        ValueError: The roles use different models or an unrecognized model.
    """
    if student_model != proposer_model:
        raise ValueError(
            "Benchmark runs require the same model for the student and proposer within each arm; "
            f"received student={student_model!r} and proposer={proposer_model!r}."
        )
    if student_model not in _EXPERIMENT_DECODING:
        supported = ", ".join(_EXPERIMENT_DECODING)
        raise ValueError(f"Unsupported experiment model {student_model!r}; expected one of: {supported}")


def validate_experiment_vllm_version(model: str, version: str) -> None:
    """Reject serving versions that predate the selected checkpoint's support.

    Args:
        model: Canonical local experiment model identifier.
        version: Installed vLLM package version, including a possible dev suffix.

    Raises:
        ValueError: The model or installed serving version is unsupported.
    """
    validate_experiment_model_pair(model, model)
    minimum = "0.25.0" if model == DEEPSEEK_V4_FLASH_MODEL else "0.17.0"
    if Version(version) < Version(minimum):
        raise ValueError(f"{model} requires vLLM>={minimum}; found {version}.")


def experiment_model_info(model: str) -> dict[str, int | float] | None:
    """Return explicit context and cost metadata for a local served model.

    Args:
        model: Canonical or hosted experiment model identifier.

    Returns:
        Local server metadata, or None for a hosted route with its own catalog.
    """
    if model == QWEN3_8_27B_MODEL:
        return dict(QWEN3_8_27B_MODEL_INFO)
    if model == DEEPSEEK_V4_FLASH_MODEL:
        return dict(DEEPSEEK_V4_FLASH_MODEL_INFO)
    return None
