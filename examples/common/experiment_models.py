"""Model identities and request settings for the paired benchmark runs."""

from copy import deepcopy

QWEN3_8_27B_REPO = "Qwen/Qwen3.8-27B"
QWEN3_8_27B_MODEL = f"hosted_vllm/{QWEN3_8_27B_REPO}"
QWEN3_8_27B_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
QWEN3_8_27B_MODEL_INFO = {
    "max_input_tokens": 262_144,
    "max_output_tokens": 16_384,
    "input_cost_per_token": 0.0,
    "output_cost_per_token": 0.0,
}
# Locally served DeepSeek-V4-Flash-0731 (the official release with enhanced agentic
# capabilities, not the preview, -Base, or -DSpark variants). This is the HotPotQA
# campaign's second arm and is distinct from DEEPSEEK_V4_FLASH_MODEL below, the
# hosted DeepSeek API model used by the HoVer and Terminal-Bench harnesses.
DEEPSEEK_V4_FLASH_0731_REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
DEEPSEEK_V4_FLASH_0731_MODEL = f"hosted_vllm/{DEEPSEEK_V4_FLASH_0731_REPO}"
DEEPSEEK_V4_FLASH_0731_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
DEEPSEEK_V4_FLASH_0731_MODEL_INFO = {
    "max_input_tokens": 262_144,
    "max_output_tokens": 16_384,
    "input_cost_per_token": 0.0,
    "output_cost_per_token": 0.0,
}
EXPERIMENT_MODELS = (QWEN3_8_27B_MODEL, DEEPSEEK_V4_FLASH_0731_MODEL)
DEEPSEEK_V4_FLASH_MODEL = "deepseek/deepseek-v4-flash"
EXPERIMENT_NUM_RETRIES = 0

_EXPERIMENT_MODEL_VERSIONS = {
    QWEN3_8_27B_MODEL: QWEN3_8_27B_REVISION,
    DEEPSEEK_V4_FLASH_0731_MODEL: DEEPSEEK_V4_FLASH_0731_REVISION,
}

# These settings follow each checkpoint's published generation configuration;
# the lower output limit is the fixed experiment contract for both model arms.
# DeepSeek recommends temperature 1.0 with top_p 0.95 for agentic scenarios.
# Sources: https://huggingface.co/Qwen/Qwen3.8-27B
#          https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731
_EXPERIMENT_DECODING = {
    QWEN3_8_27B_MODEL: {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": 16_384,
    },
    DEEPSEEK_V4_FLASH_0731_MODEL: {
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 16_384,
    },
    DEEPSEEK_V4_FLASH_MODEL: {
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 16_384,
        "reasoning_effort": "max",
    },
}

# vLLM renders DeepSeek V4 prompts with the checkpoint's own encoding rather than a
# Jinja template. Its apply_chat_template reads ``thinking`` (default False, which
# would select the no-reasoning "chat" mode) and ``reasoning_effort`` ("max" or
# "xhigh" select the maximum level; anything else but "none" maps to "high").
_EXPERIMENT_REQUEST_OVERRIDES = {
    DEEPSEEK_V4_FLASH_0731_MODEL: {
        "extra_body": {
            "chat_template_kwargs": {
                "thinking": True,
                "reasoning_effort": "max",
            },
        }
    },
}


def experiment_decoding(model: str) -> dict[str, int | float | str]:
    """Return the fixed decoding settings for one experiment model.

    Qwen3.8-27B and DeepSeek-V4-Flash-0731 use their published thinking-mode
    sampling parameters. DeepSeek's thinking mode and maximum reasoning effort
    are carried separately in its request override so the local serving runtime
    applies them through the checkpoint's prompt encoding.

    Args:
        model: Exact LiteLLM model identifier used by a benchmark run.

    Returns:
        Independent decoding-parameter mapping for the requested model.

    Raises:
        ValueError: The model is not a supported experiment runtime.
    """
    try:
        return dict(_EXPERIMENT_DECODING[model])
    except KeyError as exc:
        supported = ", ".join(_EXPERIMENT_DECODING)
        raise ValueError(f"Unsupported experiment model {model!r}; expected one of: {supported}") from exc


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

    Self-hosted DeepSeek requests enable thinking mode and maximum reasoning
    through vLLM's chat-template arguments. A deep copy keeps one client from
    mutating the policy used by later calls.

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
