"""Model identities and request settings for the paired benchmark runs."""

from copy import deepcopy

from packaging.version import Version

QWEN3_8_27B_REPO = "Qwen/Qwen3.8-27B"
QWEN3_8_27B_MODEL = f"hosted_vllm/{QWEN3_8_27B_REPO}"
QWEN3_8_27B_OPENROUTER_MODEL = "openrouter/qwen/qwen3.8-27b"
QWEN3_8_27B_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
QWEN3_8_27B_MODEL_INFO = {
    "max_input_tokens": 262_144,
    "max_output_tokens": 16_384,
    "input_cost_per_token": 0.0,
    "output_cost_per_token": 0.0,
}
DEEPSEEK_V4_FLASH_REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
DEEPSEEK_V4_FLASH_MODEL = f"hosted_vllm/{DEEPSEEK_V4_FLASH_REPO}"
DEEPSEEK_V4_FLASH_OPENROUTER_MODEL = "openrouter/deepseek/deepseek-v4-flash-0731"
DEEPSEEK_V4_FLASH_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
DEEPSEEK_V4_FLASH_MODEL_INFO = {
    "max_input_tokens": 393_216,
    "max_output_tokens": 16_384,
    "input_cost_per_token": 0.0,
    "output_cost_per_token": 0.0,
}
EXPERIMENT_MODELS = (QWEN3_8_27B_MODEL, DEEPSEEK_V4_FLASH_MODEL)
EXPERIMENT_API_PROFILES = ("direct", "openrouter")
EXPERIMENT_NUM_RETRIES = 0

_OPENROUTER_RUNTIME_MODELS = {
    QWEN3_8_27B_MODEL: QWEN3_8_27B_OPENROUTER_MODEL,
    DEEPSEEK_V4_FLASH_MODEL: DEEPSEEK_V4_FLASH_OPENROUTER_MODEL,
}

_EXPERIMENT_MODEL_VERSIONS = {
    QWEN3_8_27B_MODEL: QWEN3_8_27B_REVISION,
    QWEN3_8_27B_OPENROUTER_MODEL: QWEN3_8_27B_REVISION,
    DEEPSEEK_V4_FLASH_MODEL: DEEPSEEK_V4_FLASH_REVISION,
    DEEPSEEK_V4_FLASH_OPENROUTER_MODEL: "DeepSeek-V4-Flash-0731",
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
    QWEN3_8_27B_OPENROUTER_MODEL: {
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
    DEEPSEEK_V4_FLASH_OPENROUTER_MODEL: {
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 16_384,
    },
}

# OpenRouter's native request body keeps the provider constraint attached to
# ordinary completions and native tool calls. Fallbacks are disabled so a run
# cannot silently change inference providers after it starts.
_EXPERIMENT_REQUEST_OVERRIDES = {
    DEEPSEEK_V4_FLASH_MODEL: {
        "extra_body": {
            "chat_template_kwargs": {
                "reasoning_effort": "max",
                "thinking": True,
            },
        }
    },
    QWEN3_8_27B_OPENROUTER_MODEL: {
        "extra_body": {
            "provider": {
                "only": ["akashml"],
                "allow_fallbacks": False,
                "require_parameters": True,
                "quantizations": ["bf16"],
                "max_price": {"prompt": 0.40, "completion": 2.55},
            },
            "reasoning": {"effort": "xhigh"},
        }
    },
    DEEPSEEK_V4_FLASH_OPENROUTER_MODEL: {
        "extra_body": {
            "provider": {
                "only": ["deepinfra"],
                "allow_fallbacks": False,
                "require_parameters": True,
                "quantizations": ["fp8"],
            },
            "reasoning": {"effort": "max"},
        }
    },
}


def experiment_decoding(model: str) -> dict[str, int | float | str]:
    """Return the fixed decoding settings for one experiment model.

    Qwen3.8-27B and DeepSeek-V4-Flash-0731 use their published thinking-mode sampling
    parameters. Maximum DeepSeek reasoning is carried separately in its request
    override so the local serving runtime applies it through the checkpoint's
    template.

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
        supported = ", ".join(EXPERIMENT_MODELS)
        raise ValueError(f"Unsupported experiment model {model!r}; expected one of: {supported}") from exc


def experiment_model_version(model: str) -> str:
    """Return the exact checkpoint revision for one model.

    Args:
        model: Canonical or transport-specific experiment model identifier.

    Returns:
        Exact local checkpoint revision or the named hosted release. Hosted
        providers do not attest to the local checkpoint byte digest.

    Raises:
        ValueError: The model is not part of the experiment matrix.
    """
    if model not in _EXPERIMENT_MODEL_VERSIONS:
        supported = ", ".join(EXPERIMENT_MODELS)
        raise ValueError(f"Unsupported experiment model {model!r}; expected one of: {supported}")
    version = _EXPERIMENT_MODEL_VERSIONS[model]
    return version


def experiment_request_overrides(model: str) -> dict[str, object]:
    """Return provider-specific request fields for one runtime model.

    Self-hosted DeepSeek requests set maximum reasoning through the checkpoint's
    chat-template arguments. OpenRouter identifiers receive a native request
    body that fixes the inference provider, disables fallback routing, requires
    every requested parameter, and fixes the reasoning mode. A deep copy keeps
    one client from mutating the policy used by later calls.

    Args:
        model: Exact LiteLLM model identifier used by a benchmark run.

    Returns:
        Independent provider-request mapping, or an empty mapping when the
        selected runtime does not need a transport override.

    Raises:
        ValueError: The model is not a supported experiment runtime.
    """
    if model not in _EXPERIMENT_DECODING:
        supported = ", ".join(EXPERIMENT_MODELS)
        raise ValueError(f"Unsupported experiment model {model!r}; expected one of: {supported}")
    return deepcopy(_EXPERIMENT_REQUEST_OVERRIDES.get(model, {}))


def resolve_experiment_model(model: str, api_profile: str) -> str:
    """Resolve a scientific model identity to its API runtime identifier.

    The direct profile preserves the configured identifier used by Della and
    provider-native runs. The OpenRouter profile maps the same experimental
    arm to an exact OpenRouter model slug; request routing remains separate in
    :func:`experiment_request_overrides`.

    Args:
        model: Canonical student/proposer identity for the experiment arm.
        api_profile: Runtime route, either ``"direct"`` or ``"openrouter"``.

    Returns:
        LiteLLM model identifier used for actual completion requests.

    Raises:
        ValueError: The model or API profile is not supported.
    """
    if model not in EXPERIMENT_MODELS:
        supported = ", ".join(EXPERIMENT_MODELS)
        raise ValueError(f"Unsupported experiment model {model!r}; expected one of: {supported}")
    if api_profile == "direct":
        return model
    if api_profile == "openrouter":
        return _OPENROUTER_RUNTIME_MODELS[model]
    supported_profiles = ", ".join(EXPERIMENT_API_PROFILES)
    raise ValueError(f"Unsupported API profile {api_profile!r}; expected one of: {supported_profiles}")


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
    if student_model not in EXPERIMENT_MODELS:
        supported = ", ".join(EXPERIMENT_MODELS)
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
