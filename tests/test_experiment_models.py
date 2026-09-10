"""Tests for shared benchmark model identities and request settings."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from examples.common.experiment_models import (
    DEEPSEEK_V4_FLASH_MODEL,
    DEEPSEEK_V4_FLASH_MODEL_INFO,
    DEEPSEEK_V4_FLASH_OPENROUTER_MODEL,
    DEEPSEEK_V4_FLASH_REVISION,
    EXPERIMENT_MODELS,
    QWEN3_8_27B_MODEL,
    experiment_decoding,
    experiment_model_version,
    experiment_request_overrides,
    resolve_experiment_model,
    validate_experiment_model_pair,
    validate_experiment_vllm_version,
)


def test_deepseek_profile_uses_the_pinned_local_and_openrouter_identities() -> None:
    """Map the DeepSeek arm to its exact checkpoint and smoke-test route."""
    assert EXPERIMENT_MODELS == (QWEN3_8_27B_MODEL, DEEPSEEK_V4_FLASH_MODEL)
    assert DEEPSEEK_V4_FLASH_MODEL == "hosted_vllm/deepseek-ai/DeepSeek-V4-Flash-0731"
    assert DEEPSEEK_V4_FLASH_OPENROUTER_MODEL == "openrouter/deepseek/deepseek-v4-flash-0731"
    assert DEEPSEEK_V4_FLASH_MODEL_INFO == {
        "max_input_tokens": 393_216,
        "max_output_tokens": 16_384,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
    }
    assert resolve_experiment_model(DEEPSEEK_V4_FLASH_MODEL, "direct") == DEEPSEEK_V4_FLASH_MODEL
    assert resolve_experiment_model(DEEPSEEK_V4_FLASH_MODEL, "openrouter") == DEEPSEEK_V4_FLASH_OPENROUTER_MODEL
    assert experiment_model_version(DEEPSEEK_V4_FLASH_MODEL) == DEEPSEEK_V4_FLASH_REVISION
    assert experiment_model_version(DEEPSEEK_V4_FLASH_OPENROUTER_MODEL) == "DeepSeek-V4-Flash-0731"


def test_deepseek_profile_uses_fixed_sampling_and_maximum_reasoning() -> None:
    """Keep local and smoke runs aligned on decoding and reasoning effort."""
    expected_decoding = {
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 16_384,
    }
    assert experiment_decoding(DEEPSEEK_V4_FLASH_MODEL) == expected_decoding
    assert experiment_decoding(DEEPSEEK_V4_FLASH_OPENROUTER_MODEL) == expected_decoding
    assert experiment_request_overrides(DEEPSEEK_V4_FLASH_MODEL) == {
        "extra_body": {
            "chat_template_kwargs": {
                "reasoning_effort": "max",
                "thinking": True,
            },
        }
    }
    assert experiment_request_overrides(DEEPSEEK_V4_FLASH_OPENROUTER_MODEL) == {
        "extra_body": {
            "provider": {
                "only": ["deepinfra"],
                "allow_fallbacks": False,
                "require_parameters": True,
                "quantizations": ["fp8"],
            },
            "reasoning": {"effort": "max"},
        }
    }


@pytest.mark.parametrize("version", ["0.25.0", "0.28.0", "0.29.1.dev1"])
def test_deepseek_accepts_supported_vllm_versions(version: str) -> None:
    """Allow vLLM releases that support the July 31 checkpoint."""
    validate_experiment_vllm_version(DEEPSEEK_V4_FLASH_MODEL, version)


@pytest.mark.parametrize("version", ["0.17.0", "0.24.0", "", "unknown"])
def test_deepseek_rejects_missing_or_old_vllm_versions(version: str) -> None:
    """Reject old Qwen-only environments before a DeepSeek GPU launch."""
    with pytest.raises(ValueError):
        validate_experiment_vllm_version(DEEPSEEK_V4_FLASH_MODEL, version)


@pytest.mark.parametrize("model", ["hosted_vllm/zai-org/GLM-5.3-Flash", "deepseek/deepseek-v4-flash"])
def test_removed_model_routes_cannot_start_or_resume_a_campaign(model: str) -> None:
    """Reject the removed GLM arm and the unpinned direct-provider alias."""
    with pytest.raises(ValueError, match="Unsupported experiment model"):
        validate_experiment_model_pair(model, model)
