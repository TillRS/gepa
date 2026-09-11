"""Tests for shared benchmark model identities and request settings."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from examples.common.experiment_models import (
    DEEPSEEK_V4_1_FLASH_MODEL,
    DEEPSEEK_V4_1_FLASH_MODEL_INFO,
    DEEPSEEK_V4_1_FLASH_REVISION,
    DEEPSEEK_V4_FLASH_MODEL,
    EXPERIMENT_MODELS,
    QWEN3_8_27B_MODEL,
    experiment_decoding,
    experiment_model_version,
    experiment_request_overrides,
)


def test_deepseek_profile_uses_the_pinned_local_identity() -> None:
    """Map the DeepSeek arm to its exact local checkpoint."""
    assert EXPERIMENT_MODELS == (QWEN3_8_27B_MODEL, DEEPSEEK_V4_1_FLASH_MODEL)
    assert DEEPSEEK_V4_1_FLASH_MODEL == "hosted_vllm/deepseek-ai/DeepSeek-V4.1-Flash"
    assert DEEPSEEK_V4_1_FLASH_REVISION == "dba1be0a40aa45a94ad051997016db3960a90277"
    assert DEEPSEEK_V4_1_FLASH_MODEL_INFO == {
        "max_input_tokens": 262_144,
        "max_output_tokens": 16_384,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
    }
    assert experiment_model_version(DEEPSEEK_V4_1_FLASH_MODEL) == DEEPSEEK_V4_1_FLASH_REVISION


def test_local_deepseek_arm_is_distinct_from_the_hosted_deepseek_api_model() -> None:
    """Keep the HoVer/Terminal-Bench API identity separate from the local checkpoint."""
    assert DEEPSEEK_V4_FLASH_MODEL == "deepseek/deepseek-v4-flash"
    assert DEEPSEEK_V4_FLASH_MODEL != DEEPSEEK_V4_1_FLASH_MODEL
    assert DEEPSEEK_V4_FLASH_MODEL not in EXPERIMENT_MODELS
    assert experiment_request_overrides(DEEPSEEK_V4_FLASH_MODEL) == {}
    assert experiment_decoding(DEEPSEEK_V4_FLASH_MODEL)["reasoning_effort"] == "max"


def test_deepseek_profile_uses_fixed_sampling_and_maximum_reasoning() -> None:
    """Keep the local run fixed on decoding, thinking mode, and reasoning effort."""
    expected_decoding = {
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 16_384,
    }
    assert experiment_decoding(DEEPSEEK_V4_1_FLASH_MODEL) == expected_decoding
    assert experiment_request_overrides(DEEPSEEK_V4_1_FLASH_MODEL) == {
        "extra_body": {
            "chat_template_kwargs": {
                "thinking": True,
                "reasoning_effort": 100,
            },
        }
    }
