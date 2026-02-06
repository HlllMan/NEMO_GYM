# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Standalone scoring function extracted from app.py
# Removes: FastAPI server, async operations
# Keeps: Instruction verification logic using verifiable_instructions library

from typing import List, Literal, Optional

from mjnemogym.log import get_logger

from .verifiable_instructions import instructions_registry

_logger = get_logger("if")


def _ensure_nltk_data():
    """Download required NLTK data if needed."""
    try:
        import nltk
        nltk.download("punkt_tab", quiet=True)
    except ImportError:
        pass
    except Exception as e:
        _logger.warning(f"NLTK setup: {e}")


# Initialize NLTK data on module load
# _ensure_nltk_data()


def verify_instructions(
    model_output: str,
    instruction_id_list: List[str],
    kwargs_list: List[Optional[dict]],
    grading_mode: Literal["binary", "fraction"] = "binary",
) -> tuple[float, bool, List[bool]]:
    """
    Verify instruction following - core logic from app.py InstructionFollowingResourcesServer.verify()

    Args:
        model_output: Model-generated text response
        instruction_id_list: List of instruction IDs to verify
        kwargs_list: List of kwargs dicts for each instruction
        grading_mode: "binary" (all or nothing) or "fraction" (partial credit)

    Returns:
        (reward, follow_all, follow_list):
            - reward: Score based on grading_mode
            - follow_all: True if all instructions followed
            - follow_list: List of booleans for each instruction
    """
    is_following_list = []

    for instruction_id, kwargs in zip(instruction_id_list, kwargs_list):
        try:
            # Create instruction instance
            instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
            instruction = instruction_cls(instruction_id)

            # Handle None kwargs
            if kwargs is None:
                kwargs = {}

            # Filter out None values from kwargs
            filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}

            # Build the instruction description with the provided kwargs
            instruction.build_description(**filtered_kwargs)

            # Check if the response follows the instruction
            if instruction.check_following(model_output):
                is_following_list.append(True)
            else:
                is_following_list.append(False)

        except Exception as e:
            _logger.warning(f"instruction {instruction_id} error: {type(e).__name__}: {e}")
            is_following_list.append(False)

    # Calculate reward based on grading mode
    if grading_mode == "binary":
        reward = float(all(is_following_list))
    elif grading_mode == "fraction":
        reward = float((sum(is_following_list) / len(is_following_list)) if is_following_list else 0.0)
    else:
        raise ValueError(f"Invalid grading mode: {grading_mode}")

    follow_all = all(is_following_list)
    return reward, follow_all, is_following_list


def score_fn(model_output: str, extra_info: dict) -> float:
    """
    Standalone instruction following scoring function for verl reward manager.

    Args:
        model_output: Model-generated text response
        extra_info: Dictionary from parquet containing:
            - instruction_id_list: List of instruction IDs
            - kwargs: List of kwargs dicts for each instruction
            - grading_mode: Optional grading mode (default: "binary")

    Returns:
        float: Score based on grading_mode (1.0/0.0 for binary, fraction for partial)
    """
    idx = extra_info.get("index", "?")
    instruction_id_list = extra_info.get("instruction_id_list", [])
    kwargs_list = extra_info.get("kwargs", [])
    grading_mode = extra_info.get("grading_mode", "binary")

    if not instruction_id_list:
        _logger.debug(f"DONE idx={idx} reward=0.0 reason=no_instructions")
        return 0.0

    _logger.debug(f"START idx={idx} num_instructions={len(instruction_id_list)}")
    try:
        reward, _, _ = verify_instructions(
            model_output=model_output,
            instruction_id_list=instruction_id_list,
            kwargs_list=kwargs_list,
            grading_mode=grading_mode,
        )
        _logger.debug(f"DONE idx={idx} reward={reward}")
        return reward
    except Exception as e:
        _logger.warning(f"EXCEPTION idx={idx}: {type(e).__name__}: {e}")
        return 0.0
