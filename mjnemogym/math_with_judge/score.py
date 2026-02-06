# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Math scoring with fallback chain:
# 1. QY math parser (fast, regex-based extraction + math_verify)
# 2. DAPO (regex normalization + string comparison)
# 3. MathVerify (full symbolic parsing)

import logging
import time

from mjnemogym.log import get_logger
from mjnemogym.math_with_judge import qy_parser
from mjnemogym.math_with_judge import dapo
from mjnemogym.math_with_judge import math_verify_method

_logger = get_logger("math")

# Suppress noisy math_verify library logs
logging.getLogger("math_verify").setLevel(logging.CRITICAL)


def score_fn(model_output: str, extra_info: dict) -> float:
    """
    Math scoring function with fallback chain.

    Tries methods in order:
    1. QY math parser (fast regex + math_verify)
    2. DAPO (regex normalization + string comparison)
    3. MathVerify (full symbolic parsing)

    Returns on first non-zero result.

    Args:
        model_output: Model-generated answer text (should contain \\boxed{answer})
        extra_info: Dictionary containing expected_answer

    Returns:
        float: 1.0 (correct) or 0.0 (incorrect)
    """
    idx = extra_info.get("index", "?")
    expected_answer = extra_info.get("expected_answer", "")
    if not expected_answer:
        _logger.debug(f"idx={idx} expected_answer is empty, keys={list(extra_info.keys())}")
        return 0.0

    expected_answer = str(expected_answer)
    _logger.debug(f"START idx={idx}")
    t0 = time.time()

    # Method 1: QY math parser
    try:
        reward = qy_parser.score_fn(model_output, expected_answer)
        if reward > 0:
            _logger.debug(f"DONE idx={idx} reward={reward} method=qy elapsed={time.time()-t0:.2f}s")
            return reward
    except Exception as e:
        _logger.warning(f"idx={idx} QY method failed: {type(e).__name__}: {e}")

    # Method 2: DAPO
    try:
        reward = dapo.score_fn(model_output, expected_answer)
        if reward > 0:
            _logger.debug(f"DONE idx={idx} reward={reward} method=dapo elapsed={time.time()-t0:.2f}s")
            return reward
    except Exception as e:
        _logger.warning(f"idx={idx} DAPO method failed: {type(e).__name__}: {e}")

    # Method 3: MathVerify
    try:
        reward = math_verify_method.score_fn(model_output, expected_answer)
        _logger.debug(f"DONE idx={idx} reward={reward} method=math_verify elapsed={time.time()-t0:.2f}s")
        return reward
    except Exception as e:
        _logger.warning(f"idx={idx} MathVerify method failed: {type(e).__name__}: {e}")
        _logger.debug(f"DONE idx={idx} reward=0.0 method=all_failed elapsed={time.time()-t0:.2f}s")
        return 0.0
