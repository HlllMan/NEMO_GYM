# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Math Answer Scoring Function - Simplified from app.py

原始文件: resources_servers/math_with_judge/app.py (293 lines)
简化后: app_copy.py (134 lines)

主要改动:
【删除】
- 所有 async/await 异步操作
- FastAPI 和 Server 相关代码 (SimpleResourcesServer, setup_webserver)
- Pydantic 数据模型类 (Request/Response classes)
- LLM Judge 相关代码 (judge_model_server, _verify_answer_with_judge, _generate_judge_evaluation)
- Judge 提示词模板 (JUDGE_SYSTEM_MESSAGE, JUDGE_PROMPT_TEMPLATE)
- 配置类 (LibraryJudgeMathResourcesServerConfig)

【保留】
- MathVerifier 类：封装 math-verify 验证逻辑
- _verify_answer_with_library() 方法：完全保留原始实现，使用 math-verify 库
- _mute_output() 方法：静音 math-verify 输出

【新增】
- score_math_answer() 函数：简化的打分接口，可直接用于 RL 框架
- get_verifier() 函数：全局单例，避免重复初始化

用法:
    from app_copy import score_math_answer
    
    reward = score_math_answer(
        expected_answer="42",      # 标准答案，不要包含 \\boxed{}
        generated_answer="\\boxed{42}"  # 模型输出，推荐包含 \\boxed{}
    )
    # 返回: 1.0 (正确) 或 0.0 (错误)
"""

import contextlib
import logging
from io import StringIO
from typing import Optional

from math_verify import grader
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig


class MathVerifier:
    def __init__(self):
        logging.getLogger("math_verify").setLevel(logging.CRITICAL)

        self._library_verifier = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(
                ExprExtractionConfig(),
                LatexExtractionConfig(),
            ),
        )

    @staticmethod
    @contextlib.contextmanager
    def _mute_output():
        devnull_out, devnull_err = StringIO(), StringIO()
        with (
            contextlib.redirect_stdout(devnull_out),
            contextlib.redirect_stderr(devnull_err),
        ):
            yield

    def _verify_answer_with_library(self, expected_answer: str, generated_answer: str) -> tuple[float, Optional[str]]:
        try:
            ground_truth_parsable = "\\boxed{" + expected_answer + "}"
            with self._mute_output():
                ret_score, extracted_answer = self._library_verifier([ground_truth_parsable], [generated_answer])

            reward = float(ret_score)

            if extracted_answer is not None:
                assert len(extracted_answer) == 2
                extracted_gold, extracted_prediction = extracted_answer

                for pred in extracted_prediction:
                    if any(grader.verify(gold, pred) for gold in extracted_gold):
                        extracted_answer = pred
                        break
                else:
                    extracted_answer = extracted_prediction[0]

            return reward, extracted_answer

        except (Exception, TimeoutException):
            return 0.0, None


_global_verifier = None


def get_verifier() -> MathVerifier:
    global _global_verifier
    if _global_verifier is None:
        _global_verifier = MathVerifier()
    return _global_verifier


def score_fn(model_output: str, label: str) -> float:
    """
    Math answer scoring function for RL framework
    
    Args:
        expected_answer: Ground truth answer (e.g. "42", "\\frac{3}{4}")
                        Do NOT include \\boxed{}
        generated_answer: Model output (e.g. "\\boxed{42}")
                         Recommended format: \\boxed{answer}
        
    Returns:
        float: 1.0 (correct) or 0.0 (incorrect)
    """
    verifier = get_verifier()
    reward, _ = verifier._verify_answer_with_library(label, model_output)
    return reward
