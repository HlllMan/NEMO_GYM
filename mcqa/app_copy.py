# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MCQA Answer Scoring Function - Simplified from app.py

原始文件: resources_servers/mcqa/app.py (288 lines)
简化后: app_copy.py

主要改动:
【删除】
- 所有 async/await 异步操作
- FastAPI 和 Server 相关代码 (SimpleResourcesServer, setup_webserver)
- Pydantic 数据模型类 (Request/Response classes)
- 配置类 (MCQAResourcesServerConfig)

【保留】
- 所有答案提取函数（完全保留原始实现）
- 三种评分模式逻辑（strict_single_letter_boxed, lenient_boxed, lenient_answer_colon）
- 自定义正则表达式支持（template_metadata）

【新增】
- score_mcqa_answer() 函数：简化的打分接口，可直接用于 RL 框架

用法:
    from app_copy import score_mcqa_answer
    
    reward = score_mcqa_answer(
        model_output="The answer is \\boxed{C}",
        expected_answer="C",
        options=[{"A": "选项A"}, {"B": "选项B"}, {"C": "选项C"}, {"D": "选项D"}],
        grading_mode="strict_single_letter_boxed"
    )
    # 返回: 1.0 (正确) 或 0.0 (错误)

大模型输出格式要求:
- 模型输出是一个字符串，包含推理过程和最终答案
- 最终答案需要根据 grading_mode 格式要求：
  * strict_single_letter_boxed: \\boxed{C} (单个大写字母)
  * lenient_boxed: \\boxed{C} 或 \\boxed{选项文本}
  * lenient_answer_colon: Answer: C 或 Answer: 选项文本
- 如果提供了 template_metadata.output_regex，使用自定义正则表达式提取
"""

import re
from typing import Literal, Optional


# 正则表达式模式（完全保留原始实现）
CHOICE_LETTER_PATTERN = re.compile(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])")
STRICT_BOXED_PATTERN = re.compile(r"\\boxed\{\s*[^A-Za-z]*([A-Z])[^A-Za-z]*\s*\}")
ANSWER_COLON_PATTERN = re.compile(r"(?i)answer\s*:\s*(.+)")
BOXED_CONTENT_PATTERN = re.compile(r"\\boxed\{\s*(.*?)\s*\}", re.S)
LATEX_TEXT_WRAP_PATTERN = re.compile(r"\\text\{\s*(.*?)\s*\}", re.S)


def _strip_latex_wrappers(s: str) -> str:
    """Remove successive \\text{...} wrappers from a LaTeX string."""
    while True:
        m = LATEX_TEXT_WRAP_PATTERN.fullmatch(s)
        if not m:
            break
        s = m.group(1)
    return s


def _normalize_for_match(s: str) -> str:
    """Lowercase and collapse whitespace for robust substring/equality checks."""
    return " ".join(s.lower().split())


def _get_allowed_letters_from_options(options: Optional[list[dict[str, str]]]) -> set[str]:
    """Collect uppercase option letters from list of single-key dicts."""
    letters: set[str] = set()
    if options:
        for entry in options:
            # 提取所有非 None 值的键（MCQA 数据格式：每个选项字典只有一个非 None 的键）
            for k, v in entry.items():
                if isinstance(k, str) and len(k) == 1 and k.isalpha() and v is not None:
                    letters.add(k.upper())
    return letters


def _parse_answer_letter_strict_boxed(text: str, allowed_letters: set[str]) -> tuple[Optional[str], str, bool]:
    parsed_text = text
    m = STRICT_BOXED_PATTERN.search(text)
    if not m:
        return None, parsed_text, True
    letter = m.group(1).upper()
    if letter not in allowed_letters:
        return None, parsed_text, True
    return letter, parsed_text, False


def _match_option_text(text: str, options: list[dict[str, str]], allowed_letters: set[str]) -> Optional[str]:
    """Match boxed content against option texts and return the option letter."""
    boxed = BOXED_CONTENT_PATTERN.search(text)
    if not boxed:
        return None
    inner = boxed.group(1)
    candidate_texts = [inner, _strip_latex_wrappers(inner)]
    normalized_candidates = [_normalize_for_match(t) for t in candidate_texts]

    normalized_options: list[tuple[str, str]] = []
    for entry in options or []:
        for k, v in entry.items():
            if isinstance(k, str) and len(k) == 1 and k.upper() in allowed_letters:
                normalized_options.append((k.upper(), _normalize_for_match(v)))
                break

    matched_letters: set[str] = set()
    for cand in normalized_candidates:
        for letter, opt_norm in normalized_options:
            if opt_norm and opt_norm in cand:
                matched_letters.add(letter)
    if len(matched_letters) == 1:
        return next(iter(matched_letters))
    return None


def _parse_answer_with_custom_regex(
    text: str, regex_pattern: str, allowed_letters: set[str], options: Optional[list[dict[str, str]]]
) -> Optional[str]:
    """Parse answer using custom regex from template_metadata."""
    try:
        matches = re.findall(regex_pattern, text, re.IGNORECASE)
        if not matches:
            return None

        captured = matches[-1].strip().upper()

        if len(captured) == 1 and captured.isalpha():
            if allowed_letters and captured in allowed_letters:
                return captured
            elif not allowed_letters:
                return captured
            else:
                return captured

        normalized_captured = _normalize_for_match(captured)
        for entry in options or []:
            for k, v in entry.items():
                if k.upper() in allowed_letters and _normalize_for_match(v) == normalized_captured:
                    return k.upper()

        return None
    except re.error:
        return None


def _verify_answer(
    model_output: str,
    expected_answer: str,
    options: Optional[list[dict[str, str]]] = None,
    grading_mode: Literal[
        "strict_single_letter_boxed",
        "lenient_boxed",
        "lenient_answer_colon",
    ] = "strict_single_letter_boxed",
    template_metadata: Optional[dict] = None,
) -> tuple[float, Optional[str]]:
    """
    验证多选题答案 - 完全保留原始实现逻辑
    
    Args:
        model_output: 模型输出的文本
        expected_answer: 标准答案（单个字母，如 "C"）
        options: 选项列表，如 [{"A": "选项A"}, {"B": "选项B"}, ...]
        grading_mode: 评分模式
        template_metadata: 自定义正则表达式元数据（可选）
    
    Returns:
        (reward, extracted_answer): reward 是 0.0 或 1.0，extracted_answer 是提取出的答案
    """
    allowed_letters = _get_allowed_letters_from_options(options)
    
    pred: Optional[str] = None
    
    # Check for template_metadata first (highest priority)
    if template_metadata and "output_regex" in template_metadata:
        regex_pattern = template_metadata["output_regex"]
        pred = _parse_answer_with_custom_regex(model_output, regex_pattern, allowed_letters, options)
    
    # Fallback to existing grading_mode logic if template_metadata didn't work
    if pred is None:
        if grading_mode == "strict_single_letter_boxed":
            pred, _, _ = _parse_answer_letter_strict_boxed(model_output, allowed_letters)
        elif grading_mode == "lenient_boxed":
            pred, _, _ = _parse_answer_letter_strict_boxed(model_output, allowed_letters)
            if pred is None:
                letter_from_text = _match_option_text(model_output, options, allowed_letters)
                if letter_from_text is not None:
                    pred = letter_from_text
        elif grading_mode == "lenient_answer_colon":
            m = ANSWER_COLON_PATTERN.search(model_output)
            if m:
                candidate = _strip_latex_wrappers(m.group(1)).strip()
                if len(candidate) == 1 and candidate.isalpha():
                    letter_up = candidate.upper()
                    if letter_up in allowed_letters:
                        pred = letter_up
                if pred is None:
                    cand_norm = _normalize_for_match(candidate)
                    for entry in options or []:
                        for k, v in entry.items():
                            k_up = k.upper()
                            if k_up in allowed_letters and _normalize_for_match(v) == cand_norm:
                                pred = k_up
                                break
                        if pred is not None:
                            break
    
    gold = (expected_answer or "").strip().upper()
    is_correct = (pred == gold) if (pred is not None and gold) else False
    reward = 1.0 if is_correct else 0.0
    
    return reward, pred


def score_fn(
    model_output: str,
    expected_answer: str,
    options: Optional[list[dict[str, str]]] = None,
    grading_mode: Literal[
        "strict_single_letter_boxed",
        "lenient_boxed",
        "lenient_answer_colon",
    ] = "strict_single_letter_boxed",
    template_metadata: Optional[dict] = None,
) -> float:
    """
    MCQA answer scoring function for RL framework
    
    Args:
        model_output: 模型输出的文本（包含推理过程和最终答案）
        expected_answer: 标准答案（单个字母，如 "C"）
        options: 选项列表，如 [{"A": "选项A"}, {"B": "选项B"}, {"C": "选项C"}, {"D": "选项D"}]
        grading_mode: 评分模式
            - "strict_single_letter_boxed": 要求 \\boxed{C} 格式（单个大写字母）
            - "lenient_boxed": 允许 \\boxed{C} 或 \\boxed{选项文本}
            - "lenient_answer_colon": 允许 Answer: C 或 Answer: 选项文本
        template_metadata: 自定义正则表达式（可选），格式: {"output_regex": "..."}
    
    Returns:
        float: 1.0 (正确) 或 0.0 (错误)
    
    Example:
        >>> score_mcqa_answer(
        ...     "After analysis, the answer is \\boxed{C}",
        ...     "C",
        ...     [{"A": "选项A"}, {"B": "选项B"}, {"C": "选项C"}, {"D": "选项D"}]
        ... )
        1.0
    """
    reward, _ = _verify_answer(model_output, expected_answer, options, grading_mode, template_metadata)
    return reward
