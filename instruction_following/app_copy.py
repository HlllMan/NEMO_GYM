# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Instruction Following Answer Scoring Function - Simplified from app.py

原始文件: resources_servers/instruction_following/app.py (144 lines)
简化后: app_copy.py

主要改动:
【删除】
- 所有 async/await 异步操作
- FastAPI 和 Server 相关代码 (SimpleResourcesServer, setup_webserver)
- Pydantic 数据模型类 (Request/Response classes)
- 配置类 (InstructionFollowingResourcesServerConfig)

【保留】
- 所有指令验证逻辑（完全保留原始实现）
- verifiable_instructions 库的使用
- NLTK 数据初始化
- 两种评分模式逻辑（binary, fraction）

【新增】
- score_instruction_following() 函数：简化的打分接口，可直接用于 RL 框架

用法:
    from app_copy import score_instruction_following
    
    reward = score_instruction_following(
        model_output="<<Title>>\n\nParagraph 1\n\n***\n\nParagraph 2",
        instruction_id_list=["paragraphs:paragraphs", "detectable_format:title"],
        kwargs=[{}, None],
        grading_mode="binary"
    )
    # 返回: 1.0 (正确) 或 0.0 (错误)

大模型输出格式要求:
- 模型输出是一个纯文本字符串（str），包含模型的完整响应文本
- 不需要 role/user/assistant 等结构，只需要最终的文本内容
- 例如：model_output = "<<Title>>\n\nParagraph 1 content.\n\n***\n\nParagraph 2 content."

指令类型（instruction_id_list）示例:
- paragraphs:paragraphs - 段落数量和分隔符要求
- detectable_format:title - 格式要求（如 <<title>>）
- length_constraints:number_words - 字数限制
- keywords:forbidden_words - 禁止使用的关键词
- last_word:last_word_answer - 最后单词要求
- first_word:first_word_sent - 首词要求
- 等等（详见 verifiable_instructions 库）

验证函数会检查模型输出是否遵循所有给定的指令，返回 reward score (0.0-1.0)
"""

from typing import List, Literal

from verifiable_instructions import instructions_registry


# 全局变量：确保 NLTK 数据已下载
_nltk_initialized = False


def _ensure_nltk_data():
    """Download required NLTK data at startup."""
    global _nltk_initialized
    if _nltk_initialized:
        return
    
    try:
        import nltk
        nltk.download("punkt_tab", quiet=True)
        _nltk_initialized = True
    except ImportError:
        # ifbench not available, skip
        pass
    except Exception as e:
        print(f"NLTK setup warning: {e}")


class InstructionFollowingVerifier:
    """指令遵循验证器（保留原始类名）"""
    
    def __init__(self):
        _ensure_nltk_data()
    
    def _verify_instructions(
        self,
        final_response_text: str,
        instruction_list: List[str],
        kwargs_list: List[dict],
        grading_mode: Literal["binary", "fraction"] = "binary"
    ) -> float:
        """
        验证模型输出是否遵循所有指令
        
        Args:
            final_response_text: 模型的最终文本输出
            instruction_list: 指令类型列表
            kwargs_list: 每个指令的参数字典列表
            grading_mode: 评分模式 ("binary" 或 "fraction")
        
        Returns:
            reward score (0.0 或 1.0 for binary, 0.0-1.0 for fraction)
        """
        is_following_list = []
        
        for instruction_id, kwargs in zip(instruction_list, kwargs_list):
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
                if instruction.check_following(final_response_text):
                    is_following_list.append(True)
                else:
                    is_following_list.append(False)
            
            except Exception as e:
                # If there's an error processing the instruction, mark as failed
                print(f"Error processing instruction {instruction_id}: {e}")
                is_following_list.append(False)
        
        # Calculate overall success
        if grading_mode == "binary":
            reward = float(all(is_following_list))
        elif grading_mode == "fraction":
            reward = float((sum(is_following_list) / len(is_following_list)) if is_following_list else 0.0)
        else:
            raise ValueError(f"Invalid reward mode: {grading_mode}")
        
        return reward


# 单例模式：避免重复初始化
_global_verifier = None


def get_verifier() -> InstructionFollowingVerifier:
    """获取全局验证器实例（单例模式）"""
    global _global_verifier
    if _global_verifier is None:
        _global_verifier = InstructionFollowingVerifier()
    return _global_verifier


def score_fn(
    model_output: str,
    instruction_id_list: List[str],
    kwargs: List[dict],
    grading_mode: Literal["binary", "fraction"] = "binary"
) -> float:
    """
    评分函数：检查模型输出是否遵循指令
    
    Args:
        model_output: 模型的文本输出（纯文本字符串）
        instruction_id_list: 指令类型列表，例如 ["paragraphs:paragraphs", "detectable_format:title"]
        kwargs: 每个指令的参数字典列表，例如 [{}, None, {"num_words": 204}]
        grading_mode: 评分模式
            - "binary": 所有指令必须遵循，reward = 1.0 或 0.0
            - "fraction": 遵循指令的比例，reward = 遵循数/总指令数
    
    Returns:
        reward score (float): 0.0-1.0
    
    Example:
        >>> reward = score_instruction_following(
        ...     model_output="<<Title>>\\n\\nPara 1\\n\\n***\\n\\nPara 2",
        ...     instruction_id_list=["paragraphs:paragraphs", "detectable_format:title"],
        ...     kwargs=[{}, None],
        ...     grading_mode="binary"
        ... )
        >>> print(reward)  # 1.0 或 0.0
    """
    verifier = get_verifier()
    reward = verifier._verify_instructions(
        final_response_text=model_output,
        instruction_list=instruction_id_list,
        kwargs_list=kwargs,
        grading_mode=grading_mode
    )
    return reward
