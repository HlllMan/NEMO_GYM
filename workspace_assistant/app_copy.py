#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
 Workplace Assistant Answer Scoring Function - Simplified from app.py

 原始文件: resources_servers/workplace_assistant/app.py (124 lines)
 简化后: app_copy.py

 主要改动:
 【删除】
 - 所有 async/await 异步操作
 - FastAPI 和 Server 相关代码 (SimpleResourcesServer, setup_webserver, route_to_python_function, seed_session)
 - Pydantic 数据模型类 (Request/Response/VerifyRequest/VerifyResponse)
 - 配置类 (WorkbenchResourcesServerConfig)

 【保留】
 - 核心打分逻辑：通过 utils.is_correct 对比“工具调用导致的环境状态变化”

 【新增】
 - score_workbench() 函数：简化的打分接口，可直接用于 RL 框架

 大模型输出 / 打分输入格式说明:
 - 数据集提供:
   - ground_truth: 正确的工具调用序列 (list[dict[str, str]] 或其 JSON 字符串)
 - 模型需要输出:
   - predicted_actions: 实际调用的工具序列 list[dict[str, str]]，每个元素形如:
       {
           "name": "calendar_create_event",
           "arguments": "{ \"event_name\": \"...\", ... }"
       }
 - Rule-based reward:
   - 使用 utils.is_correct(predicted_actions, ground_truth_actions, error)
   - 检查执行这些 tool calls 后，所有工作台数据库状态是否与 ground truth 一致
   - 返回值: 1.0 (完全一致) 或 0.0 (不一致)
"""

from typing import Any, Dict, List, Union

from resources_servers.workplace_assistant.utils import is_correct


def score_fn(
    ground_truth_actions: Union[List[Dict[str, Any]], str],
    predicted_actions: List[Dict[str, Any]],
    error: str | None = None,
) -> float:
    """
    简化版打分函数：直接复用 is_correct 作为 rule-based reward。

    Args:
        ground_truth_actions:
            - 正确的工具调用序列
            - 类型可以是 list[dict[str, Any]] 或 JSON 字符串
        predicted_actions:
            - 模型实际输出的工具调用序列，list[dict]，每个元素通常包含:
                {
                    "name": "<tool_name>",
                    "arguments": "<JSON-encoded-arguments>"
                }
        error:
            - 可选的错误信息（如果模型在生成过程中已经出错，可以传入非空字符串直接判 0 分）

    Returns:
        reward: 1.0 (任务完成，环境状态一致) 或 0.0 (未完成 / 出错)
    """

    # is_correct 的签名为:
    #   is_correct(predicted_actions, ground_truth_actions, error: str) -> bool
    # 这里不改变其内部逻辑，只做一个轻量封装。
    if error:
        # 如果外部已经判定有错误，直接传给 is_correct
        return float(is_correct(predicted_actions, ground_truth_actions, error))

    return float(is_correct(predicted_actions, ground_truth_actions, ""))
