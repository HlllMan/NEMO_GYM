# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Structured Outputs Answer Scoring Function - Simplified from app.py

原始文件: resources_servers/structured_outputs/app.py (100 lines)
简化后: app_copy.py

主要改动:
【删除】
- 所有 async/await 异步操作
- FastAPI 和 Server 相关代码 (SimpleResourcesServer, setup_webserver)
- Pydantic 数据模型类 (Request/Response classes)
- 配置类 (StructuredOutputsResourcesServerConfig)

【保留】
- 所有 JSON Schema 验证逻辑（完全保留原始实现）
- openapi_schema_validator 库的使用
- strictify_schema_json 方法
- evaluate_structured_output_response_json 方法

【新增】
- score_structured_output() 函数：简化的打分接口，可直接用于 RL 框架

用法:
    from app_copy import score_structured_output
    
    reward = score_structured_output(
        model_output='{"printerModel": "Prusa i3 MK4", ...}',
        schema_str='{"type": "object", "required": [...], ...}',
        schema_type="json"
    )
    # 返回: 1.0 (符合 schema) 或 0.0 (不符合 schema)

大模型输出格式要求:
- 模型输出是一个纯文本字符串（str），包含 JSON 格式的响应
- 不需要 role/user/assistant 等结构，只需要最终的 JSON 文本内容
- 例如：model_output = '{"printerModel": "Prusa i3 MK4", "buildVolume": {"x": 250, "y": 210, "z": 200}, ...}'
- 输出必须是有效的 JSON 格式，且必须符合提供的 schema_str 约束
- 验证会检查：
  * JSON 格式是否正确
  * 是否包含所有 required 字段
  * 字段类型是否符合 schema
  * 数值是否在 min/max 范围内
  * enum 值是否在允许列表中
  * 是否没有额外的属性（如果 additionalProperties: false）
"""

import json
from typing import Any, Dict

from openapi_schema_validator import validate as validate_against_schema_openapi


class StructuredOutputsVerifier:
    """结构化输出验证器（保留原始类名）"""
    
    def strictify_schema_json(self, schema: Dict[str, Any]):
        """Make a schema strict as per OpenAPI guidelines"""
        if isinstance(schema, Dict):
            if "properties" in schema:
                schema["required"] = list(schema["properties"])
                schema["additionalProperties"] = False
            for k, v in schema.items():
                self.strictify_schema_json(v)
    
    def evaluate_structured_output_response_json(self, schema_str: str, response_text: str) -> float:
        """
        评估模型输出是否符合 JSON Schema
        
        Args:
            schema_str: JSON Schema 的字符串表示
            response_text: 模型的 JSON 输出文本
        
        Returns:
            reward score: 1.0 (符合) 或 0.0 (不符合)
        """
        try:
            schema = json.loads(schema_str)
            self.strictify_schema_json(schema)
            response_obj = json.loads(response_text)
            validate_against_schema_openapi(response_obj, schema)
            return 1.0
        except Exception as e:
            # 在调试模式下，可以打印错误信息
            # 注意：生产环境应该静默返回0.0
            import os
            if os.getenv('DEBUG_STRUCTURED_OUTPUTS') == '1':
                print(f"验证失败: {e}")
                import traceback
                traceback.print_exc()
            return 0.0


# 单例模式：避免重复初始化
_global_verifier = None


def get_verifier() -> StructuredOutputsVerifier:
    """获取全局验证器实例（单例模式）"""
    global _global_verifier
    if _global_verifier is None:
        _global_verifier = StructuredOutputsVerifier()
    return _global_verifier


def score_fn(
    model_output: str,
    schema_str: str,
    schema_type: str = "json"
) -> float:
    """
    评分函数：检查模型输出是否符合结构化输出要求（JSON Schema）
    
    Args:
        model_output: 模型的文本输出（应该是 JSON 格式的字符串）
        schema_str: JSON Schema 的字符串表示
        schema_type: Schema 类型，目前只支持 "json"
    
    Returns:
        reward score (float): 1.0 (符合 schema) 或 0.0 (不符合 schema)
    
    Example:
        >>> schema = '{"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}}'
        >>> output = '{"name": "test"}'
        >>> reward = score_structured_output(output, schema, "json")
        >>> print(reward)  # 1.0
    """
    if schema_type != "json":
        raise NotImplementedError(f"SchemaType must be 'json', got {schema_type}")
    
    verifier = get_verifier()
    reward = verifier.evaluate_structured_output_response_json(schema_str, model_output)
    return reward
