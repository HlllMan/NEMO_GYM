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
Competitive Coding Answer Scoring Function - Simplified from app.py

原始文件: resources_servers/code_gen/app.py (155 lines)
简化后: app_copy.py

主要改动:
【删除】
- 所有 async/await 异步操作
- FastAPI 和 Server 相关代码 (SimpleResourcesServer, CompCodingResourcesServer, CompCodingResourcesServerConfig)
- Pydantic 数据模型类 (CompCodingRunRequest, CompCodingVerifyRequest, CompCodingVerifyResponse)
- Ray 远程调用 (check_correctness_remote)，改用同步版本
- Semaphore 并发控制（简化版本不需要）

【保留】
- 核心打分逻辑：使用 extract_code 提取代码，使用 check_correctness 执行测试
- UnitTests 数据模型（用于验证输入格式）

【新增】
- score_code_gen() 函数：简化的打分接口，可直接用于 RL 框架

大模型输出 / 打分输入格式说明:
- 数据集提供:
  - verifier_metadata["unit_tests"]: 单元测试用例
    {
      "inputs": ["test_input_1", "test_input_2", ...],
      "outputs": ["expected_output_1", "expected_output_2", ...],
      "fn_name": Optional[str]  # 可选，函数名
    }
- 模型需要输出:
  - model_output: 纯文本字符串，包含 Python 代码
  - 代码应该在 ```python 代码块中，例如:
    ```python
    # Your code here
    def solve():
        ...
    ```
  - 如果没有代码块，会尝试提取整个输出作为代码
- Rule-based reward:
  - 使用 extract_code 从模型输出中提取 Python 代码
  - 使用 check_correctness 执行代码并运行所有单元测试
  - 返回值: 1.0 (所有测试通过) 或 0.0 (任何测试失败 / 代码提取失败)
"""
import json
import multiprocessing
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 添加当前文件所在目录到 sys.path，以便导入 lcb_integration
_current_dir = Path(__file__).parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

# 导入代码提取工具（不依赖 ray）
from lcb_integration.extraction_utils import LMStyle, extract_code

# 导入测试工具（不依赖 ray）
from lcb_integration.testing_util import run_test


def _temp_run(in_outs, generation, debug, result, metadata_list, timeout):
    """临时运行函数，用于多进程执行测试"""
    try:
        # 调用 run_test，它会执行所有测试用例
        res, metadata = run_test(in_outs, test=generation, debug=debug, timeout=timeout)
        result.append(res)
        metadata_list.append(metadata)
    except Exception as e:
        if debug:
            print(f"测试执行错误: {e}")
        result.append([False] * len(in_outs["inputs"]))
        metadata_list.append(None)


def check_correctness(sample, generation, timeout, debug=True):
    """
    检查代码正确性（同步版本，不依赖 ray）。
    
    Args:
        sample: 包含 "input_output" 字段的字典，input_output 是 JSON 字符串
        generation: 要测试的代码字符串
        timeout: 每个测试用例的超时时间（秒）
        debug: 是否输出调试信息
    
    Returns:
        (result, metadata): 
        - result: 测试结果列表，每个元素是 True/False
        - metadata: 元数据字典
    """
    # 解析 JSON
    try:
        in_outs = json.loads(sample["input_output"])
    except (ValueError, MemoryError):
        return [-1], None

    # 使用多进程执行测试（避免阻塞）
    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    p = multiprocessing.Process(
        target=_temp_run,
        args=(in_outs, generation, debug, result, metadata_list, timeout),
    )
    p.start()
    p.join(timeout=(timeout + 1) * len(in_outs["inputs"]) + 5)
    if p.is_alive():
        p.kill()
    if not result:
        # 超时或出错，认为所有测试失败
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        metadata_list = [None]
        if debug:
            print("global timeout")
    return result[0], metadata_list[0]


class UnitTests:
    """单元测试数据模型（简化版，不使用 Pydantic）"""
    def __init__(self, inputs: List[str], outputs: List[str], fn_name: Optional[str] = None):
        self.inputs = inputs
        self.outputs = outputs
        self.fn_name = fn_name
    
    def model_dump_json(self) -> str:
        """转换为 JSON 字符串，用于 check_correctness"""
        data = {
            "inputs": self.inputs,
            "outputs": self.outputs,
        }
        if self.fn_name is not None:
            data["fn_name"] = self.fn_name
        return json.dumps(data)


def score_fn(
    model_output: str,
    unit_tests: Dict[str, Any],
    unit_test_timeout_secs: int = 10,
    debug: bool = False,
) -> float:
    """
    简化版打分函数：直接使用 check_correctness 作为 rule-based reward。

    Args:
        model_output:
            - 模型输出的纯文本字符串，应包含 Python 代码
            - 代码应该在 ```python 代码块中
        unit_tests:
            - 单元测试用例字典，格式:
              {
                  "inputs": ["test_input_1", ...],
                  "outputs": ["expected_output_1", ...],
                  "fn_name": Optional[str]
              }
        unit_test_timeout_secs:
            - 每个测试用例的超时时间（秒），默认 10
        debug:
            - 是否输出调试信息，默认 False

    Returns:
        reward: 1.0 (所有测试通过) 或 0.0 (任何测试失败 / 代码提取失败)
    """
    # 1) 检查模型输出是否为空
    if not model_output or not model_output.strip():
        return 0.0

    # 2) 验证并解析单元测试
    try:
        tests = UnitTests(
            inputs=unit_tests["inputs"],
            outputs=unit_tests["outputs"],
            fn_name=unit_tests.get("fn_name"),
        )
    except (KeyError, TypeError) as e:
        if debug:
            print(f"❌ 单元测试格式错误: {e}")
        return 0.0

    # 3) 提取代码（从代码块或原始输出）
    code = extract_code(model_output, LMStyle.OpenAIChat)
    if not code:
        if debug:
            print("❌ 无法从模型输出中提取代码")
        return 0.0

    # 4) 执行测试
    try:
        sample = {"input_output": tests.model_dump_json()}
        result, metadata = check_correctness(
            sample=sample,
            generation=code,
            timeout=unit_test_timeout_secs,
            debug=debug,
        )
        
        # result 是一个列表，每个元素对应一个测试用例的结果
        # True 表示通过，False/其他值表示失败
        if result and all(r == True for r in result):
            return 1.0
        else:
            return 0.0
    except Exception as e:
        if debug:
            print(f"❌ 执行测试时出错: {e}")
        return 0.0
