import sys
import os

# 适配路径
sys.path.append("/public/data0/HOME/jdnlp1004/miaoji.norman/miaoji")

print("🚀 正在对 NEMO_GYM 全量 6 模块进行压力测试...\n")

try:
    from math_with_judge.app_copy import score_fn as math_fn
    from code_gen.app_copy import score_fn as code_fn
    from instruction_following.app_copy import score_fn as if_fn
    from structured_outputs.app_copy import score_fn as so_fn
    from mcqa.app_copy import score_fn as mcqa_fn
    from workspace_assistant.app_copy import score_fn as wa_fn
except ImportError as e:
    print(f"❌ 导入失败，请检查 PYTHONPATH 或文件夹完整性: {e}")
    sys.exit(1)

# 全量测试用例
test_suite = [
    {"name": "MATH", "fn": math_fn, "out": "x=5", "rew": {"answer": "5"}},
    {"name": "CODE", "fn": code_fn, "out": "print(1)", "rew": {"unit_tests": "pass"}},
    {"name": "IF",   "fn": if_fn,   "out": "Short.", "rew": {}},
    {"name": "SO",   "fn": so_fn,   "out": '{"a":1}', "rew": {"schema": {"type":"object"}}},
    # --- 补全这两项 ---
    {
        "name": "MCQA", 
        "fn": mcqa_fn, 
        "out": "(A)", 
        "rew": {"answer": "A"}
    },
    {
        "name": "WA",   
        "fn": wa_fn,   
        "out": "Action: send_email(to='boss', body='Hi')", 
        "rew": {"expected_action": "send_email"} 
    }
]

for case in test_suite:
    try:
        # 注意：WA 的 rew_keys 结构在仓库中可能更复杂，这里做冒烟测试
        score = case['fn'](model_output=case['out'], **case['rew'] if case['rew'] else {})
        print(f"✅ [{case['name']:<4}] 运行成功 | 得分: {score}")
    except Exception as e:
        # 如果是 WA 报错，可能是因为需要特定的 mock 数据库状态，但只要没报 ImportError 就说明环境 OK
        print(f"⚠️  [{case['name']:<4}] 运行提示: {e}")

print("\n🎉 如果 [WA] 没报 ModuleNotFoundError，说明你的 Agent 环境也配好了。")