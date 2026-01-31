"""
使用示例
"""

from math_with_judge.app_copy import score_fn as math_score_fn
from code_gen.app_copy import score_fn as code_score_fn
from mcqa.app_copy import score_fn as mcqa_score_fn
from instruction_following.app_copy import score_fn as if_score_fn
from structured_outputs.app_copy import score_fn as so_score_fn
from workspace_assistant.app_copy import score_fn as wa_score_fn   

score_fn_dict={
    "math": math_score_fn,
    "code": code_score_fn,
    "mcqa": mcqa_score_fn,
    "if": if_score_fn,
    "so": so_score_fn,
    "wa": wa_score_fn,
}



def main():
    
    # TODO: 在这里定义大模型的回答接口预留

    obj = {} #数据集文件中的一行
    prompt = obj["common"]["prompt"]
    model_output = ""  # 大模型的回答
    
    score_fn_name = obj["common"]["score_fn_name"] # 评分函数名称
    rew_keys = obj["rew_keys"]
    
    score = score_fn_dict[score_fn_name](model_output=model_output, **rew_keys)


if __name__ == "__main__":
    main()
