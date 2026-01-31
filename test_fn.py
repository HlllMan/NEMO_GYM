"""
使用示例
"""

from mjnemogym import score_fn_dict



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
