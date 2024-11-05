import torch
from flask import Flask, request
from importlib import import_module
import numpy as np
CLS =  "[CLS]"
id2name = {0: 'finance', 1: 'realty', 2: 'stocks', 3: 'education', 4: 'science',
              5: 'society', 6: 'politics', 7: 'sports', 8: 'game', 9: 'entertainment'}

def inference(model, config, input_text, pad_size=32):
    """
    模型推理函数，用于对输入文本进行情感分析的推理。
    参数：
    - model: 已加载的情感分析模型。
    - config: 模型配置信息。
    - input_text: 待分析的文本。
    - pad_size: 指定文本填充的长度。
    """
# 输入文本分词预处理
    content = config.tokenizer.tokenize(input_text)
    content = [CLS] + content
    seq_len = len(content)
    token_ids = config.tokenizer.convert_tokens_to_ids(content)
# 规范文本长度
    if seq_len < pad_size:
        mask = [1] * len(token_ids) + [0] * (pad_size - seq_len)
        token_ids += [0] * (pad_size - seq_len)
    else:
        mask = [1] * pad_size
        token_ids = token_ids[:pad_size]
        seq_len = pad_size
# 文本转张量
    x = torch.LongTensor(token_ids).to(config.device)
    seq_len = torch.LongTensor(seq_len).to(config.device)
    mask = torch.LongTensor(mask).to(config.device)
# 格式对齐
    x = x.unsqueeze(0)
    seq_len = seq_len.unsqueeze(0)
    mask = mask.unsqueeze(0)
    data = (x, seq_len, mask)

# 模型预测
    output = model(data)
# 获取模型预测结果id
    predict_result = torch.max(output.data, 1)[1]
    return predict_result

# 创建Flask应用
app = Flask(__name__)

# 定义路由，接收POST请求并进行推理
@app.route('/v1/main_server/', methods=["POST"])
def main_server():
    # 从POST请求中获取用户ID和文本数据
    uid = request.form['uid']
    text = request.form['text']
    # 调用推理函数获取预测结果
    res = inference(model, config, text)
    result= id2name[res.item()]
    # print(result)
    return result

#预测主函数
if __name__=='__main__':
    # 加载模型配置
    model_name ='bert'
    x = import_module('models.'+ model_name)
    config = x.Config()
    # 设置随机种子
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    # 创建并初始化模型
    model = x.Model(config).to(config.device)
    model.load_state_dict(torch.load(config.save_path,map_location=config.device))
    # 输入待分析文本
    input_text = '日本地震：金吉列关注在日学子系列报道'

    #进行模型推理
    # res = inference(model,config,input_text)
    # 获取类别名称
    # result= id2name[res.item()]
    # print(result)

    app.run()



