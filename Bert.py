# -*- coding: utf-8 -*-
import time
import os
import numpy as np
from bert_serving.client import BertClient
import json
from typing import Dict

bc = BertClient(ip='10.201.16.3')
topk = 10
# 项目的根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 针对json格式的文本再次处理
def get_dict():
    docs = []

    f = open("./data/train_data.txt", 'r', encoding='utf8')
    for idx, line in enumerate(f):
        """if idx % 1000 == 0:
            print('已完成：{}'.format(idx))"""
        # 转换为python字典

        line: Dict = json.loads(line)
        docs.append(line["fullText"])


    return docs

def case_recommend(target):
    sentences=get_dict()
    sentences_vec = bc.encode(sentences)
    # todo: 保存sentences_vec...
    f = open('emb.txt', 'w', encoding='utf8')
    f.write(json.dumps(sentences_vec.tolist(), ensure_ascii=False))
    #sentences_vec = json.load(open('emb.txt', 'r', encoding='utf8'))
    test_vec = bc.encode(target)
    score = np.sum(test_vec * sentences_vec, axis=1) / np.linalg.norm(sentences_vec, axis=1)
    topk_idx = np.argsort(score)[::-1][:topk]
    for idx in topk_idx:
        print(' %s\t%s' % (score[idx], sentences[idx][0:50]))

if __name__ == '__main__':

    # 测试数据集
    test_set = json.load(open(os.path.join(project_root, 'Text_Similarity/test_data/test_data.json'), 'r', encoding='utf8'))

    for i in dict(test_set).values():
        start_time = time.time()
        a=[]
        a.append(i)
        case_recommend(a)
        print('time cost：', round(time.time() - start_time, 2))







