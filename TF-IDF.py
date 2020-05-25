# -*- coding:utf-8 -*-

import codecs
from typing import Dict
import json
import os
import jieba
import joblib
import time
import torch
import gensim
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.doc2vec import Doc2Vec, LabeledSentence

# 项目的根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model, doc_matrix, id2case = None, None, None

def load_stop_word():
    """
    加载停用词表，返回一个列表
    :return:
    """
    file_path = os.path.join(project_root, "Text_Similarity/dict/stopword.txt")
    f = open(file_path, 'r', encoding='utf8')
    stopword = [line.strip() for line in f.readlines()]
    return stopword

def cut(para, stop_word):
    """
    对输入的文本进行分词，单线程进行分词
    :param para: 传入的参数，如果传入一个string，即对string分词；
    :param stop_word:停用词
    :return: 分完词的列表
    """
    text_list = []
    if isinstance(para, str):
        string = para.replace('\u3000', '').replace(' ', '').replace("\n", '')
        cut_ = jieba.cut(string)
        cut_list = [word for word in cut_ if word not in stop_word]
        text_list.append(' '.join(cut_list))
        return text_list
    else:
        raise ValueError("传入的参数必须是string类型!")





def cut_train_data(file):
    """
    对指定的一个格式的文件进行分词处理,格式如下：
    {
        "id":"受案号",
        "fullText":"全文描述",
        "accusation":["罪名1", "罪名2", ...],
        "":""
    }
    并且在这个过程中会严格的建立双向索引字典
    字典的格式为int(数字，从0开始)-->str(案号)和str(案号)-->int(数字，从0开始)
    :param file: 输入训练的数据
    :return: 分完词的列表，每个元素为str，str是为以空格分隔的分词字符串
    """
    # 项目的停用词列表
    stop_word = load_stop_word()
    # id到案号的映射
    id2case = dict()
    # 案号到id的映射
    case2id = dict()
    # 分完词的矩阵，其索引严格对应两个字典
    case_cut_matrix = []
    f = open(file, 'r', encoding='utf8')
    for idx, line in enumerate(f):
        if idx % 1000 == 0:
            print('已完成：{}'.format(idx))
        # 转换为python字典
        line: Dict = json.loads(line)
        # 案号
        key = line["id"]
        id2case[len(id2case)] = key
        case2id[key] = len(case2id)
        # 对一个文档进行分词后的产生的列表
        cut_case = cut(line["fullText"], stop_word)
        assert len(cut_case) == 1
        # 加入分词的总文档中
        case_cut_matrix.append(cut_case[0])
    # 断言，两个字典必须是相等长度的
    assert len(case2id) == len(id2case)
    assert len(case_cut_matrix) == len(case2id)
    print(len(case_cut_matrix))
    print('分完词后总案件数量为：{}'.format(len(case_cut_matrix)))
    # 持久化两个字典文件，实现罪名到id的映射
    f = open(os.path.join(project_root, "Text_Similarity/dict/id2case.json"), 'w', encoding='utf8')
    f1 = open(os.path.join(project_root, "Text_Similarity/dict/case2id.json"), 'w', encoding='utf8')
    f.write(json.dumps(id2case, ensure_ascii=False, indent=4))
    f1.write(json.dumps(case2id, ensure_ascii=False, indent=4))
    # 以下的操作是非必要的，因为分完词直接return可能时间太长，故先保存分完词的分词列表
    # 这里可能需要判断是不是存在分完词的列表了
    # if os.path.exists()
    f2 = open(os.path.join(project_root, "Text_Similarity/dict/cut_matrix.txt"), 'w', encoding="utf8")
    f2.write(json.dumps(case_cut_matrix, ensure_ascii=False))
    return case_cut_matrix


def create_model(train_data):
    """
    训练TFIDF模型，并且保存模型的参数
    :param train_data: 训练数据
    :return: None
    """
    # tfidf模型初始化
    model = TfidfVectorizer(
        token_pattern=r"(?u)\b\w\w+\b",
        min_df=0.01,
        max_features=3000,
        # ngram_range=(1, 3),
        use_idf=1,
        # smooth_idf=1
    )
    # 使用训练数据得到的tfidf矩阵
    matrix = model.fit_transform(train_data)
    print(matrix.shape)
    # 得到的稀疏矩阵
    sparse.save_npz(os.path.join(project_root, 'Text_Similarity/model/matrix.npz'), matrix)
    # 持久化模型参数
    joblib.dump(model, os.path.join(project_root, 'Text_Similarity/model/tfidf.model'), compress=3)


def start_train(data_file):
    """
    全量训练数据，生成一些模型文件，外部调用这个接口来开始进行模型的训练
    :param data_file: 格式化的数据文件
    :return: None
    """
    # 如果不存在分完词的矩阵，进行文档的分词操作
    if not os.path.exists(os.path.join(project_root, 'dict/cut_matrix.txt')):
        print('正在进行分词操作！')
        case_cut_matrix = cut_train_data(data_file)
        assert isinstance(case_cut_matrix, list), '文档持久化格式错误，需要是list格式！'
        print('开始训练模型a！')
        start_time = time.time()
        create_model(case_cut_matrix)
        print('模型训练完成，消耗时间为：{}秒'.format(round(time.time() - start_time, 3)))
    # 存在分完词的矩阵，不存在模型，就直接进行tfidf训练
    if os.path.exists(os.path.join(project_root, 'dict/cut_matrix.txt')) \
            and not os.path.exists(os.path.join(project_root, 'model/tfidf.model')):
        print('已经存在分完词的矩阵，进入训练步骤！')
        # 分完词的文档列表
        data = json.load(open(os.path.join(project_root, 'dict/cut_matrix.txt'), 'r', encoding='utf8'))
        assert isinstance(data, list), '文档持久化格式错误，需要是list格式！'
        print('开始训练模型b！')
        start_time = time.time()
        create_model(data)
        print('模型训练完成，消耗时间为：{}秒'.format(round(time.time() - start_time, 3)))

def new_case_recommend(test_corpus, model, doc_matrix, file_index, stop_word=load_stop_word(), k=10):
    """
    计算给定文档在指定语料库中的相似文档，核心函数，效率不够可能要c++重写
    :param stop_word: 停用词列表
    :param test_corpus:输入的语料
    :param k: 返回前k个相似的文档
    :param model: 训练好的TFIDF模型
    :param doc_matrix: 训练好的文档向量
    :param file_index: 文件，存储文档的索引
    :return: 一个余弦相似度的距离列表
    """
    # tfidf模型所有特征关键字的列表
    # feature_name = model.get_feature_names()
    # print(len(feature_name))
    # 对新文档进行分词操作, 返回一个列表
    test_doc = cut(test_corpus, stop_word)
    # 对new文档进行tfidf模型的向量化，得到一个矩阵
    case_matrix = model.transform(test_doc)
    # 计算向量相似度
    distance = cosine_similarity(case_matrix, doc_matrix)
    value, index = torch.topk(torch.Tensor(distance).squeeze(), k)
    # 这个测试文档与前k个文档的相似距离
    value = value.tolist()
    # k个相似文档的索引
    index = index.tolist()

    for i, dis in zip(index, value):
        # 文件受案号
        file = file_index[str(i)]
        f = open("../Text_Similarity/data/train_data.json", 'r', encoding='utf8')
        for idx, line in enumerate(f):
            if idx == i:
                line: Dict = json.loads(line)
                file = line["fullText"][0:50]
                break
            # 转换为python字典
        f.close()
        print("相似的文档：{},相似距离：{}".format(file, dis))


# 加载配置文件等
def load_config():
    """
    加载一些配置文件和模型参数
    :return: None
    """
    global model, doc_matrix, id2case
    # 模型参数文件
    model = joblib.load(os.path.join(project_root, 'Text_Similarity/model/tfidf.model'))
    print('模型参数加载完毕！')
    # 文档进行向量化后的文件
    doc_matrix = sparse.load_npz(os.path.join(project_root, 'Text_Similarity/model/matrix.npz'))
    print('文档向量加载完毕！')
    # 文件索引表
    id2case = json.load(open(os.path.join(project_root, 'Text_Similarity/dict/id2case.json'), 'r', encoding='utf8'))
    jieba.initialize()
    print('文件索引表加载完毕！')



if __name__ == '__main__':
    start_train('../Text_Similarity/data/train_data.json')
    # 测试数据集
    test_set = json.load(
        open(os.path.join(project_root, 'Text_Similarity/data/test_data.json'), 'r', encoding='utf8'))

    for i in dict(test_set).values():
        load_config()
        #start_time = time.time()
        new_case_recommend(i, model, doc_matrix, id2case, k=10)
        #print('计算相似文档时间消耗为：{}秒'.format(round(time.time() - start_time, 2)))
