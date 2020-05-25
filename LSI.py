# coding:utf-8

import os
from typing import Dict
import codecs
import time
import json
import jieba
from gensim import corpora,models,similarities
from collections import defaultdict   #用于创建一个空的字典，在后续统计词频可清理频率少的词语
import heapq


# 项目的根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据预处理
def data_process(data):
    stopwords = codecs.open('dict/stopword.txt', 'r', encoding='utf8').readlines()
    stopwords = [w.strip() for w in stopwords]
    test_doc = []
    for line in data:
        line = jieba.lcut(line.strip())
        a=[]
        a.extend(w for w in line if w not in stopwords)
        test_doc.append(a)
    return test_doc


# 针对json格式的文本再次处理
def get_dict():
    docs = []
    f = open("../Text_Similarity/data/train_data.json", 'r', encoding='utf8')
    for idx, line in enumerate(f):
        if idx % 1000 == 0:
            print('已完成：{}'.format(idx))
        # 转换为python字典
        line: Dict = json.loads(line)
        docs.append(line["fullText"])

    texts = data_process(docs)
    return texts

# 开始训练模型
start_time0 = time.time()

# 获得处理后的数据
texts=get_dict()

# 计算词语的频率
frequency = defaultdict(int)
for text in texts:
    for word in text:
        frequency[word] += 1

# 对频率低的词语进行过滤（可选）
texts = [[word for word in text if frequency[word] > 10] for text in texts]

# 通过语料库将文档的词语进行建立词典，即字符串到id的映射
dictionary = corpora.Dictionary(texts)

# 将文档由字符串转化为id
corpus = [dictionary.doc2bow(text) for text in texts]

# 8. tfidf模型
tfidf = models.TfidfModel(corpus)

corpus_tfidf = tfidf[corpus]

lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=200)
# 模型的保存
lsi.save('model/test_lsi.model')
print('模型训练消耗为：{}秒'.format(round(time.time() - start_time0, 2)))

index = similarities.MatrixSimilarity(lsi[corpus])

stopwords = codecs.open('dict/stopword.txt', 'r', encoding='utf8').readlines()
stopwords = [w.strip() for w in stopwords]
# 文件索引表
id2case = json.load(open(os.path.join(project_root, 'Text_Similarity/dict/id2case.json'), 'r', encoding='utf8'))

# 测试数据集
test_set = json.load(
    open(os.path.join(project_root, 'Text_Similarity/data/test_data.json'), 'r', encoding='utf8'))

for target in dict(test_set).values():
    start_time=time.time()
    # 对测试案例文档的处理
    line = jieba.lcut(target.strip())
    query_doc = []
    query_doc.extend(w for w in line if w not in stopwords)

    query_bow = dictionary.doc2bow(query_doc)  # 文档转换成bow，向量矩阵
    query_lsi = lsi[query_bow]  # 得到新的主题分布
    sims = index[query_lsi]
    sims = sims.tolist()  # 将ndarray转换成数组

    # 得到相似度数组
    re1 = heapq.nlargest(10, sims)  # 求最大的10个元素，并排序
    re2 = map(sims.index, heapq.nlargest(10, sims))  # 求最大的10个索引    nsmallest与nlargest相反，求最小
    re3 = list(re2)  # 因为re2由map()生成的不是list，所以需要添加list()

    for count, sim in zip(re3, re1):
        file = id2case[str(count)]
        f = open("../Text_Similarity/data/train_data.json", 'r', encoding='utf8')
        for idx, line in enumerate(f):
            if idx == count:
                line: Dict = json.loads(line)
                file = line["fullText"][0:50]
                break
            # 转换为python字典
        f.close()
        print("相似的文档索引：{},相似距离：{}".format(file, sim))
    print('计算相似文档时间消耗为：{}秒'.format(round(time.time() - start_time, 2)))

