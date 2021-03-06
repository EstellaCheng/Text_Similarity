# coding:utf-8
import time
import codecs
import json
import gensim
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim import corpora, models
import numpy as np
import jieba
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import os
import torch
import heapq
from typing import Dict

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
    f = open("../Text_Similarity/data/data_0.5.txt", 'r', encoding='utf8')
    for idx, line in enumerate(f):
        if idx % 1000 == 0:
            print('已完成：{}'.format(idx))
        # 转换为python字典
        line: Dict = json.loads(line)

        docs.append(line["fullText"])

    results = data_process(docs)
    dictionary=Dictionary(results)
    return dictionary,results


##训练lda模型
def train_model():
    # 构建词典
    dictionary=get_dict()[0]   # 第一个返回值
    train=get_dict()[1]        # 第二个返回值

    # 基于词典，使【词】→【稀疏向量】，并将向量放入列表，形成【稀疏向量集】
    corpus = [dictionary.doc2bow(text) for text in train]
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=7,minimum_probability=0.001)

    #模型的保存/ 加载
    lda.save('model/test_lda.model')


def all_lda():
    data=get_dict()[1]
    lda = models.ldamodel.LdaModel.load('model/test_lda.model')
    dictionary = get_dict()[0]
    all_lda=[]

    for i in data:
        doc_bow = dictionary.doc2bow(i)  # 文档转换成bow，向量矩阵
        doc_lda = lda[doc_bow]
        all_lda.append(doc_lda)

    return all_lda

#计算两个文档的相似度
def lda_sim(s1,s2):
    stopwords = codecs.open('dict/stopword.txt', 'r', encoding='utf8').readlines()
    stopwords = [w.strip() for w in stopwords]
    lda = models.ldamodel.LdaModel.load('model/test_lda.model')
    dictionary=get_dict()[0]

    # 处理s1
    line = jieba.lcut(s1.strip())
    a = []
    a.extend(w for w in line if w not in stopwords)

    doc_bow = dictionary.doc2bow(a)  # 文档转换成bow，向量矩阵
    doc_lda = lda[doc_bow]  # 得到新文档的主题分布


    # 处理s1
    line2 = jieba.lcut(s2.strip())
    a2 = []
    a2.extend(w for w in line2 if w not in stopwords)

    doc_bow2 = dictionary.doc2bow(a2) # 文档转换成bow，向量矩阵
    doc_lda2 = lda[doc_bow2]  # 得到新文档的主题分布
    sim = gensim.matutils.cossim(doc_lda, doc_lda2)

    #得到文档之间的相似度，越大表示越相近
    return sim

def case_recommend(s1,documents_lda):
    start_time=time.time()
    #documents_lda=all_lda()
    stopwords = codecs.open('dict/stopword.txt', 'r', encoding='utf8').readlines()
    stopwords = [w.strip() for w in stopwords]
    lda = models.ldamodel.LdaModel.load('model/test_lda.model')
    dictionary=get_dict()[0]
    # 文件索引表
    id2case = json.load(open(os.path.join(project_root, 'Text_Similarity/dict/id2case.json'), 'r', encoding='utf8'))

    # 处理s1
    line = jieba.lcut(s1.strip())
    a = []
    a.extend(w for w in line if w not in stopwords)

    doc_bow = dictionary.doc2bow(a)  # 文档转换成bow，向量矩阵
    doc_lda = lda[doc_bow]  # 得到新文档的主题分布
    sims=[]

    # 得到文档之间的相似度，越大表示越相近，拿测试文档向量与所有训练集中每个案例的lda向量比较，计算相似度
    for i in documents_lda:
        sim = gensim.matutils.cossim(doc_lda, i)
        sims.append(sim)

    # 得到相似度数组
    re1 = heapq.nlargest(5, sims)  # 求最大的五个元素，并排序
    re2 = map(sims.index, heapq.nlargest(3, sims))  # 求最大的5个索引    nsmallest与nlargest相反，求最小
    re3 = list(re2)  # 因为re2由map()生成的不是list，所以需要添加list()

    for count, sim in zip(re3,re1):
        file = id2case[str(count)]
        print("相似的文档索引：{},相似距离：{}".format(file, sim))
    print('计算相似文档时间消耗为：{}秒'.format(round(time.time() - start_time, 2)))

if __name__ == '__main__':
    start_time = time.time()
    #train_model()
    documents_lda = all_lda()
    print('计算所有文档的lda消耗为：{}秒'.format(round(time.time() - start_time, 2)))

    """test="吕某某寻衅滋事一审刑事判决书\n上海市杨浦区人民法院\n刑事判决书\n（2017）沪0110刑初644号\n公诉机关上海市杨浦区人民检察院。\n被告人吕某某，男，1961年5月22日出生，汉族，户籍在上海市杨浦区。\n辩护人白新春，上海四维乐马律师事务所律师。\n上海市杨浦区人民检察院以沪杨检公诉刑诉〔2017〕627号起诉书指控被告人吕某某犯寻衅滋事罪，于2017年7月14日向本院提起公诉。本院依法组成合议庭，公开开庭审理了本案。上海市杨浦区人民检察院指派检察员夷某出庭支持公诉，被害人王1及杨浦区法律援助中心指派的诉讼代理人王小箴，被告人吕某某及杨浦区法律援助中心指派的辩护人白新春，鉴定人万某到庭参加诉讼。期间，公诉机关建议延期审理两次。现已审理终结。\n上海市杨浦区人民检察院指控，2014年11月6日８时许，被告人吕某某在本市杨浦区平凉路1298弄小区内，因琐事对残疾被害人王1进行殴打并致王受伤。其间，吕某某使用王1所拄拐杖殴打王。\n2016年10月2日，被告人吕某某接民警电话通知至公安机关投案，但未如实供述上述罪行。\n该院确认被告人吕某某随意殴打他人，情节恶劣，其行为已触犯《中华人民共和国刑法》第二百九十三条第一款第(一)项，应当以寻衅滋事罪追究刑事责任。\n被告人吕某某当庭自愿认罪，供认殴打被害人王1，但否认持拐杖殴打。其辩护人认为吕某某接电话主动至公安机关接受调查，对主要犯罪事实没有异议，自愿认罪，有自首情节；其用拐杖殴打有责任，系初犯，是由于情绪失控，法律意识淡薄，家庭也受打击，深感后悔。\n被害人王1及其诉讼代理人基本同意公诉意见，王1坚持其右侧第8、9、10、11肋骨骨折均系遭被告人吕某某殴打所致。其诉讼代理人认为王1的右侧四根肋骨骨折是吕某某殴打所致，由于王1的体质和医疗条件才未能及时发现；吕某某推诿搪塞，认罪态度不好。\n公诉人认为，持拐杖殴打是本案主要犯罪事实，被告人吕某某否认持拐杖殴打，即未如实供述主要犯罪事实，故不能认定自首。对于被害人的伤势，现无司法鉴定意见，伤情以《验伤通知书》为准。\n经审理查明，2014年11月6日８时许，被告人吕某某途经本市杨浦区平凉路1298弄小区，见王1(肢体XXX残疾)拄拐在后跟随，认为王是有意尾随而心生不满，返身质问遂起争执，继而将王1击倒，骑压在王身上继续殴打，其间，吕某某起身持王1的拐杖殴打王。当日，王1、吕某某至上海市杨浦区中心医院验伤，王1经检验，神志清，对答切题，左侧前额部2×2cm2血肿；心肺(-)，左侧8-10前肋触痛(+)；胸廓挤压(-)腹部(-)；右膝挫伤2×2cm2；左小膝稍肿，活动可。诊断为脑外伤，头皮血肿，左侧胸部软组织挫伤。右膝关节、左小腿皮肤挫伤软组织挫伤。经摄X线片显示，左侧诸肋骨走行自然，未见明显错位性骨折。胸廓对称，两膈光整，双肋膈角锐利。吕某某经检验，神清，心肺(-)，腹软，无压痛。左膝皮肤挫伤2×1cm2，右中指稍肿活动可。诊断为左膝皮肤挫伤，右中指软组织挫伤。\n2016年10月2日，被告人吕某某接民警电话通知自行至公安机关，到案后始终否认持拐殴打王1。\n认定上述事实的证据如下：\n1、被害人王1陈述，2014年11月6日8时许，其行至平凉路1298弄小区准备去眉州路近沈阳路上的芭比馒头买包子，一中年男子走在其前面，突然转身拳击其头部致其当场倒地，对方抢过其拐杖挥打其右肋部数下，然后骑在其身上，挥拳击打其头部、肋部、胸口，起身踢踩其肋部数脚。其是肢体XXX残疾，平时走路必须使用拐杖，残疾人的样子很明显。其辨认笔录证明其辨认出无故对其殴打的中年男子是吕某某。\n2、证人张新华在2014年11月8日的笔录中陈述，2014年11月6日8时20分许，其在家中看到相邻小区绰号叫“小狮子”的中年男子在与另一名身材较高的男子争吵，“小狮子”问对方为什么跟着他，还用拐杖打他，说完打对方耳光，把该男子打倒在地，随后坐在该男子身上，挥拳朝该男子头部、脸部打了七八拳，还把对方的拐杖抢过来朝对方身上打了数下，直至被周边群众拉开。“小狮子”姓吕，住在平凉路1258弄小区，身材较高的男子明显是残疾人，必须依靠拐杖行走。其辨认笔录证明其辨认出绰号“小狮子”的吕姓男子是吕某某。\n其在2017年3月17日的笔录中陈述，当时其从窗口往下看，见邻居“小狮子”与一个撑拐杖的人在争吵，吵了没几句，“小狮子”打那个撑拐杖的人耳光，那人失去重心倒在地上，“小狮子”骑在那人身上打了那人头部、脸部七八拳，后起身拿起地上的拐杖打了那人背部几下，直到被边上的人拉开。其目击了整个过程，被打的是个撑拐杖的残疾男子。\n3、证人杜某某陈述，2014年11月6日8时许，其在家门口做家务时听到平凉路1298弄处有争吵声，过去后看到邻居绰号“小狮子”的男子把一名男子按在地上，不停挥拳殴打那名男子的身体，周围邻居没有一个上去劝，其上去把“小狮子”拉开。其没有看到事情的整个过程，过去时“小狮子”已经按住那名男子进行殴打。其辨认笔录证明其辨认出绰号“小狮子”的吕姓男子是吕某某。\n4、证人王某2陈述，其是王1的母亲，当时不在场，王1告诉其当时他是去买馒头，对方走在前面，突然返身挥拳殴打他，同时用王1的拐杖打他。王1当时被打得稀里糊涂，也叫不出对方的名字，被打细节是在和其慢慢聊的过程中回想起来的。\n5、上海市公安局《验伤通知书》2份、病史等证明，王1、吕某某的伤势情况。\n6、拐杖的照片证明，经被害人王1、证人张新华分别确认，案发时吕某某用来殴打王1的拐杖即是王1平时使用的拐杖。\n7、中国残疾人联合会于2009年6月签发的《残疾人证》证明，王1为XXX残疾人，残疾XXX。\n8、上海市公安局杨浦分局大桥派出所出具的《工作情况》等证明，本案的案发情况及吕某某的到案经过。\n9、被告人吕某某的供述证明，其到案后供认为琐事与王1发生争执，为泄愤殴打王1，但否认持拐殴打。\n以上证据均经庭审举证、质证，证据之间能够相互印证的，本院予以采信。\n庭审后，被告人吕某某自愿向本院交纳赔偿款人民币2000元。\n针对主要争议焦点，本院评判如下：\n1、关于被告人吕某某不符合自首构成条件的认定\n刑法第六十七条第一款规定，犯罪以后自动投案，如实供述自己的罪行的，是自首。如实供述自己的罪行，是指犯罪嫌疑人如实交代自己的主要犯罪事实。本案中，吕某某接民警电话通知自行至公安机关可以视作自动投案，但其到案后否认根据现有证据可以证实的持拐殴打被害人这一主要犯罪事实，应认定其未如实供述自己罪行，故吕某某不符合自首的构成条件。其辩护人的辩护意见与相关法律规定不符，不予采纳。\n2、关于被告人吕某某的殴打行为与被害人王1的右肋骨折之间是否具有因果关系的认定\n根据经当庭质证的证据及鉴定人的当庭说明，现有证据尚不足以认定被告人吕某某的殴打行为与被害人王1的右侧四根肋骨骨折之间具有因果关系，故采纳公诉人认为被害人的伤情现以《验伤通知书》为准的意见，被害人及其诉讼代理人认为被害人右侧四根肋骨骨折系被告人殴打所致的意见不予采纳。\n本院认为，被告人吕某某为发泄情绪，随意殴打他人，情节恶劣，破坏社会秩序，其行为已构成寻衅滋事罪，依法应予处罚。公诉机关指控的罪名成立。吕某某当庭自愿认罪，并在庭审后自愿交纳赔偿款，可以酌情从轻处罚。吕某某的犯罪事实、情节、对社会的危害程度等具体情况均在量刑中综合考虑。为严肃国家法制，维护社会管理秩序，依照《中华人民共和国刑法》第二百九十三条第一款第(一)项之规定，判决如下：\n被告人吕某某犯寻衅滋事罪，判处有期徒刑八个月。\n(刑期从判决执行之日起计算。判决执行以前先行羁押的，羁押一日折抵刑期一日，即自2018年5月18日起，扣除先行羁押的十五日，至2019年1月2日止。)\n如不服本判决，可在接到判决书的第二日起十日内，通过本院或者直接向上海市第二中级人民法院提出上诉。书面上诉的，应当提交上诉状正本一份，副本一份。\n审判长孙颖\n人民陪审员卢晓亮\n人民陪审员刘伟福\n二〇一八年五月二十五日\n书记员吕超\n附：相关法律条文\n《中华人民共和国刑法》\n第二百九十三条有下列寻衅滋事行为之一，破坏社会秩序的，处五年以下有期徒刑、拘役或者管制：\n（一）随意殴打他人，情节恶劣的；\n……\n"
    test2="林良东聚众斗殴、故意毁坏财物、寻衅滋事二审刑事判决书\n广东省阳江市中级人民法院\n刑事判决书\n（2018）粤17刑终267号\n抗诉机关阳江市江城区人民检察院\n原审被告人林良东，男，1991年10月28日生，广东省阳江市人，汉族，初中文化，无业，住广东省阳江市海陵岛试验区。因本案于2017年10月5日被刑事拘留，同年11月8日被逮捕。现押于阳江市看守所。\n指定辩护人谭力铭，广东荣耀律师事务所律师。\n阳江市江城区人民法院审理阳江市江城区人民检察院指控原审被告人林良东犯聚众斗殴罪、故意毁坏财物罪、寻衅滋事罪一案，于2018年8月29日作出（2018）粤1702刑初111号刑事判决。宣判后，原公诉机关阳江市江城区人民检察院提出抗诉，广东省阳江市人民检察院支持抗诉。\n本院受理后，依法组成合议庭，于2018年11月12日公开开庭审理了本案，阳江市人民检察院指派检察员刘瑞波出庭履行职务，原审被告人林良东及其辩护人谭力铭到庭参加诉讼。现已审理终结。\n原判认定：一、聚众斗殴罪。\n2017年7月28日23时许，被告人林良东与黄某2、方某（两人在逃，另案处理）、“黑仔”、“军仔”、“阿某1”(身份不明）等人在阳江市闸坡镇新天地酒吧喝酒。次日凌晨2时许，林良东、黄某2、方某等人离开酒吧时，在酒吧门口与几名外省男子（身份不明）发生口角，进而引发双方持械打斗。被告人林良东等人持刀追砍外省男子至新天地酒吧的停车场被新天地的保安袁某、马某1等人阻拦。当林良东等人回到新天地酒吧门口时，一外省男子驾驶车牌号码为粤Ｑ×××××的白色本田牌思域小车撞向林良东等人。林良东、黄某2等人被撞后，持刀追赶白色思域小车。当追至新天地酒吧停车场时，有些外省男子就坐进马某1停放在停车场的车牌号码为粤Ｑ×××××的风神牌小车内，黄某2等人见此随即持刀对该车的后挡风玻璃、车尾盖等部位进行追砍，外省男子即驾车逃离现场。被告人林良东等人遂对外省男子进行追赶，后在闸坡镇海宝酒店对面路边发现外省男子驾驶的粤Ｑ×××××白色本田思域小车停放在该处，遂对该车进行打砸。\n经鉴定，粤Ｑ×××××本田牌思域小车的损失价值是16221元，粤Ｑ×××××风神牌小车的损失价值是2801元。\n上述事实，有公诉机关提交并经法庭质证、认证的下列证据予以证明：受案登记表、立案决定书；被告人林良东的供述与辩解；证人证言；鉴定意见；现场勘验检查笔录、现场照片；汽车租赁合同；接受证据材料清单；视频截图至指认、视听资料；到案经过及拘留证、逮捕证；身份证明、违法犯罪记录查询情况说明。\n二、寻衅滋事罪。\n吴某（另案处理）与被害人林某1因琐事存在纠纷。2017年8月16日凌晨1时许，吴某、方某等人在闸坡镇天羽酒吧门口碰见被害人林某1。吴某与林某1发生口角，并相互推搡。后被告人林良东纠集“肥仔”（身份不明）等人去到现场，伙同吴某、方某对林某1进行殴打。其中吴某、“肥仔”持刀将林某1砍伤。2017年10月5日，民警在新天地酒吧门口将林良东抓获。\n经鉴定，被害人林某1的损伤程度评定为轻微伤。\n被告人林良东在法庭上对公诉机关指控的上述犯罪事实及证据无异议，并承认控罪。且有公诉机关提交，经法庭质证、认证的受案登记表及立案决定书、现场勘验检查笔录、鉴定意见、证人证言、辨认笔录、被害人的陈述、被告人的供述与辩解等证据证实。\n原审认为：被告人林良东等人持械对外省人进行聚众斗殴的一个行为中，造成了外省人的车辆粤Ｑ×××××和马某1的车辆粤Ｑ×××××因被打砸而损毁，被告人等人的行为同时触犯了聚众斗殴罪和故意毁坏财物罪两个罪名，是想象竞合犯，应择一重罪处罚。被告人林良东无视国家法律，持械聚众斗毁，其行为已构成聚众斗殴罪；被告人林良东持械随意殴打他人，情节恶劣，其行为又构成寻衅滋事罪，应依照《中华人民共和国刑法》第二百九十二条第一款、第二百九十三条第一款的规定予以处罚。被告人林良东一人犯数罪，应予以数罪并罚。公诉机关指控被告人林良东犯聚众斗殴罪、寻衅滋事罪，罪名均成立；指控被告人林良东犯故意毁坏财物罪，罪名不成立。被告人林良东归案后，能如实交代自己的犯罪行为，依法可从轻处罚。被告人林良东称不构成聚众斗殴罪的辩解意见不成立，不予采纳。根据被告人的犯罪事实、情节、对社会的危害程度以及悔罪表现，依照《中华人民共和国刑法》第二百九十二条第一款第（四）项、第二百九十三条第一款第（一）项、第六十七条第三款的规定，判决：被告人林良东犯聚众斗殴罪，判处有期徒刑三年八个月；犯寻衅滋事罪，判处有期徒刑一年；合处有期徒刑四年八个月，决定执行有期徒刑四年二个月。\n阳江市江城区人民检察院抗诉提出：被告人林良东在参与聚众斗殴后，再实施故意毁坏财物的犯罪事实清楚，证据确实充分，分别构成聚众斗殴罪，故意毁坏财物罪。阳江市江城区人民法院（2018）粤1702刑初111号刑事判决确有错误，为维持司法公正，准确惩治犯罪，依照《中华人民共和国刑事诉讼法》相关的规定，提出抗诉，提请本院依法判处。\n阳江市人民检察院支持抗诉及出庭检察员的意见：原审被告人林良东与同伙在新天地酒吧门口持械与外省男子斗殴，其行为已构成聚众斗殴罪。斗殴结束后，林良东因被外省男子驾驶的粤Ｑ×××××小汽车撞倒而追赶该车辆，林良东等人追至海宝酒店，当时外省男子已逃离现场，车内空无一人，林良东等人为报复而对粤Ｑ×××××小汽车进行打砸，且后将该车开至滨海医院、北洛湾返回海宝酒店附近时还对该车进行第二次打砸。林良东等人是另起犯意，基于对粤Ｑ×××××小汽车进行故意毁坏的主观故意所实施的犯罪行为。前面的聚众斗殴行为和后面的故意毁坏财物行为，是在两个不同犯罪故意支配下实施的两个不同性质的犯罪行为，而非同一行为同时触犯两个罪名，不应认定为想象竞合犯，应对原审被告人林良东以聚众斗殴罪和故意毁坏财物罪定罪，数罪并罚。为维持司法公正，准确惩治犯罪，依照《中华人民共和国刑事诉讼法》第二百三十六条的规定，请本院依法改判。\n原审被告人林良东提出的辩解意见：对原审法院认定的事实不持异议，但认为在与外省男子斗殴的事实中，其只构成故意毁坏财物罪，不构成聚众斗殴罪；且寻衅滋事的事实中，其只是被纠集过去，只推了被害人林某1两下，不构成寻衅滋事罪。\n原审被告人林良东的辩护人提出的辩护意见：1、原审被告人林良东与外省男子的斗殴行为中，对车辆粤Ｑ×××××进行打砸，斗殴和打砸都是基于泄愤这一犯意，是一个行为同时触犯了聚众斗殴和故意毁坏财物两个罪名，是想象竞合犯。应择一重罪处罚。2、本案中，有部分犯罪行为不是原审被告人亲手实施，虽是共同犯罪，但量刑上也应有所体现。且原审被告人能如实供述其犯罪事实，有一定的悔罪表现，希望二审从轻处罚。\n经审理查明，一、聚众斗殴罪、故意毁坏财物罪。\n2017年7月28日23时许，原审被告人林良东与黄某2、方某（两人在逃，另案处理）、“黑仔”、“军仔”、“阿某1”(身份不明）等人在阳江市闸坡镇新天地酒吧喝酒。次日凌晨2时许，林良东、黄某2、方某等人离开酒吧时，在酒吧门口与几名外省男子（身份不明）发生口角，进而引发双方持械打斗。原审被告人林良东等人持刀追砍外省男子至新天地酒吧的停车场被新天地的保安袁某、马某1等人阻拦。当林良东等人回到新天地酒吧门口时，一外省男子驾驶车牌号码为粤Ｑ×××××的白色本田牌思域小车撞向林良东等人。林良东、黄某2等人被撞后，持刀追赶白色思域小车。当追至新天地酒吧停车场时，有些外省男子就坐进马某1停放在停车场的车牌号码为粤Ｑ×××××的风神牌小车内，黄某2等人见此随即持刀对该车的后挡风玻璃、车尾盖等部位进行追砍，外省男子即驾车逃离现场。原审被告人林良东等人遂对外省男子进行追赶，当林良东等人追至距离斗殴现场450米的闸坡镇海宝酒店附近，发现粤Ｑ×××××小汽车内空无一人，遂对该车进行第一次打砸。后林良东等人驾驶该车去到闸坡镇北洛湾再返回海宝酒店附近对该车进行第二次打砸。\n经鉴定，粤Ｑ×××××本田牌思域小车的损失价值是16221元，粤Ｑ×××××风神牌小车的损失价值是2801元。\n上述事实，有原公诉机关提交并经一审、二审法庭质证、认证的下列证据予以证明：\n1、受案登记表、立案决定书。\n证实：公安机关对本案于2017年8月2日立案侦查。\n2、原审被告人林良东的供述与辩解。\n2017年7月29日，我和阿屁、黑仔、军仔等人在新天地会所大厅吃饭，后和几名四川仔吵了起来。四川仔追打他们，军仔跑到一楼的便利店拿出三把菜刀，分给黑仔、宏记，去追打对方。新天地的保安拦开双方。过了一会，一名四川男子驾驶一辆白色的本田思域小车驾车将一辆女装摩托车的二个妇女撞倒在地上，又开车来撞我。我、华某被撞飞。那名四川男子驾车往海宝酒店方向开去。于是军仔开摩托车追他，过了十分钟左右（视频反映约5分钟）军仔回来告诉我，四川男子的车停在海宝酒店门口，于是我叫阿某1驾驶摩托车搭我、华某去到海宝酒店，四川男子将车停在门口，人不在。我拿砖头砸四川男的车，阿屁也拿砖头砸副驾驶室玻璃。接着我驾驶这辆小车，阿某1搭着华某到滨海医院。医生说没大碍，然后阿某1开摩托车搭着华某和阿屁去北洛湾，我开这辆小车跟在后面，我们停在在北洛湾一块空地处，过了一会我们又把车开回到海宝酒店，阿屁拿一条长棍砸车的挡风玻璃，然后我们就各自回家。\n3、证人证言\n（1）证人马某1的证言：\n2017年7月28日晚上22时，我在新天地KTV上班，一直到29日凌晨2时许，我就下班了，我同黄某3、袁某、李某和李某的女同事一起回阳江市区，当小车开到中国石化加油站那里，我发现我的手机丢在新天地那里充电，就叫袁某调转车头到新天地KTV停车场，我拿完手机下到新天地门口的时候，我就看见有十几名海陵男子跟几个四川的人在相互吵架，双方都动手，用手推搡，海陵男子人多，动手打了四川男子，并且那些海陵男子不知道从何处拿的牛某刀和菜刀去砍那些四川人，看到这种情况，我马上去拦那些海陵男子，当时我还穿了保安制服，当时车上的黄某3和袁某看见有人打架，车没有熄就马上跑上来拦架，这帮海陵男子从门口一直追到停车场。并且看到一名男子手持一支枪状物对着那些四川人的头部，我马上拉这名男子，并且叫海陵的这帮人不要闹事，海陵这帮人就散开了。我回到停车场找我的车回市区，在停车场看见一辆白色的小轿车在停车场附近，其中有四川的人上车了，这辆白色小车在新天地KTV停车场往蓝波湾的方向驶去，在新天地KTV门口撞了一名驾驶摩托车的人，然后继续向新天地KTV的方向驶去，当时在门口停了几辆摩托车，也把摩托车撞了，然后那些海陵的十几名男子拿刀追那辆小汽车，但是那辆小汽车开走了，当时还有四川的几个人还在停车场那里，就走进了我的车里面，我就叫他们不要开走我的车，但那几个四川人没有理会我，就想开车走，海陵那班人看见几个四川人上了我的车，马上跑过来砸我的车，用刀和其他工具砸，然后在车上的四川人就开我的车往闸坡隧道口方向跑了，于是我就打电话报警，那些海陵人也走了。一会警察过来我跟警察讲有人开走我的车，我就坐滴滴车回阳江市。到了凌晨3时许左右，开走我车的那几个四川人打电话给袁某，告诉他我的小汽车停放在阳江市第三人民医院，我就到市第三人民医院，发现我的小汽车停放在医院，小汽车很多部位都被砸烂了，其中驾驶位后排的门窗玻璃部分被砸了，后尾箱砸了。我被砸的车牌号是粤Ｑ×××××风神轿车。\n（民警提供一段2017年7月29日新天地门口停车场的视频给其看），经我观看，看到当晚参与打架的几个海陵仔，已经截图给我指认了。车我已经修好了，我把相关报价单和收款收据单递给公安机关。\n当时我的车被四川仔开到第三人民医院，我到第三人民医院，一个四川仔告诉我，当晚是海陵仔调戏他老婆，好像是摸了摸他老婆的屁股。\n辨认笔录：马某1辨认出方某就是当晚在场打架的男子；黄某2是当晚持枪的男子；林良东是当晚在场打架的男子。\n（2）证人王某的证言：\n2017年7月29日凌晨2时许，新天地酒吧已经关掉大厅音乐了，最后离开的一伙人是后来发生打架的那几个重庆人，因我们内保人员袁某认识那伙重庆人当中的一个，知道他们是重庆人。我们离开酒吧大厅，我和马某1、袁某三个人下到一楼拿车，坐马某1的粤Ｑ×××××轿车回阳江市，开车的是袁某，当车开到加油站对面公路边时，马某1说手机丢在新天地收银台，就叫袁某调转车头回酒吧停车场，我们在车上等，突然看见酒吧门前有人打架，我和袁某就下车向酒吧跑过去，我们去拦架看到马某1也下来了，发生打架的一伙是经常在新天地酒吧唱歌的海陵仔，而与他们打架的是最后离场的那伙重庆人。我们看见海陵仔手中都有刀，有人拿着牛某刀，其中一个男子手里还拿一把黑色手枪。我们三个人都在劝开双方人员，后来酒吧的黄总（女）出现在现场，看见她在劝海陵仔一方不要打了，双方的人几乎被分开了之后，那伙重庆仔一方的人在我们停放马某1轿车的位置，后来重庆人一方又开了一台白色本田轿车出来，开到酒吧正门口，撞到一个开摩托车的路人，之后往酒吧门前停放海陵仔摩托车那里撞上去了，撞到了好几台摩托车倒地上了，接着那几个海陵仔也拿着刀去追之前的重庆人，我们看到他们拿着刀追，我们拦不住了，另还有几个重庆人就跑上了马某1那台轿车并将车开走了，马某1在现场打电话报警。后我和马某1、袁某坐滴滴车回阳江。回阳江之后，袁某接到一个电话，是一个重庆人打给他的，告诉马某1那台车停放在第三人民医院后面，马某1就联系闸坡派出所。\n我看到海陵仔一方有人拿着牛某刀、菜刀、其中一个男子是持枪的。重庆那方的人是没有器械的，我看到他们是空手的。我不清楚双方人员的受伤情况。我来派出所的时候，听马某1说重庆人一方有个男子的眼睛弄伤了。\n辨认笔录：辨认出方某就是当晚在场打架的男子；黄某2是当晚持枪的男子；林良东是当晚在场打架的男子。\n（3）证人袁某的证言：\n我过来反映2017年7月29日凌晨2时20分在阳江市海陵岛新天地KTV门口发生打架一事，当时我在现场拦架。打架是阳江市海陵闸坡的那班人，总共有七、八名男子跟我的一个四川朋友“阿某2”及他几个朋友在闸坡新天地酒吧门口和停车场打架。海陵那班人经常会去新天地喝酒，另外我只认识一个四川老乡“阿某2”，其他人我不认识，“阿某2”这边共七人，五男二女。\n2017年7月28日晚上22时40分我开始在新天地上班，“阿某2”他们一班人也过来新天地喝酒，到29日凌晨2时许，新天地下班了，“阿某2”这班人差不多2时10分左右离开，我就下班了，和黄某3、马某1、李某的女同事一起回阳江市，当时是我开车，马某1的小车粤Ｑ×××××停放在新天地的停车场那里。当小车开了200米到中国石化加油站那里，马某1发现手机丢在新天地收银台充电，我就调头回新天地停车场，马某1下车去拿手机，我们四人没有下车，这时，“阿某2”这班人来到我车上，讲有些海陵本地男子想找他们打架，我就叫他们上我的车，但他们没有上车。我在车上看海陵那班仔总共有七、八人跟“阿某2”他们在吵架，双方都动手，用手互相推，因海陵的那些人比较多，手上拿了牛某刀去捅，我看见“阿某2”一起过来的男子被捅了胸部，另外有几个男子也是手持菜刀的，这时马某1下来，就把海陵那班人劝开来，我也劝“阿某2”那班人走。“阿某2”这班人的朋友开了一辆白色小汽车在新天地停车场往蓝波湾酒店方向驶去，在新天地门口撞了一名驾驶摩托车的人，然后继续向新天地门口方向驶去，撞了门口停的几辆摩托车，海陵的十几名男子拿刀追那辆小车，但那小车开走了，当时还有“阿某2”和他的几个朋友还在停车场，就走进马某1的车里面，想开车走，海陵那班人看见马上跑过去砸车，用刀和其他工具砸，“阿某2”那班人就开马某1的车往闸坡隧道口方向跑了，不知道谁报了警，马某1告诉警察讲有人开走了他的小汽车，就同我坐滴滴回阳江市。到了凌晨3时10分许，我们回到平冈时，“阿某2”打电话告诉我小汽车停放在第三人民医院，我们直接去到第三人民医院，发现马某1的小汽车很多部位都被砸烂了，其中驾驶位后排的门窗玻璃部分被砸碎了，副驾驶室后排的门窗玻璃部分被砸碎了，驾驶位的车门被砸凹了，后尾箱砸凹了，事情经过就是这样。\n辨认笔录：袁某辨认出方某就是当晚在场打架的男子；黄某2是当晚持枪的男子；林良东是当晚在场持刀的男子。\n（4）证人黄某1的证言：\n2017年7月29日凌晨2时许在新天地酒吧门前打架的事我知道，其中一伙人是重庆来的，另外一方是海陵本地的，我认识其中一个叫林良东的男子。当时大家准备下班离开的，见到他们打架，我们在那里拦架。重庆一方有五、六人左右，海陵仔一共有十来人，因我认得林良东，让他叫海陵仔不要打了，让马某1叫重庆仔离开。当时林良东几个海陵仔坐上摩托车准备走的，突然之间有一台白色的本田车开到酒吧门前的空地处，当时有两个女子开着摩托车经过，一个女子被那台车撞了，那台车又马上向酒吧门前冲上去，连续撞倒五六台停在酒吧门前的海陵仔的摩托车，当时林良东也在摩托车上被撞倒了。我觉得他们是故意开车撞林良东的，海陵仔这方就开摩托车去追那台车，那台车向蓝波湾方向开去了。我在新天地酒店周边调取的视频资料看见除林良东外，还看到杰仔、狗初在场，其他人员我不认识。\n行车记录仪视频，我见到重庆仔开车撞到林良东他们接着开到海宝酒店门口对面路边就下车逃跑了，过了一会林良东、一个海陵仔和一个女的来到车前，林良东拿砖砸车挡风玻璃，另外一个男子砸副驾驶室玻璃。\n辨认笔录：黄某1认出方某、林良东、黄某2就是当晚在场打架的男子。\n（5）证人关某1的证言：\n2017年7月29日在闸坡新天地酒吧门口发生的打架事件，我在现场，一方是海陵仔，我认识的只有林良东，另一方是外省人。根据公安机关提供的2017年7月29日在新天地的相关视频和一段行车记录仪，我见到林良东先踹了对方男子一脚，双方就引发打架，接着三个海陵仔拿着菜刀追打对方至停车场那边，突然对方驾驶一辆白色轿车将我和林良东撞到就逃跑了，林良东的朋友搭着我和林良东去追这辆车，后来发现这辆车停在海宝酒店门口对面路边，林良东和另外一个男子拿起砖砸车玻璃。接着林良东的朋友搭我支滨海医院，林良东开那辆小车跟在后面，到了滨海医院，我见没有什么医生在，就说还要看了，接着我们去了北洛湾。过了一会，林良东朋友搭我回家了。视频中林良他们又回到海宝酒店对面，林良东的朋友又砸挡风玻璃。林良东等人共砸了两台车，在视频中，我看到几个海陵仔持刀追砍一台停在酒吧门口的小轿车。\n辨认笔录：关某1辨认出林良东。\n（6）证人关某2的证言：\n2017年7月17日中午，我将一辆车牌号为粤Ｑ×××××的小汽车租给一名叫许岳林的重庆男子。7月29日，发现其车被砸了，且被扣留在闸坡派出所交警大队的停车场。\n（7）证人马某2的证言：\n2017年7月29日12时，我在阳光水恋便利店门口捡到疑似弹夹。\n4、鉴定意见。\n（1）阳价认定〔2017〕088号《价格认定结论书》，鉴定意见：粤Ｑ×××××小型汽车被毁坏的价值16221元。\n（2）阳价认定〔2017〕090号《价格认定结论书》，鉴定意见：粤Ｑ×××××小型汽车被毁坏的价值2801元。\n民警已将鉴定意见告知林良东。\n5、现场勘验检查笔录、现场照片。\n证实：民警对案发现场进行了勘验，制作了现场图，拍摄了现场照片。现场照片显示了粤Ｑ×××××号小车的损坏情况。\n6、汽车租赁合同等。\n证实：许岳林向江城区驾得乐汽车租赁店租得粤Ｑ×××××号小汽车。\n7、接受证据材料清单。\n证实：马某1提供了其修理粤Ｑ×××××汽车花费3500元的相关收据。\n8、视频截图至指认、视听资料。\n证实：林良东等人指认出其本人使用砖头砸四川仔开的思域车。马某1等人对视频进行了指认，指认出林良东是带头打人的海陵男子。视听资料证实双方打斗的过程。\n粤Ｑ×××××号小车行车记录仪视频：2017年7月29日2时28分40秒，该车在新天地门口撞倒林良东等人后离开，2时29分20秒到达海宝酒店门口停下，驾驶员下车离开。2时34分20秒，有人打砸该车挡风玻璃。接着有人驾驶该车先后到达滨海医院、北洛湾一块空地停处，后于3时14分许回到海宝酒店门口停下，再次打砸该车。9、到案经过及拘留证、逮捕证。\n证实：2017年10月5日，民警在新天地酒吧门口将林良东抓获。2017年10月5日，林良东被刑事拘留，11月8日被逮捕。\n10、身份证明、违法犯罪记录查询情况说明。\n二、寻衅滋事罪。\n吴某（另案处理）与被害人林某1因琐事存在纠纷。2017年8月16日凌晨1时许，吴某、方某等人在闸坡镇天羽酒吧门口碰见被害人林某1。吴某与林某1发生口角，并相互推搡。后被告人林良东纠集“肥仔”（身份不明）等人去到现场，伙同吴某、方某对林某1进行殴打。其中吴某、“肥仔”持刀将林某1砍伤。2017年10月5日，民警在新天地酒吧门口将林良东抓获。\n经鉴定，被害人林某1的损伤程度评定为轻微伤。\n原审被告人林良东在一审开庭时对原公诉机关指控的上述犯罪事实及证据无异议，并承认控罪。且有原公诉机关提交，经一审法庭质证、认证的受案登记表及立案决定书、现场勘验检查笔录、鉴定意见、证人证言、辨认笔录、被害人的陈述、原审被告人的供述与辩解等证据证实。原审被告人林良东及其辩护人在二审审理期间未提供新的证据，本院对原审判决认定的事实和证据予以确认。\n对抗诉机关的抗诉意见、阳江市人民检察院支持刑事抗诉意见、原审被告人林良东及其辩护人提出的辩解、辩护意见，综合评判如下：\n一、本案的主要争议焦点在于原审被告人林良东的的聚众斗殴行为和对粤Ｑ×××××小汽车打砸的故意毁坏财物行为是不是想象竞合犯，择一重罪处罚？想象竞合犯是指基于一个罪过，实施一个犯罪行为，同时触犯数个犯罪客体，触犯数个罪名的情况。本案中，原审被告人林良东等人在新天地酒吧门口持械与外省男子斗殴，因被一外省男子驾驶的粤Ｑ×××××小汽车撞倒，遂追赶该车辆，追至闸坡镇海宝酒店附近，发现该小汽车内空无一人，出于报复、泄愤，于是打砸该车。原审被告人林良东等人并不是在聚众斗殴的过程中造成他人财物的损失，而是在找不到斗殴对象、斗殴停止后的情况下，打砸粤Ｑ×××××小汽车，这是两个犯罪行为，不应认定为想象竞合犯。抗诉机关的抗诉意见、阳江市人民检察院支持刑事抗诉意见有理，予以采纳；原审被告人林良东的辩护人提出的辩护意见无理，不予采纳。\n二、原审被告人林良东无视国家法律，持械聚众斗毁，其行为已构成聚众斗殴罪；原审被告人林良东持械随意殴打他人，情节恶劣，其行为又构成寻衅滋事罪。原审判决认定林良东犯聚众斗殴罪、寻衅滋事罪事实清楚，证据确实充分。原审被告人林良东二审开庭时辩解其不构成聚众斗殴罪、寻衅滋事罪理据不足，不予采纳。\n本院认为，原审被告人林良东无视国家法律，持械聚众斗殴，其行为已构成聚众斗殴罪；原审被告人林良东故意毁坏他人财物，数额较大，其行为已构成故意毁坏财物罪；原审被告人林良东持械随意殴打他人，情节恶劣，其行为又构成寻衅滋事罪。原审被告人林良东一人犯数罪，依法应予以数罪并罚。原审被告人林良东归案后，能如实交代自己的犯罪行为，有悔罪表现，依法可从轻处罚。原审判决认定事实基本清楚，证据确实、充分，审判程序合法，但认定聚众斗殴罪和故意毁坏财物罪是想象竞合犯，择一重罪处罚的定性错误，原审判决认为原审被告人林良东故意毁坏财物行为是聚众斗殴罪的加重情节，导致聚众斗殴罪量刑偏重，同时不认定故意毁坏财物罪错误；原审判决遗漏引用《中华人民共和国刑法》第六十九条第一款不当，应予纠正。抗诉机关的抗诉意见、阳江市人民检察院支持刑事抗诉意见有理，予以采纳；原审被告人林良东及其辩护人提出的辩解、辩护意见无理，不予采纳。经本院审判委员会讨论决定，依照《中华人民共和国刑法》第二百九十二条第一款第（四）项、第二百七十五条、第二百九十三条第一款第（一）项、第六十七条第三款、第六十九条第一款、《中华人民共和国刑事诉讼法》第二百三十六条第一款第（一）、（二）项，判决如下：\n一、维持阳江市江城区人民法院（2018）粤1702刑初111号刑事判决对原审被告人林良东犯聚众斗殴罪的定罪部分和寻衅滋事罪的定罪部分和量刑部分。\n二、撤销阳江市江城区人民法院（2018）粤1702刑初111号刑事判决对原审被告人林良东犯聚众斗殴罪的量刑部分；\n三、原审被告人林良东犯聚众斗殴罪，判处有期徒刑三年；犯故意毁坏财物罪，判处有期徒刑八个月；犯寻衅滋事罪，判处有期徒刑一年；合处有期徒刑四年八个月，决定执行有期徒刑四年二个月。\n（刑期从判决执行之日起计算，判决执行以前先行羁押的，羁押一日折抵刑期一日，即自2017年10月5日起至2021年12月4日止）\n本判决为终审判决\n审判长柯绍和\n审判员莫介云\n审判员陈仲贯\n二〇一八年十一月十九日\n法官助理佘世韬\n书记员陈贵连\n附相关法律条文：\n《中华人民共和国刑法》\n第二百九十二条聚众斗殴的，对首要分子和其他积极参加的，处三年以下有期徒刑、拘役或者管制；有下列情形之一的，对首要分子和其他积极参加的，处三年以上十年以下有期徒刑：\n（一）多次聚众斗殴的；\n（二）聚众斗殴人数多，规模大，社会影响恶劣的；\n（三）在公共场所或者交通要道聚众斗殴，造成社会秩序严重混乱的；\n（四）持械聚众斗殴的。\n聚众斗殴，致人重伤、死亡的，依照本法第二百三十四条、第二百三十二条的规定定罪处罚。\n第二百七十五条故意毁坏公私财物，数额较大或者有其他严重情节的，处三年以下有期徒刑、拘役或者罚金；数额巨大或者有其他特别严重情节的，处三年以上七年以下有期徒刑。\n第二百九十三条有下列寻衅滋事行为之一，破坏社会秩序的，处五年以下有期徒刑、拘役或者管制：\n（一）随意殴打他人，情节恶劣的；\n（二）追逐、拦截、辱骂、恐吓他人，情节恶劣的；\n（三）强拿硬要或者任意损毁、占用公私财物，情节严重的；\n（四）在公共场所起哄闹事，造成公共场所秩序严重混乱的。\n纠集他人多次实施前款行为，严重破坏社会秩序的，处五年以上十年以下有期徒刑，可以并处罚金。\n第六十七条犯罪以后自动投案，如实供述自己的罪行的，是自首。对于自首的犯罪分子，可以从轻或者减轻处罚。其中，犯罪较轻的，可以免除处罚。\n被采取强制措施的犯罪嫌疑人、被告人和正在服刑的罪犯，如实供述司法机关还未掌握的本人其他罪行的，以自首论。\n犯罪嫌疑人虽不具有前两款规定的自首情节，但是如实供述自己罪行的，可以从轻处罚；因其如实供述自己罪行，避免特别严重后果发生的，可以减轻处罚。\n第六十九条判决宣告以前一人犯数罪的，除判处死刑和无期徒刑的以外，应当在总和刑期以下、数刑中最高刑期以上，酌情决定执行的刑期，但是管制最高不能超过三年，拘役最高不能超过一年，有期徒刑总和刑期不满三十五年的，最高不能超过二十年，总和刑期在三十五年以上的，最高不能超过二十五年。\n数罪中有判处有期徒刑和拘役的，执行有期徒刑。数罪中有判处有期徒刑和管制，或者拘役和管制的，有期徒刑、拘役执行完毕后，管制仍须执行。\n数罪中有判处附加刑的，附加刑仍须执行，其中附加刑种类相同的，合并执行，种类不同的，分别执行。\n《中华人民共和国刑事诉讼法》\n第二百三十六条第二审人民法院对不服第一审判决的上诉、抗诉案件，经过审理后，应当按照下列情形分别处理：\n（一）原判决认定事实和适用法律正确、量刑适当的，应当裁定驳回上诉或者抗诉，维持原判；\n（二）原判决认定事实没有错误，但适用法律有错误，或者量刑不当的，应当改判；\n（三）原判决事实不清楚或者证据不足的，可以在查清事实后改判；也可以裁定撤销原判，发回原审人民法院重新审判。\n原审人民法院对于依照前款第三项规定发回重新审判的案件作出判决后，被告人提出上诉或者人民检察院提出抗诉的，第二审人民法院应当依法作出判决或者裁定，不得再发回原审人民法院重新审判。\n"
    test3 = "唐洋、肖宏海走私、贩卖、运输、制造毒品一审刑事判决书广东省佛山市南海区人民法院刑事判决书(2018)粤0605刑初3157号公诉机关佛山市南海区人民检察院。被告人唐洋，男，1991年4月21日出生于湖北省来凤县，土家族，初中文化，无业，住来凤县，因本案于2018年5月19日被羁押，同日被刑事拘留，同年6月15日被逮捕，现押于佛山市南海区看守所。指定辩护人罗志辉，广东邦南律师事务所律师。被告人肖宏海，男，1972年8月17日出生于湖北省来凤县，土家族，小学文化，无业，住来凤县，2000年7月因犯抢夺罪被判处有期徒刑七个月；2008年2月26日因犯贩卖毒品罪被江苏省苏州市虎丘区人民法院判处有期徒刑八年，并处罚金人民币八千元，2012年12月18日刑满释放。因本案于2018年5月19日被羁押，同日被刑事拘留，同年6月15日被逮捕，现押于佛山市南海区看守所。指定辩护人魏莉莉，广东群豪律师事务所律师。佛山市南海区人民检察院以佛南检公诉刑诉[2018]3099号起诉书指控被告人唐洋、肖宏海犯运输毒品罪，于2018年8月29日向本院提起公诉。本院依法组成合议庭，适用简易程序，公开开庭进行了审理。在审理过程中，两被告人否认指控的事实，本院依法适用普通程序进行审理，佛山市南海区人民检察院指派检察员陆婉瑶出庭支持公诉，二被告人及其辩护人到庭参加了诉讼。本案现已审理终结。公诉机关指控，2018年5月19日凌晨1时许，被告人唐洋、肖宏海从广州乘坐摩托车到达佛山市南海区大沥镇黄某冲市场附近，联系一名外国男子（另案处理）到该外国男子出租屋购买毒品。被告人唐洋、肖宏海分别以人民币3300元、9000元的价格向该外国男子购买毒品海洛因。当天12时许，唐洋、肖宏海携带所购买的海洛因乘坐一辆大巴汽车返回湖北省恩施，汽车行驶至佛山市南海区海八路路段时，被民警当场查获，后民警于某身上以及肖宏海行李箱搜获两人所购毒品。经称量，民警从唐洋身上搜获的白色粉末状疑似毒品海洛因净重39.6克，黄色固体海洛因可疑物净重12.7克；从肖宏海行李箱搜获白色粉状海洛因可疑物净重80.4克，黄色固体海洛因疑似物净重30.6克。经鉴定，从唐洋、肖宏海处搜获的白色粉末状海洛因可疑物检出烟酰胺成分，黄色固体海洛因可疑物检出海洛因成分。就上述指控，公诉机关向法庭出示了二被告人的供述、抓获经过证明、称量和取样笔录；扣押物品清单；赃物和作案工具的照片；现场勘查记录、尿液（毒品）检测报告、视频资料、户籍和前科材料等证据予以证实，并认为被告人唐洋、肖宏海运输毒品海洛因十克以上不满五十克，其行为已触犯《中华人民共和国刑法》第三百四十七条第三款之规定，应当以运输毒品罪追究其刑事责任。被告人肖宏海因犯贩卖毒品罪被判过刑，又犯运输毒品罪，应当依照《中华人民共和国刑法》第三百五十六条之规定处罚。被告人唐洋、肖宏海对公诉机关指控的事实均无异议，但辩称自己的行为属于非法持有毒品罪。二辩护人均辩称,被告人唐洋、肖宏海有吸毒史，所购买的毒品是为了自己吸食，因此，其行为属于非法持有毒品罪。被告人唐洋、肖宏海自愿认罪，有悔罪表现，建议给予从轻、减轻处罚。经审理查明，公诉机关指控被告人唐洋、肖宏海运输毒品的事实清楚，证据确实、充分，本院予以确认。本院认为，被告人唐洋、肖宏海运输毒品海洛因十克以上不满五十克，其行为已构成运输毒品罪。公诉机关所控罪名成立。二被告人携带毒品乘坐大巴车，企图将毒品运送至湖北老家，后在运输途中被民警查获，海洛因的数量已达到刑法中数量较大的标准，因此，二被告人的行为依法应认定为运输毒品罪，且属犯罪既遂。二被告人的辩解及二辩护人的辩护意见缺乏理据，本院不予采纳。肖宏海曾因犯贩卖毒品罪被判过刑，现又犯运输毒品罪，属于毒品再犯，依法从重处罚。二被告人归案后能如实供述自己的罪行，依法从轻处罚。采纳二辩护人的相关辩护意见。依照《中华人民共和国刑法》第三百四十七条第三款、第三百五十六条、第六十七条第三款、第五十二条、第五十三条、第六十四条之规定，判决如下：一、被告人唐洋犯运输毒品罪，判处有期徒刑七年，并处罚金人民币七万元。（刑期从判决执行之日起计算。判决执行以前先行羁押的，羁押一日折抵刑期一日，即自2018年5月19日起至2025年5月18日止。罚金从本判决发生法律效力之日起30日内缴纳）。二、被告人肖宏海犯运输毒品罪，判处有期徒刑八年，并处罚金人民币八万元。（刑期从判决执行之日起计算。判决执行以前先行羁押的，羁押一日折抵刑期一日，即自2018年5月19日起至2026年5月18日止。罚金从本判决发生法律效力之日起30日内缴纳）。三、缴获的毒品（未随案移送），由公安机关予以没收、销毁。如不服本判决，可在接到判决书的第二日起十日内，通过本院或者直接向广东省佛山市中级人民法院提出上诉。书面上诉的，应当提交上诉状正本一份，副本二份。审判长陈伟忠人民陪审员何凤微人民陪审员童霭婷二〇一八年十月二十四日法官助理李冰冰书记员黄诗美法律条文《中华人民共和国刑法》第三百四十七条走私、贩卖、运输、制造毒品，无论数量多少，都应当追究刑事责任，予以刑事处罚。走私、贩卖、运输、制造毒品，有下列情形之一的，处十五年有期徒刑、无期徒刑或者死刑，并处没收财产：（一）走私、贩卖、运输、制造鸦片一千克以上、海洛因或者甲基苯丙胺五十克以上或者其他毒品数量大的；（二）走私、贩卖、运输、制造毒品集团的首要分子；（三）武装掩护走私、贩卖、运输、制造毒品的；（四）以暴力抗拒检查、拘留、逮捕，情节严重的；（五）参与有组织的国际贩毒活动的。走私、贩卖、运输、制造鸦片二百克以上不满一千克、海洛因或者甲基苯丙胺十克以上不满五十克或者其他毒品数量较大的，处七年以上有期徒刑，并处罚金。走私、贩卖、运输、制造鸦片不满二百克、海洛因或者甲基苯丙胺不满十克或者其他少量毒品的，处三年以下有期徒刑、拘役或者管制，并处罚金；情节严重的，处三年以上七年以下有期徒刑，并处罚金。单位犯第二款、第三款、第四款罪的，对单位判处罚金，并对其直接负责的主管人员和其他直接责任人员，依照各该款的规定处罚。利用、教唆未成年人走私、贩卖、运输、制造毒品，或者向未成年人出售毒品的，从重处罚。对多次走私、贩卖、运输、制造毒品，未经处理的，毒品数量累计计算。"

    #new_case_recommend(test)
    print(lda_sim(test,test))"""
    # 测试数据集
    test_set = json.load(
        open(os.path.join(project_root, 'Text_Similarity/test_data/test_data.json'), 'r', encoding='utf8'))

    for i in dict(test_set).values():
        case_recommend(i,documents_lda)











