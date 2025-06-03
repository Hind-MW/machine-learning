@[TOC]
# 电商产品评论数据情感分析


针对用户在电商平台上留下的评论数据，对其进行分词、词性标注和去除停用词等文本预处理。基于预处理后的数据进行情感分析，并使用LDA主题模型提取评论关键信息，以了解用户的需求、意见、购买原因及产品的优缺点等，最终提出改善产品的建议

----

## 数据预处理

### 评论去重

一些电商平台为了避免一些客户长时间不进行评论，往往会设置一道程序，如果用户超过规定的时间仍然没有做出评论，系统就会自动替客户做出评论，这类数据显然没有任何分析价值。由语言的特点可知，在大多数情况下，不同购买者之间的有价值的评论是不会出现完全重复的，如果不同购物者的评论完全重复，那么这些评论一般都是毫无意义的。为了存留更多的有用语料，本节针对完全重复的语料下手，仅删除完全重复部分，以确保保留有用的文本评论信息。

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import jieba.posseg as psg


import warnings
warnings.filterwarnings("ignore")

%matplotlib inline

path = '/home/kesci/input/emotion_analysi7147'
reviews = pd.read_csv(path+'/reviews.csv')
print(reviews.shape)
reviews.head()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/157e78837367488dbcd5b612f2839d11.png)

```python
# 删除数据记录中所有列值相同的记录
reviews = reviews[['content','content_type']].drop_duplicates()
content = reviews['content']
reviews.shape
reviews
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d5be5591dc8546c2ae52d8a11ab1ab8a.png)
### 数据清洗

通过人工观察数据发现，评论中夹杂着许多数字与字母，对于本案例的挖掘目标而言，这类数据本身并没有实质性帮助。另外，由于该评论文本数据主要是围绕京东商城中美的电热水器进行评价的，其中“京东”“京东商城”“美的”“热水器”“电热水器”等词出现的频数很大，但是对分析目标并没有什么作用，因此可以在分词之前将这些词去除，对数据进行清洗

```python
# 去除英文、数字、京东、美的、电热水器等词语
strinfo = re.compile('[0-9a-zA-Z]|京东|美的|电热水器|热水器|')
content = content.apply(lambda x: strinfo.sub('',x))
```
### 分词、词性标注、去除停用词

词是文本信息处理的基础环节，是将一个单词序列切分成单个单词的过程。准确地分词可以极大地提高计算机对文本信息的识别和理解能力。相反，不准确的分词将会产生大量的噪声，严重干扰计算机的识别理解能力，并对这些信息的后续处理工作产生较大的影响。中文分词的任务就是把中文的序列切分成有意义的词，即添加合适的词串使得所形成的词串反映句子的本意，中文分词的关键问题为切分歧义的消解和未登录词的识别。

未登录词是指词典中没有登录过的人名、地名、机构名、译名及新词语等。当采用匹配的办法来切分词语时，由于词典中没有登录这些词，会引起自动切分词语的困难。

分词最常用的工作包是jieba分词包，jieba分词是Python写成的一个分词开源库，专门用于中文分词，其有3条基本原理，即实现所采用技术。
1. 基于Trie树结构实现高效的词图扫描，生成句子中汉字所有可能成词情况所构成的有向无环图（DAG）。
2. 采用动态规划查找最大概率路径，找出基于词频的最大切分组合。
3. 对于未登录词，采用HMM模型，使用了Viterbi算法，将中文词汇按照BEMS 4个状态来标记。

```python
# 分词
worker = lambda s: [(x.word, x.flag) for x in psg.cut(s)] # 自定义简单分词函数
seg_word = content.apply(worker)
seg_word.head()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2b52e6b9eac84168a8d7f6d4b4ed8b6f.png)

```python
# 将词语转为数据框形式，一列是词，一列是词语所在的句子ID，最后一列是词语在该句子的位置
n_word = seg_word.apply(lambda x: len(x))  # 每一评论中词的个数

n_content = [[x+1]*y for x,y in zip(list(seg_word.index), list(n_word))]

# 将嵌套的列表展开，作为词所在评论的id
index_content = sum(n_content, [])

seg_word = sum(seg_word, [])
# 词
word = [x[0] for x in seg_word]
# 词性
nature = [x[1] for x in seg_word]

content_type = [[x]*y for x,y in zip(list(reviews['content_type']), list(n_word))]
# 评论类型
content_type = sum(content_type, [])

result = pd.DataFrame({"index_content":index_content, 
                       "word":word,
                       "nature":nature,
                       "content_type":content_type})
result.head()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1b23fb7852ad4a93b151e7db6a747e1b.png)

```python
# 构造各词在对应评论的位置列
n_word = list(result.groupby(by = ['index_content'])['index_content'].count())
index_word = [list(np.arange(0, y)) for y in n_word]
# 词语在该评论的位置
index_word = sum(index_word, [])
# 合并评论id
result['index_word'] = index_word

result.head()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/41c4be5bb00c4e4c9890ae3ef3eb1086.png)
### 提取含名词的评论

由于本案例的目标是对产品特征的优缺点进行分析，类似“不错，很好的产品”“很不错，继续支持”等评论虽然表达了对产品的情感倾向，但是实际上无法根据这些评论提取出哪些产品特征是用户满意的。评论中只有出现明确的名词，如机构团体及其他专有名词时，才有意义，因此需要对分词后的词语进行词性标注。之后再根据词性将含有名词类的评论提取出来。

```python
# 提取含有名词类的评论,即词性含有“n”的评论
ind = result[['n' in x for x in result['nature']]]['index_content'].unique()
result = result[[x in ind for x in result['index_content']]]
result.head()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/be69b52b2aa54776be1edaf52279e688.png)
### 绘制词云

绘制词云查看分词效果，词云会将文本中出现频率较高的“关键词”予以视觉上的突出。首先需要对词语进行词频统计，将词频按照降序排序，选择前100个词，使用wordcloud模块中的WordCloud绘制词云，查看分词效果

```python
import matplotlib.pyplot as plt
from wordcloud import WordCloud

frequencies = result.groupby('word')['word'].count()
frequencies = frequencies.sort_values(ascending = False)
backgroud_Image=plt.imread(path+'/pl.jpg')

# 自己上传中文字体到kesci
font_path = '/home/kesci/work/data/fonts/MSYHL.TTC'
wordcloud = WordCloud(font_path=font_path, # 设置字体，不设置就会出现乱码
                      max_words=100,
                      background_color='white',
                      mask=backgroud_Image)# 词云形状

my_wordcloud = wordcloud.fit_words(frequencies)
plt.imshow(my_wordcloud)
plt.axis('off') 
plt.show()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9799e39b9cf2474f936f03461c914577.png)
由图可以看出，对评论数据进行预处理后，分词效果较为符合预期。其中“安装”“师傅”“售后”“物流”“服务”等词出现频率较高，因此可以初步判断用户对产品的这几个方面比较重视

```python
# 将结果保存
result.to_csv("./word.csv", index = False, encoding = 'utf-8')
```
## 词典匹配

### 评论数据情感倾向分析

匹配情感词情感倾向也称为情感极性。在某商品评论中，可以理解为用户对该商品表达自身观点所持的态度是支持、反对还是中立，即通常所指的正面情感、负面情感、中性情感。由于本案例主要是对产品的优缺点进行分析，因此只要确定用户评论信息中的情感倾向方向分析即可，不需要分析每一评论的情感程度。

对评论情感倾向进行分析首先要对情感词进行匹配，主要采用词典匹配的方法，本案例使用的情感词表是2007年10月22日知网发布的“情感分析用词语集（beta版）”，主要使用“中文正面评价”词表、“中文负面评价”“中文正面情感”“中文负面情感”词表等。将“中文正面评价”“中文正面情感”两个词表合并，并给每个词语赋予初始权重1，作为本案例的正面评论情感词表。将“中文负面评价”“中文负面情感”两个词表合并，并给每个词语赋予初始权重-1，作为本案例的负面评论情感词表。

一般基于词表的情感分析方法，分析的效果往往与情感词表内的词语有较强的相关性，如果情感词表内的词语足够全面，并且词语符合该案例场景下所表达的情感，那么情感分析的效果会更好。针对本案例场景，需要在知网提供的词表基础上进行优化，例如“好评”“超值”“差评”“五分”等词只有在网络购物评论上出现，就可以根据词语的情感倾向添加至对应的情感词表内。将“满意”“好评”“很快”“还好”“还行”“超值”“给力”“支持”“超好”“感谢”“太棒了”“厉害”“挺舒服”“辛苦”“完美”“喜欢”“值得”“省心”等词添加进正面情感词表。将“差评”“贵”“高”“漏水”等词加入负面情感词表。读入正负面评论情感词表，正面词语赋予初始权重1，负面词语赋予初始权重-1。

```python
word = pd.read_csv("./word.csv")

# 读入正面、负面情感评价词
pos_comment = pd.read_csv(path+"/正面评价词语（中文）.txt", header=None,sep="\n", 
                          encoding = 'utf-8', engine='python')
neg_comment = pd.read_csv(path+"/负面评价词语（中文）.txt", header=None,sep="\n", 
                          encoding = 'utf-8', engine='python')
pos_emotion = pd.read_csv(path+"/正面情感词语（中文）.txt", header=None,sep="\n", 
                          encoding = 'utf-8', engine='python')
neg_emotion = pd.read_csv(path+"/负面情感词语（中文）.txt", header=None,sep="\n", 
                          encoding = 'utf-8', engine='python') 

# 合并情感词与评价词
positive = set(pos_comment.iloc[:,0])|set(pos_emotion.iloc[:,0])
negative = set(neg_comment.iloc[:,0])|set(neg_emotion.iloc[:,0])

# 正负面情感词表中相同的词语
intersection = positive&negative

positive = list(positive - intersection)
negative = list(negative - intersection)

positive = pd.DataFrame({"word":positive,
                         "weight":[1]*len(positive)})
negative = pd.DataFrame({"word":negative,
                         "weight":[-1]*len(negative)}) 

posneg = positive.append(negative)


# 将分词结果与正负面情感词表合并，定位情感词
data_posneg = posneg.merge(word, left_on = 'word', right_on = 'word', 
                           how = 'right')
data_posneg = data_posneg.sort_values(by = ['index_content','index_word'])

data_posneg.head()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2981fa60cfd84283bded8115a9527c8d.png)
### 修正情感倾向

情感倾向修正主要根据情感词前面两个位置的词语是否存在否定词而去判断情感值的正确与否，由于汉语中存在多重否定现象，即当否定词出现奇数次时，表示否定意思；当否定词出现偶数次时，表示肯定意思。按照汉语习惯，搜索每个情感词前两个词语，若出现奇数否定词，则调整为相反的情感极性。

本案例使用的否定词表共有19个否定词，分别为：不、没、无、非、莫、弗、毋、未、否、别、無、休、不是、不能、不可、没有、不用、不要、从没、不太。

读入否定词表，对情感值的方向进行修正。计算每条评论的情感得分，将评论分为正面评论和负面评论，并计算情感分析的准确率。

```python
# 载入否定词表
notdict = pd.read_csv(path+"/not.csv")

# 构造新列，作为经过否定词修正后的情感值
data_posneg['amend_weight'] = data_posneg['weight']
data_posneg['id'] = np.arange(0, len(data_posneg))

# 只保留有情感值的词语
only_inclination = data_posneg.dropna().reset_index(drop=True)

index = only_inclination['id']


for i in np.arange(0, len(only_inclination)):
    # 提取第i个情感词所在的评论
    review = data_posneg[data_posneg['index_content'] == only_inclination['index_content'][i]]
    review.index = np.arange(0, len(review))
    # 第i个情感值在该文档的位置
    affective = only_inclination['index_word'][i]
    if affective == 1:
        ne = sum([i in notdict['term'] for i in review['word'][affective - 1]])%2
        if ne == 1:
            data_posneg['amend_weight'][index[i]] = -data_posneg['weight'][index[i]]          
    elif affective > 1:
        ne = sum([i in notdict['term'] for i in review['word'][[affective - 1, 
                  affective - 2]]])%2
        if ne == 1:
            data_posneg['amend_weight'][index[i]] = -data_posneg['weight'][index[i]]
            

            
# 更新只保留情感值的数据
only_inclination = only_inclination.dropna()

# 计算每条评论的情感值
emotional_value = only_inclination.groupby(['index_content'],
                                           as_index=False)['amend_weight'].sum()

# 去除情感值为0的评论
emotional_value = emotional_value[emotional_value['amend_weight'] != 0]
```
### 查看情感分析效果

```python
# 给情感值大于0的赋予评论类型（content_type）为pos,小于0的为neg
emotional_value['a_type'] = ''
emotional_value['a_type'][emotional_value['amend_weight'] > 0] = 'pos'
emotional_value['a_type'][emotional_value['amend_weight'] < 0] = 'neg'

emotional_value.head()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8d92f986b3e8414bbf7b11ea3ae1c15a.png)

```python
# 查看情感分析结果
result = emotional_value.merge(word, 
                               left_on = 'index_content', 
                               right_on = 'index_content',
                               how = 'left')
result.head()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f7d0f216a038475d9f3a46c2b8a4e0c3.png)

```python
result = result[['index_content','content_type', 'a_type']].drop_duplicates()
result.head()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/006cc2c9f789488c9051c57e4ecc2dde.png)
假定用户在评论时不存在“选了好评的标签，而写了差评内容”的情况，比较原评论的评论类型与情感分析得出的评论类型，绘制情感倾向分析混淆矩阵，查看词表的情感分析的准确率。

```python
# 交叉表:统计分组频率的特殊透视表
confusion_matrix = pd.crosstab(result['content_type'], result['a_type'], 
                               margins=True)
confusion_matrix.head()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a579da1537b1404583854971c4e40941.png)

```python
(confusion_matrix.iat[0,0] + confusion_matrix.iat[1,1])/confusion_matrix.iat[2,2]
```

```python
# 提取正负面评论信息
ind_pos = list(emotional_value[emotional_value['a_type'] == 'pos']['index_content'])
ind_neg = list(emotional_value[emotional_value['a_type'] == 'neg']['index_content'])
posdata = word[[i in ind_pos for i in word['index_content']]]
negdata = word[[i in ind_neg for i in word['index_content']]]
```

```python
# 绘制词云
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# 正面情感词词云
freq_pos = posdata.groupby('word')['word'].count()
freq_pos = freq_pos.sort_values(ascending = False)
backgroud_Image=plt.imread(path+'/pl.jpg')
wordcloud = WordCloud(font_path=font_path,
                      max_words=100,
                      background_color='white',
                      mask=backgroud_Image)
pos_wordcloud = wordcloud.fit_words(freq_pos)
plt.imshow(pos_wordcloud)
plt.axis('off') 
plt.show()


# 负面情感词词云
freq_neg = negdata.groupby(by = ['word'])['word'].count()
freq_neg = freq_neg.sort_values(ascending = False)
neg_wordcloud = wordcloud.fit_words(freq_neg)
plt.imshow(neg_wordcloud)
plt.axis('off') 
plt.show()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2b46c7d118164edf859567bb0f6e80fa.png)

```python
# 将结果写出,每条评论作为一行
posdata.to_csv("./posdata.csv", index = False, encoding = 'utf-8')
negdata.to_csv("./negdata.csv", index = False, encoding = 'utf-8')
```
由图正面情感评论词云可知，“不错”“满意”“好评”等正面情感词出现的频数较高，并且没有掺杂负面情感词语，可以看出情感分析能较好地将正面情感评论抽取出来。

由图负面情感评论词云可知，“差评”“垃圾”“不好”“太差”等负面情感词出现的频数较高，并且没有掺杂正面情感词语，可以看出情感分析能较好地将负面情感评论抽取出来。

____

## LinearSVC模型预测情感
将数据集划分为训练集和测试集(8:2)，通过TfidfVectorizer将评论文本向量化，在来训练LinearSVC模型，查看模型在训练集上的得分，预测测试集

```python
reviews.head()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/01cff654798240b8a2bdf6e4562fd899.png)

```python
reviews['content_type'] = reviews['content_type'].map(lambda x:1.0 if x == 'pos' else 0.0)
reviews.head()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/727e615744334815829ecf6e385a916b.png)

```python
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF  # 原始文本转化为tf-idf的特征矩阵
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

# 将有标签的数据集划分成训练集和测试集
train_X,valid_X,train_y,valid_y = train_test_split(reviews['content'],reviews['content_type'],test_size=0.2,random_state=42)

train_X.shape,train_y.shape,valid_X.shape,valid_y.shape
```

```python
# 模型构建
model_tfidf = TFIDF(min_df=5, max_features=5000, ngram_range=(1,3), use_idf=1, smooth_idf=1)
# 学习idf vector
model_tfidf.fit(train_X)
# 把文档转换成 X矩阵（该文档中该特征词出现的频次），行是文档个数，列是特征词的个数
train_vec = model_tfidf.transform(train_X)
train_vec.toarray()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5263caaccf3b40af88d32bb675abdff4.png)

```python
# 模型训练
model_SVC = LinearSVC()
clf = CalibratedClassifierCV(model_SVC)
clf.fit(train_vec,train_y)
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6524131ba31c412da17a461fa9013559.png)

```python
# 把文档转换成矩阵
valid_vec = model_tfidf.transform(valid_X)
# 验证
pre_valid = clf.predict_proba(valid_vec)
pre_valid[:5]
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1a0c7f5383d344a2a9c018edf4e76677.png)

```python
pre_valid = clf.predict(valid_vec)
print('正例:',sum(pre_valid == 1))
print('负例:',sum(pre_valid == 0))
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/174e92691a234cc0b7dd8d6d4f2c552d.png)

```python
from sklearn.metrics import accuracy_score

score = accuracy_score(pre_valid,valid_y)
print("准确率:",score)
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b76afba4153f4d8da9c67a348b313003.png)

## LDA模型

LDA是一种文档主题生成模型，包含词、主题和文档三层结构。

1. 主题模型在自然语言处理等领域是用来在一系列文档中发现抽象主题的一种统计模型。判断两个文档相似性的传统方法是通过查看两个文档共同出现的单词的多少，如TF（词频）、TF-IDF（词频—逆向文档频率）等，这种方法没有考虑文字背后的语义关联，例如，两个文档共同出现的单词很少甚至没有，但两个文档是相似的，因此在判断文档相似性时，需要使用主题模型进行语义分析并判断文档相似性。如果一篇文档有多个主题，则一些特定的可代表不同主题的词语就会反复出现，此时，运用主题模型，能够发现文本中使用词语的规律，并且把规律相似的文本联系到一起，以寻求非结构化的文本集中的有用信息。例如，在美的电热水器的商品评论文本数据中，代表电热水器特征的词语如“安装”“出水量”“服务”等会频繁地出现在评论中，运用主题模型，把热水器代表性特征相关的情感描述性词语与对应特征的词语联系起来，从而深入了解用户对电热水器的关注点及用户对于某一特征的情感倾向



2. LDA主题模型潜在狄利克雷分配，即LDA模型（Latent Dirichlet Allocation，LDA）是由Blei等人在2003年提出的生成式主题模型。所谓生成模型，就是说，我们认为一篇文章的每个词都是通过“以一定概率选择了某个主题，并从这个主题中以一定概率选择某个词语”这样一个过程得到。文档到主题服从多项式分布，主题到词服从多项式分布。LDA模型也被称为3层贝叶斯概率模型，包含文档（d）、主题（z）、词（w）3层结构，能够有效对文本进行建模，和传统的空间向量模型（VSM）相比，增加了概率的信息。通过LDA主题模型，能够挖掘数据集中的潜在主题，进而分析数据集的集中关注点及其相关特征词。LDA模型采用词袋模型（Bag of Words，BOW）将每一篇文档视为一个词频向量，从而将文本信息转化为易于建模的数字信息。定义词表大小为L，一个L维向量（1，0，0，…，0，0）表示一个词。由N个词构成的评论记为d=（w1，w2，…，wN）。假设某一商品的评论集D由M篇评论构成，记为D=（d1，d2，…，dM）。M篇评论分布着K个主题，记为Zi=（i=1，2，…，K）。记a和b为狄利克雷函数的先验参数，q为主题在文档中的多项分布的参数，其服从超参数为a的Dirichlet先验分布，f为词在主题中的多项分布的参数，其服从超参数b的Dirichlet先验分布。

```python
import re
import itertools

from gensim import corpora, models


# 载入情感分析后的数据
posdata = pd.read_csv("./posdata.csv", encoding = 'utf-8')
negdata = pd.read_csv("./negdata.csv", encoding = 'utf-8')


# 建立词典
pos_dict = corpora.Dictionary([[i] for i in posdata['word']])  # 正面
neg_dict = corpora.Dictionary([[i] for i in negdata['word']])  # 负面

# 建立语料库
pos_corpus = [pos_dict.doc2bow(j) for j in [[i] for i in posdata['word']]]  # 正面
neg_corpus = [neg_dict.doc2bow(j) for j in [[i] for i in negdata['word']]]   # 负面
```
### 主题数寻优

基于相似度的自适应最优LDA模型选择方法，确定主题数并进行主题分析。实验证明该方法可以在不需要人工调试主题数目的情况下，用相对少的迭代找到最优的主题结构。

具体步骤如下：
1. 取初始主题数k值，得到初始模型，计算各主题之间的相似度（平均余弦距离）。
2. 增加或减少k值，重新训练模型，再次计算各主题之间的相似度。
3. 重复步骤2直到得到最优k值。

```python
# 余弦相似度函数
def cos(vector1, vector2):
    dot_product = 0.0;  
    normA = 0.0;  
    normB = 0.0;  
    for a,b in zip(vector1, vector2): 
        dot_product += a*b  
        normA += a**2  
        normB += b**2  
    if normA == 0.0 or normB==0.0:  
        return(None)  
    else:  
        return(dot_product / ((normA*normB)**0.5))   

# 主题数寻优
def lda_k(x_corpus, x_dict):  
    
    # 初始化平均余弦相似度
    mean_similarity = []
    mean_similarity.append(1)
    
    # 循环生成主题并计算主题间相似度
    for i in np.arange(2,11):
        # LDA模型训练
        lda = models.LdaModel(x_corpus, num_topics = i, id2word = x_dict)
        for j in np.arange(i):
            term = lda.show_topics(num_words = 50)
            
        # 提取各主题词
        top_word = []
        for k in np.arange(i):
            top_word.append([''.join(re.findall('"(.*)"',i)) \
                             for i in term[k][1].split('+')])  # 列出所有词
           
        # 构造词频向量
        word = sum(top_word,[])  # 列出所有的词   
        unique_word = set(word)  # 去除重复的词
        
        # 构造主题词列表，行表示主题号，列表示各主题词
        mat = []
        for j in np.arange(i):
            top_w = top_word[j]
            mat.append(tuple([top_w.count(k) for k in unique_word]))  
            
        p = list(itertools.permutations(list(np.arange(i)),2))
        l = len(p)
        top_similarity = [0]
        for w in np.arange(l):
            vector1 = mat[p[w][0]]
            vector2 = mat[p[w][1]]
            top_similarity.append(cos(vector1, vector2))
            
        # 计算平均余弦相似度
        mean_similarity.append(sum(top_similarity)/l)
    return(mean_similarity)
```

```python
# 计算主题平均余弦相似度
pos_k = lda_k(pos_corpus, pos_dict)
neg_k = lda_k(neg_corpus, neg_dict)
```

```python
# 绘制主题平均余弦相似度图形
from matplotlib.font_manager import FontProperties  
font = FontProperties(size=14)


fig = plt.figure(figsize=(10,8))
ax1 = fig.add_subplot(211)
ax1.plot(pos_k)
ax1.set_xlabel('正面评论LDA主题数寻优', fontproperties=font)

ax2 = fig.add_subplot(212)
ax2.plot(neg_k)
ax2.set_xlabel('负面评论LDA主题数寻优', fontproperties=font)
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/20623b13d67e4aefbd25befca6627ff7.png)
由图可知，对于正面评论数据，当主题数为2或3时，主题间的平均余弦相似度就达到了最低。因此，对正面评论数据做LDA，可以选择主题数为3；对于负面评论数据，当主题数为3时，主题间的平均余弦相似度也达到了最低。因此，对负面评论数据做LDA，也可以选择主题数为3。

----

### 评价主题分析结果

根据主题数寻优结果，使用Python的Gensim模块对正面评论数据和负面评论数据分别构建LDA主题模型，设置主题数为3，经过LDA主题分析后，每个主题下生成10个最有可能出现的词语以及相应的概率

```python
# LDA主题分析
pos_lda = models.LdaModel(pos_corpus, num_topics = 3, id2word = pos_dict)  
neg_lda = models.LdaModel(neg_corpus, num_topics = 3, id2word = neg_dict)
```

```python
pos_lda.print_topics(num_words = 10)
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3dcca8b31d0b47b9951794adbd697fff.png)

```python
neg_lda.print_topics(num_words = 10)
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bf5dc76322b34c7fa3fe9a026d7d6b74.png)
## 可视化模型训练结果

```python
import pyLDAvis

vis = pyLDAvis.gensim.prepare(pos_lda,pos_corpus,pos_dict)
# 需要的三个参数都可以从硬盘读取的，前面已经存储下来了

# 在浏览器中心打开一个界面
# pyLDAvis.show(vis)

# 在notebook的output cell中显示
pyLDAvis.display(vis)
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/78f14f9f4e5943db93b8adb12ab000a3.png)
综合以上对主题及其中的高频特征词的分析得出，美的电热水器有价格实惠、性价比高、外观好看、服务好等优势。相对而言，用户对美的电热水器的抱怨点主要体现在安装的费用高及售后服务差等方面。因此，用户的购买原因可以总结为以下几个方面：美的是大品牌值得信赖、美的电热水器价格实惠、性价比高。

根据对京东平台上美的电热水器的用户评价情况进行LDA主题模型分析，对美的品牌提出以下两点建议：
1. 在保持热水器使用方便、价格实惠等优点的基础上，对热水器进行加热功能上的改进，从整体上提升热水器的质量。
2. 提升安装人员及客服人员的整体素质，提高服务质量，注重售后服务。建立安装费用收取的明文细则，并进行公布，以减少安装过程中乱收费的现象。适度降低安装费用和材料费用，以此在大品牌的竞争中凸显优势。


```xml
#[完整数据集以及代码](https://mbd.pub/o/bread/aJaWmJxy)
```
