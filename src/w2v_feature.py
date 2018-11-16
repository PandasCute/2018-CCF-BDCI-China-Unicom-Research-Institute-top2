# -*- coding: utf-8 -*-
# @Time    : 2018/11/13 4:08 PM
# @Author  : Inf.Turing
# @Site    : 
# @File    : w2v_feature.py
# @Software: PyCharm

# 预处理复赛数据
import os
import pandas as pd

from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import pandas as pd
import multiprocessing
import numpy as np

L = 10

path = './'
save_path = path + '/w2v'
if not os.path.exists(save_path):
    print(save_path)
    os.makedirs(save_path)

train1 = pd.read_csv(path + '/train_all.csv')
train = pd.read_csv(path + '/train_2.csv')
test = pd.read_csv(path + '/test_2.csv')

data = pd.concat([train, test, train1]).reset_index(drop=True).sample(frac=1, random_state=2018).fillna(0)
data = data.replace('\\N', 999)
sentence = []
for line in list(data[['1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee']].values):
    sentence.append([str(float(l)) for idx, l in enumerate(line)])

print('training...')
model = Word2Vec(sentence, size=L, window=2, min_count=1, workers=multiprocessing.cpu_count(),
                 iter=10)
print('outputing...')

for fea in ['1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee']:
    values = []
    for line in list(data[fea].values):
        values.append(line)
    values = set(values)
    print(len(values))
    w2v = []
    for i in values:
        a = [i]
        a.extend(model[str(float(i))])
        w2v.append(a)
    out_df = pd.DataFrame(w2v)

    name = [fea]
    for i in range(L):
        name.append(name[0] + 'W' + str(i))
    out_df.columns = name
    out_df.to_csv(save_path + '/' + fea + '.csv', index=False)
