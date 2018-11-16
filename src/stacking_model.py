# -*- coding: utf-8 -*-
# @Time    : 2018/11/13 4:07 PM
# @Author  : Inf.Turing
# @Site    : 
# @File    : stacking_model.py
# @Software: PyCharm

import os
import pandas as pd

import lightgbm as lgb
import numpy as np
from sklearn.metrics import f1_score

path = './'

w2v_path = path + '/w2v'
train = pd.read_csv(path + '/train_2.csv')
test = pd.read_csv(path + '/test_2.csv')

train_first = pd.read_csv(path + '/train_all.csv')
train['data_type'] = 0
test['data_type'] = 0
train_first['data_type'] = 1
data = pd.concat([train, test, train_first], ignore_index=True).fillna(0)
data['label'] = data.current_service.astype(int)
data = data.replace('\\N', 999)
data['gender'] = data.gender.astype(int)

origin_cate_feature = ['service_type', 'complaint_level', 'contract_type', 'gender', 'is_mix_service',
                       'is_promise_low_consume',
                       'many_over_bill', 'net_service']

origin_num_feature = ['1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee',
                      'age', 'contract_time',
                      'former_complaint_fee', 'former_complaint_num',
                      'last_month_traffic', 'local_caller_time', 'local_trafffic_month', 'month_traffic',
                      'online_time', 'pay_num', 'pay_times', 'service1_caller_time', 'service2_caller_time']

for i in origin_num_feature:
    data[i] = data[i].astype(float)

w2v_features = []
for col in ['1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee']:
    df = pd.read_csv(w2v_path + '/' + col + '.csv')
    df = df.drop_duplicates([col])
    fs = list(df)
    fs.remove(col)
    w2v_features += fs
    data = pd.merge(data, df, on=col, how='left')
count_feature_list = []


def feature_count(data, features=[]):
    if len(set(features)) != len(features):
        print('equal feature !!!!')
        return data
    new_feature = 'count'
    for i in features:
        new_feature += '_' + i.replace('add_', '')
    try:
        del data[new_feature]
    except:
        pass
    temp = data.groupby(features).size().reset_index().rename(columns={0: new_feature})

    data = data.merge(temp, 'left', on=features)
    count_feature_list.append(new_feature)
    return data


data = feature_count(data, ['1_total_fee'])
data = feature_count(data, ['2_total_fee'])
data = feature_count(data, ['3_total_fee'])
data = feature_count(data, ['4_total_fee'])

data = feature_count(data, ['former_complaint_fee'])

data = feature_count(data, ['pay_num'])
data = feature_count(data, ['contract_time'])
data = feature_count(data, ['last_month_traffic'])
data = feature_count(data, ['online_time'])

for i in ['service_type', 'contract_type']:
    data = feature_count(data, [i, '1_total_fee'])
    data = feature_count(data, [i, '2_total_fee'])
    data = feature_count(data, [i, '3_total_fee'])
    data = feature_count(data, [i, '4_total_fee'])

    data = feature_count(data, [i, 'former_complaint_fee'])

    data = feature_count(data, [i, 'pay_num'])
    data = feature_count(data, [i, 'contract_time'])
    data = feature_count(data, [i, 'last_month_traffic'])
    data = feature_count(data, [i, 'online_time'])

# 差值特征
diff_feature_list = ['diff_total_fee_1', 'diff_total_fee_2', 'diff_total_fee_3', 'last_month_traffic_rest',
                     'rest_traffic_ratio',
                     'total_fee_mean', 'total_fee_max', 'total_fee_min', 'total_caller_time', 'service2_caller_ratio',
                     'local_caller_ratio',
                     'total_month_traffic', 'month_traffic_ratio', 'last_month_traffic_ratio', 'pay_num_1_total_fee',
                     '1_total_fee_call_fee', '1_total_fee_call2_fee', '1_total_fee_trfc_fee']

data['diff_total_fee_1'] = data['1_total_fee'] - data['2_total_fee']
data['diff_total_fee_2'] = data['2_total_fee'] - data['3_total_fee']
data['diff_total_fee_3'] = data['3_total_fee'] - data['4_total_fee']

data['pay_num_1_total_fee'] = data['pay_num'] - data['1_total_fee']

data['last_month_traffic_rest'] = data['month_traffic'] - data['last_month_traffic']
data['last_month_traffic_rest'][data['last_month_traffic_rest'] < 0] = 0
data['rest_traffic_ratio'] = (data['last_month_traffic_rest'] * 15 / 1024) / data['1_total_fee']

total_fee = []
for i in range(1, 5):
    total_fee.append(str(i) + '_total_fee')
data['total_fee_mean'] = data[total_fee].mean(1)
data['total_fee_max'] = data[total_fee].max(1)
data['total_fee_min'] = data[total_fee].min(1)

data['total_caller_time'] = data['service2_caller_time'] + data['service1_caller_time']
data['service2_caller_ratio'] = data['service2_caller_time'] / data['total_caller_time']
data['local_caller_ratio'] = data['local_caller_time'] / data['total_caller_time']

data['total_month_traffic'] = data['local_trafffic_month'] + data['month_traffic']
data['month_traffic_ratio'] = data['month_traffic'] / data['total_month_traffic']
data['last_month_traffic_ratio'] = data['last_month_traffic'] / data['total_month_traffic']

data['1_total_fee_call_fee'] = data['1_total_fee'] - data['service1_caller_time'] * 0.15
data['1_total_fee_call2_fee'] = data['1_total_fee'] - data['service2_caller_time'] * 0.15
data['1_total_fee_trfc_fee'] = data['1_total_fee'] - (
        data['month_traffic'] - 2 * data['last_month_traffic']) * 0.3

data.loc[data.service_type == 1, '1_total_fee_trfc_fee'] = None

cate_feature = origin_cate_feature
num_feature = origin_num_feature + count_feature_list + diff_feature_list + w2v_features
for i in cate_feature:
    data[i] = data[i].astype('category')
for i in num_feature:
    data[i] = data[i].astype(float)

feature = cate_feature + num_feature

print(len(feature), feature)

data = data[data.label != 999999]

train_x = data[(data.data_type == 1)][feature]
train_y = data[(data.data_type == 1)].label

test_x = data[(data.data_type == 0) & (data.label != 0)][feature]
test_y = data[(data.data_type == 0) & (data.label != 0)].label

lgb_model = lgb.LGBMClassifier(
    boosting_type="gbdt", num_leaves=120, reg_alpha=0, reg_lambda=0.,
    max_depth=-1, n_estimators=2500, objective='multiclass', metric="None",
    subsample=0.9, colsample_bytree=0.5, subsample_freq=1,
    learning_rate=0.035, random_state=2018, n_jobs=10
)
lgb_model.fit(train_x, train_y, categorical_feature=cate_feature)
print(lgb_model.best_score_)

stacking_path = path + '/stack'
if not os.path.exists(stacking_path):
    print(stacking_path)
    os.makedirs(stacking_path)
    train_proba = lgb_model.predict_proba(test_x[feature])
    test_proba = lgb_model.predict_proba(data[data.label == 0][feature])
    print(len(train_proba), len(test_proba))
    stacking_train = data[(data.data_type == 0) & (data.label != 0)][['user_id']]
    stacking_test = data[data.label == 0][['user_id']]
    for i in range(11):
        stacking_train['stacking_' + str(i)] = train_proba[:, i]
        stacking_test['stacking_' + str(i)] = test_proba[:, i]
    stacking_train.to_csv(stacking_path + '/train.csv', index=False)
    stacking_test.to_csv(stacking_path + '/test.csv', index=False)

score = f1_score(y_true=test_y, y_pred=lgb_model.predict(test_x), average=None)
print(score)