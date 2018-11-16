# 预处理复赛数据
import os
import pandas as pd

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import f1_score

path = './'

w2v_path = path + '/w2v'

train = pd.read_csv(path + '/train_2.csv')
test = pd.read_csv(path + '/test_2.csv')

train_stacking = pd.read_csv(path + '/stack/train.csv')
test_stacking = pd.read_csv(path + '/stack/test.csv')

print(len(train), len(test))
train = train.merge(train_stacking, 'left', 'user_id')
test = test.merge(test_stacking, 'left', 'user_id')
print(len(train), len(test))

train_first = pd.read_csv(path + '/train_all.csv')
train['data_type'] = 0
test['data_type'] = 0
train_first['data_type'] = 1
data = pd.concat([train, test], ignore_index=True).fillna(0)
train_first = train_first.fillna(0)
data['label'] = data.current_service.astype(int)
data = data.replace('\\N', 0)
train_first = train_first.replace('\\N', 0)
data['gender'] = data.gender.astype(int)
train_first['gender'] = train_first.gender.astype(int)

data.loc[data['service_type'] == 3, 'service_type'] = 4

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
    train_first[i] = train_first[i].astype(float)

w2v_features = []
for col in ['1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee']:
    df = pd.read_csv(w2v_path + '/' + col + '.csv')
    df = df.drop_duplicates([col])
    fs = list(df)
    fs.remove(col)
    w2v_features += fs
    print(len(data))
    data = pd.merge(data, df, on=col, how='left')
    print(len(data))
print(w2v_features)

count_feature_list = []


def feature_count(data, features=[]):
    if len(set(features)) != len(features):
        print('equal feature !!!!')
        return data
    new_feature = 'count'
    for i in features:
        new_feature += '_' + i.replace('add_', '')
    temp = data.groupby(features).size().reset_index().rename(columns={0: new_feature})
    data = data.merge(temp, 'left', on=features)
    count_feature_list.append(new_feature)
    if 'service_type' in features:
        temp_2 = train_first.groupby(features).size().reset_index().rename(columns={0: 'train_' + new_feature})
        data = data.merge(temp_2, 'left', on=features)
        count_feature_list.append('train_' + new_feature)
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
num_feature = origin_num_feature + count_feature_list + diff_feature_list + w2v_features + ['stacking_' + str(i) for i
                                                                                            in range(11)]
for i in cate_feature:
    data[i] = data[i].astype('category')
for i in num_feature:
    data[i] = data[i].astype(float)

feature = cate_feature + num_feature

print(len(feature), feature)

lgb_model = lgb.LGBMClassifier(
    boosting_type="gbdt", num_leaves=152, reg_alpha=0, reg_lambda=0.,
    max_depth=-1, n_estimators=1000, objective='multiclass', class_weight='balanced',
    subsample=0.9, colsample_bytree=0.5, subsample_freq=1,
    learning_rate=0.03, random_state=2018, n_jobs=10
)
lgb_model.fit(data[data.label != 0][feature], data[data.label != 0].label, categorical_feature=cate_feature)
result_type1 = pd.DataFrame()
result_type1['user_id'] = data[(data.label == 0) & (data.service_type == 1)]['user_id']
result_type1['predict'] = lgb_model.predict(data[(data.label == 0) & (data.service_type == 1)][feature])
print(len(result_type1))


def f1_macro(preds, labels):
    labels = np.argmax(labels.reshape(8, -1), axis=0)
    score = f1_score(y_true=labels, y_pred=preds, average='macro')
    return 'f1_macro', score ** 2, True


X = data[(data.label != 0) & (data.label != 999999)][feature].reset_index(drop=True)
y = data[(data.label != 0) & (data.label != 999999)].label.reset_index(drop=True)

label2current_service = dict(
    zip(range(0, len(set(y))), sorted(list(set(y)))))
current_service2label = dict(
    zip(sorted(list(set(y))), range(0, len(set(y)))))

cv_pred = []
skf = StratifiedKFold(n_splits=5, random_state=20181, shuffle=True)
for index, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(index)
    lgb_model = lgb.LGBMClassifier(
        boosting_type="gbdt", num_leaves=120, reg_alpha=0, reg_lambda=0.,
        max_depth=-1, n_estimators=800, objective='multiclass', class_weight='balanced',
        subsample=0.9, colsample_bytree=0.5, subsample_freq=1,
        learning_rate=0.03, random_state=2018 + index, n_jobs=10, metric="None", importance_type='gain'
    )
    train_x, test_x, train_y, test_y = X.loc[train_index], X.loc[test_index], y.loc[train_index], y.loc[test_index]

    train_x = train_x[train_x.service_type == 4]
    train_y = train_y[(train_x.service_type == 4).index]
    test_x = test_x[test_x.service_type == 4]
    test_y = test_y[(test_x.service_type == 4).index]
    print(test_y.unique())

    eval_set = [(test_x, test_y)]
    lgb_model.fit(train_x, train_y, categorical_feature=cate_feature)
    y_test = lgb_model.predict(data[(data.label == 0) & (data.service_type != 1)][feature])
    y_test = pd.Series(y_test).map(current_service2label)
    if index == 0:
        cv_pred = np.array(y_test).reshape(-1, 1)
    else:
        cv_pred = np.hstack((cv_pred, np.array(y_test).reshape(-1, 1)))

submit = []
for line in cv_pred:
    submit.append(np.argmax(np.bincount(line)))
result = pd.DataFrame()
result['user_id'] = data[(data.label == 0) & (data.service_type != 1)]['user_id']
result['predict'] = submit
result['predict'] = result['predict'].map(label2current_service)
result.loc[result['user_id'] == '4VNcD6kE0sjnAvFX', 'predict'] = 999999
result = result.append(result_type1)

print(len(result), result.predict.value_counts())
print(result.sort_values('user_id').head())
result[['user_id', 'predict']].to_csv(
    path + '/sub.csv', index=False)
