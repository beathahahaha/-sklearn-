# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import warnings
from tqdm import tqdm
import time
warnings.filterwarnings('ignore')


 
train_data = pd.read_csv('dataTrain.csv')
test_data = pd.read_csv('dataA.csv')
data = pd.concat([train_data,test_data]).reset_index(drop=True)

#查看各个特征的名称和数据类型
display(data.info())

# 判断训练集和测试集中是否有缺失值，若有，则打印比例
def judge_missing(df):
    if(df.columns[df.isnull().any()].tolist()==[]):
        print("    there's no missings")
        return
    else:
        t = df.isnull().sum()/len(df)*100
        t = t[t>0]
        t = t.sort_values(ascending = False)
        missing_df = pd.DataFrame({'Features':list(t.index),'Percentage of Missings':list(t)})
        print(missing_df)
        
print("train:")
judge_missing(train_data)  
print("test:")
judge_missing(test_data)

#暴力人工特征
train_data['f47'] = train_data['f1'] * 10 + train_data['f2']
test_data['f47'] = test_data['f1'] * 10 + test_data['f2']


#暴力连续值 + - * /组合
# 暴力Feature 位置
loc_f = ['f1', 'f2', 'f4', 'f5', 'f6']
for df in [train_data, test_data]:
    for i in range(len(loc_f)):
        for j in range(i + 1, len(loc_f)):
            df[f'{loc_f[i]}+{loc_f[j]}'] = df[loc_f[i]] + df[loc_f[j]]
            df[f'{loc_f[i]}-{loc_f[j]}'] = df[loc_f[i]] - df[loc_f[j]]
            df[f'{loc_f[i]}*{loc_f[j]}'] = df[loc_f[i]] * df[loc_f[j]]
            df[f'{loc_f[i]}/{loc_f[j]}'] = df[loc_f[i]] / (df[loc_f[j]]+1)

# 暴力Feature 通话
com_f = ['f43', 'f44', 'f45', 'f46']
for df in [train_data, test_data]:
    for i in range(len(com_f)):
        for j in range(i + 1, len(com_f)):
            df[f'{com_f[i]}+{com_f[j]}'] = df[com_f[i]] + df[com_f[j]]
            df[f'{com_f[i]}-{com_f[j]}'] = df[com_f[i]] - df[com_f[j]]
            df[f'{com_f[i]}*{com_f[j]}'] = df[com_f[i]] * df[com_f[j]]
            df[f'{com_f[i]}/{com_f[j]}'] = df[com_f[i]] / (df[com_f[j]]+1)


#离散值编码
def category_encode(train, test):
    data = pd.concat([train,test]).reset_index(drop=True)
    for col in data.columns[data.dtypes == 'object']:
        lb = LabelEncoder()
        lb.fit(data[col])
        train[col] = lb.transform(train[col])
        test[col] = lb.transform(test[col])
category_encode(train_data, test_data)


#生成有用特征列（去掉id和label列），训练集，测试集，标签
feature_columns = [i for i in train_data.columns if i not in ['id','label']]  #此处需要人工定义，若要模块化，需要规范定义输入数据格式
target = 'label'

train = train_data[feature_columns]
label = train_data[target]

test = test_data[feature_columns]


def model_train(model, model_name, kfold=5):
    global model_name_list, final_auc_list, cost_minutes_list
    
    start_t = time.time()
    oof_preds = np.zeros((train.shape[0]))
    test_preds = np.zeros(test.shape[0])
    skf = StratifiedKFold(n_splits=kfold)
    print(f"Model = {model_name}")
    model_name_list.append(model_name)
    for k, (train_index, test_index) in enumerate(skf.split(train, label)):
        x_train, x_test = train.iloc[train_index, :], train.iloc[test_index, :]
        y_train, y_test = label.iloc[train_index], label.iloc[test_index]

        model.fit(x_train,y_train)

        y_pred = model.predict_proba(x_test)[:,1]

        #oof_preds[test_index] = y_pred.ravel()
        oof_preds[test_index] = y_pred

        auc = roc_auc_score(y_test,y_pred)
        print("- KFold = %d, val_auc = %.4f" % (k, auc))
        test_fold_preds = model.predict_proba(test)[:, 1]
        #test_preds += test_fold_preds.ravel()
        test_preds += test_fold_preds
    
    overall_auc = roc_auc_score(label, oof_preds)
    print("Overall Model = %s, AUC = %.4f" % (model_name, overall_auc))
    final_auc_list.append(overall_auc)
    end_t = time.time()
    cost_minutes = (end_t - start_t)/60
    print(f"cost {cost_minutes} minutes")
    cost_minutes_list.append(cost_minutes)
    return test_preds / kfold


# gbc = GradientBoostingClassifier()
# gbc_test_preds = model_train(gbc, "GradientBoostingClassifier", 5)


train = train[:50000]  #这一步需要再智能化
label = label[:50000]


gbc = GradientBoostingClassifier(
    n_estimators=50, 
    learning_rate=0.1,
    max_depth=5
)
hgbc = HistGradientBoostingClassifier(
    max_iter=100,
    max_depth=5
)
xgbc = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    n_estimators=100, 
    max_depth=6, 
    learning_rate=0.1
)
gbm = LGBMClassifier(
    objective='binary',
    boosting_type='gbdt',
    num_leaves=2 ** 6, 
    max_depth=8,
    colsample_bytree=0.8,
    subsample_freq=1,
    max_bin=255,
    learning_rate=0.05, 
    n_estimators=100, 
    metrics='auc'
)
cbc = CatBoostClassifier(
    iterations=210, 
    depth=6, 
    learning_rate=0.03, 
    l2_leaf_reg=1, 
    loss_function='Logloss', 
    verbose=0
)


clf = StackingClassifier(
    estimators=estimators, 
    final_estimator=LogisticRegression()
)
# model_train(clf,"stacking")
# print("over 2!")

all_model_name = [
    ('gbc', gbc),
    ('hgbc', hgbc),
    ('xgbc', xgbc),
    ('lgbm', gbm),
    ('cbc', cbc),
    ("stacking",clf)
]
model_name_list = []
final_auc_list = []
cost_minutes_list = []
for i in tqdm(range(len(all_model_name))):
    model_train(all_model_name[i][1], all_model_name[i][0])
    
model_df = pd.DataFrame({"model_name":model_name_list, "final_auc":final_auc_list, "cost_minutes":cost_minutes_list})
print(model_df)
print("over")

model_df = pd.DataFrame({"model_name":model_name_list, "final_auc":final_auc_list, "cost_minutes":cost_minutes_list})
model_df.sort_values(columns="final_auc",replace=True)
print(model_df)
print("over")

X_train, X_test, y_train, y_test = train_test_split(train, label, stratify=label, random_state=2022)


clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)
print('auc = %.8f' % auc)


ff = []
for col in feature_columns:
    x_test = X_test.copy() #这里容易出错，建议写成copy.deepcopy(X_test)
    x_test[col] = 0
    auc1 = roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1])
    if auc1 < auc:
        ff.append(col)  #选取有用的特征加入到ff
    print('%5s | %.8f | %.8f' % (col, auc1, auc1 - auc))


clf.fit(X_train[ff], y_train)
y_pred = clf.predict_proba(X_test[ff])[:, 1]
auc = roc_auc_score(y_test, y_pred)
print('auc = %.8f' % auc)


train = train[ff]
test = test[ff]

clf_test_preds = model_train(clf, "StackingClassifier", 10)

submission['label'] = clf_test_preds
submission.to_csv('submission.csv', index=False)


submission = test_data
submission['label'] = clf_test_preds
submission[['id','label']].to_csv('submission_0911628.csv', index=False)












