import pandas as pd
from sklearn.model_selection import KFold
import gc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import datetime
from sklearn import metrics
import xgboost as xgb
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

def my_scorer(y_true, y_predicted,X_test):
    loss_train = np.sum((y_true - y_predicted)**2, axis=0) / (X_test.shape[0])  #RMSE
    loss_train = loss_train **0.5
    score = 1/(1+loss_train)
    return score

def xgmodel(features, test_features, encoding = 'ohe', n_folds = 4):

    # Extract the ids
    train_ids = features.index
    test_ids = test_features.index

    # Extract the labels for training
    labels = features['power']

    # Remove the ids and target
    features = features.drop(columns = ['power'])


    # One Hot Encoding
    if encoding == 'ohe':
        features = pd.get_dummies(features)
        test_features = pd.get_dummies(test_features)

        # Align the dataframes by the columns
        features, test_features = features.align(test_features, join = 'inner', axis = 1)

        # No categorical indices to record
        cat_indices = 'auto'

    # Integer label encoding
    elif encoding == 'le':

        # Create a label encoder
        label_encoder = LabelEncoder()

        # List for storing categorical indices
        cat_indices = []

        # Iterate through each column
        for i, col in enumerate(features):
            if features[col].dtype == 'object':
                # Map the categorical features to integers
                features[col] = label_encoder.fit_transform(np.array(features[col].astype(str)).reshape((-1,)))
                test_features[col] = label_encoder.transform(np.array(test_features[col].astype(str)).reshape((-1,)))

                # Record the categorical indices
                cat_indices.append(i)

    # Catch error if label encoding scheme is not valid
    else:
        raise ValueError("Encoding must be either 'ohe' or 'le'")

    print('Training Data Shape: ', features.shape)
    print('Testing Data Shape: ', test_features.shape)

    # Extract feature names
    feature_names = list(features.columns)

    # Convert to np arrays
    features = np.array(features)
    test_features = np.array(test_features)

    # Create the kfold object
    k_fold = KFold(n_splits = n_folds, shuffle = True, random_state = 50)

    # Empty array for feature importances
    feature_importance_values = np.zeros(len(feature_names))

    # Empty array for test predictions
    test_predictions = np.zeros(test_features.shape[0])
    train_predictions = np.zeros(features.shape[0])

    # Empty array for out of fold validation predictions
    out_of_fold = np.zeros(features.shape[0])

    # Lists for recording validation and training scores
    #valid_scores = []
    train_scores = []
    train_num = 0
    # Iterate through each fold
    for train_indices, valid_indices in k_fold.split(features):
        train_num += 1
        # Training data for the fold
        train_features, train_labels = features[train_indices], labels[train_indices]
        # Validation data for the fold
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]

        # Create the model
        model = xgb.XGBRegressor(objective = 'reg:linear',n_estimators=16000,min_child_weight=1,num_leaves=20,
                                   learning_rate = 0.01, max_depth=6,n_jobs=20,
                                   subsample = 0.6, colsample_bytree = 0.4, colsample_bylevel = 1)

        # Train the model
        model.fit(train_features, train_labels,eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
                  early_stopping_rounds = 300, verbose = 600)

        # Record the best iteration
        best_iteration = 16000

        # Record the feature importances
        feature_importance_values += model.feature_importances_ / k_fold.n_splits

        # Make predictions
        test_predictions += model.predict(test_features)/ k_fold.n_splits
        train_predictions += model.predict(features)/ k_fold.n_splits
        # Record the out of fold predictions
        out_of_fold = model.predict(valid_features)/ k_fold.n_splits

        # Record the best score
        train_score = my_scorer(valid_labels,out_of_fold,valid_features)

        # valid_scores.append(valid_score)
        train_scores.append(train_score)
        #if len(train_scores) == 0 or train_score >= max(train_scores):
        model.save_model('./weather/xg2lgmodel'+ str(train_num) + '.model')
        # 模型保存为文本格式，便于分析、优化和提供可解释性
        #clf = model.get_booster()
        #clf.dump_model('./weather/dump.txt')
        # Clean up memory
        gc.enable()
        del model, train_features, valid_features
        gc.collect()

    # Make the submission dataframe
    submission = pd.DataFrame({'ID': test_ids, 'power': test_predictions})
    train_sub = pd.DataFrame({'ID': train_ids, 'power': train_predictions})
    # Make the feature importance dataframe
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})

    # Overall validation score
    #valid_auc = roc_auc_score(labels, out_of_fold)

    # Add the overall scores to the metrics
    #valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))

    # Needed for creating dataframe of validation scores
    fold_names = list(range(n_folds))
    fold_names.append('overall')

    # Dataframe of validation scores
    metric = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            })

    return submission, feature_importances, metric,train_sub


###训练lightgbm模型
def gbmmodel(features, test_features, encoding = 'ohe', n_folds = 4):

    # Extract the ids
    train_ids = features.index
    test_ids = test_features.index

    # Extract the labels for training
    labels = features['power']

    # Remove the ids and target
    features = features.drop(columns = ['power'])

    # One Hot Encoding
    if encoding == 'ohe':
        features = pd.get_dummies(features)

        test_features = pd.get_dummies(test_features)

        # Align the dataframes by the columns
        features, test_features = features.align(test_features, join = 'inner', axis = 1)

        # No categorical indices to record
        cat_indices = 'auto'

    # Integer label encoding
    elif encoding == 'le':

        # Create a label encoder
        label_encoder = LabelEncoder()

        # List for storing categorical indices
        cat_indices = []

        # Iterate through each column
        for i, col in enumerate(features):
            if features[col].dtype == 'object':
                # Map the categorical features to integers
                features[col] = label_encoder.fit_transform(np.array(features[col].astype(str)).reshape((-1,)))
                test_features[col] = label_encoder.transform(np.array(test_features[col].astype(str)).reshape((-1,)))

                # Record the categorical indices
                cat_indices.append(i)

    # Catch error if label encoding scheme is not valid
    else:
        raise ValueError("Encoding must be either 'ohe' or 'le'")

    print('Training Data Shape: ', features.shape)
    print('Testing Data Shape: ', test_features.shape)

    # Extract feature names
    feature_names = list(features.columns)

    # Convert to np arrays
    features = np.array(features)
    test_features = np.array(test_features)

    # Create the kfold object
    k_fold = KFold(n_splits = n_folds, shuffle = True, random_state = 50)

    # Empty array for feature importances
    feature_importance_values = np.zeros(len(feature_names))

    # Empty array for test predictions
    test_predictions = np.zeros(test_features.shape[0])
    train_predictions = np.zeros(features.shape[0])
    # Empty array for out of fold validation predictions
    out_of_fold = np.zeros(features.shape[0])

    # Lists for recording validation and training scores
    valid_scores = []
    train_scores = []
    train_num = 0
    # Iterate through each fold
    for train_indices, valid_indices in k_fold.split(features):
        train_num += 1
        # Training data for the fold
        train_features, train_labels = features[train_indices], labels[train_indices]
        # Validation data for the fold
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]

        # Create the model
        model = lgb.LGBMRegressor(objective = 'regression',n_estimators=12000,min_child_samples=20,num_leaves=20,
                                   learning_rate = 0.005, feature_fraction=0.8,
                                   subsample = 0.5, n_jobs = -1, random_state = 50)

        # Train the model
        model.fit(train_features, train_labels, eval_metric = 'rmse',
                  eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names = ['valid', 'train'], categorical_feature = cat_indices,
                  early_stopping_rounds = 2000, verbose = 600)

        # Record the best iteration
        best_iteration = model.best_iteration_

        # Record the feature importances
        feature_importance_values += model.feature_importances_ / k_fold.n_splits

        # Make predictions
        test_predictions += model.predict(test_features, num_iteration = best_iteration)/ k_fold.n_splits
        train_predictions += model.predict(features, num_iteration = best_iteration)/ k_fold.n_splits
        # Record the out of fold predictions
        out_of_fold[valid_indices] = model.predict(valid_features, num_iteration = best_iteration)/ k_fold.n_splits

        # Record the best score
        valid_score = model.best_score_['valid']['rmse']
        train_score = model.best_score_['train']['rmse']

        valid_scores.append(valid_score)
        train_scores.append(train_score)
        model.booster_.save_model('./weather/lg2xgmodel' + str(train_num) + '.model')
        # Clean up memory
        gc.enable()
        del model, train_features, valid_features
        gc.collect()

    # Make the submission dataframe
    submission = pd.DataFrame({'ID': test_ids, 'power': test_predictions})
    train_sub = pd.DataFrame({'ID': train_ids, 'power': train_predictions})
    # Make the feature importance dataframe
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})

    # Overall validation score
    #valid_auc = roc_auc_score(labels, out_of_fold)

    # Add the overall scores to the metrics
    #valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))
    valid_scores.append(np.mean(valid_scores))
    # Needed for creating dataframe of validation scores
    fold_names = list(range(n_folds))
    fold_names.append('overall')

    # Dataframe of validation scores
    metric = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid':valid_scores})

    return submission, feature_importances, metric,train_sub


def data_split(data, n,m):
    # 前n个样本的所有值为输入，来预测未来1个功率值
    in_, out_ = [], []
    n_samples = data.shape[0] - n - m
    for i in range(n_samples):
        # 用前期的n套完整的数据
        # in_.append(data[i:i + n, :])
        # 用前期n套+同期数据 的拆分版本
        for j in range(i,i+n):
            for k in range(0,4):
                in_.append(data[j,k])
        # 历史数据:仅包含values的历史值
        # for j in range(i,i+n):
        #     in_.append(data[j,1])
        # 加上同期的其他数据
        # in_.append(data[i+n,0])
        # for k in range(2, 4):
        #     in_.append(data[i+n,k])
        out_.append(data[i + n, 1])

    # reshape
    # 如果数据格式统一：
    # input_data = np.array(in_).reshape(len(in_), -1)
    # output_data = np.array(out_).reshape(len(out_), -1)
    # 如果不统一:
    input_data = np.array(in_).reshape(-1,dimension) # 24*4history + 3同期数据
    output_data = np.array(out_).reshape(len(out_), -1)
    return input_data, output_data




# In[2] 加载数据
data = pd.read_csv('prepared_data.csv').iloc[:, :].values
n_steps = 96  # 基于前6小时的数据
m = 1
dimension = 4*n_steps
input_data, output_data = data_split(data, n_steps,m)
# 数据划分 前80%作为训练集 后20%作为测试集
n = range(input_data.shape[0])
m1 = int(0.8 * input_data.shape[0])
train_data = input_data[n[0:m1], :]
train_label = output_data[n[0:m1]]
test_data = input_data[n[m1:], :]
test_label = output_data[n[m1:]]
# 变换为dataframe,列名为由'time','value','is_weekday','weather'组成的历史数据，共96组，96*4列
columns_names = [j+'_'+str(i) for i in range(0,96) for j in ['time','value','is_weekday','weather']]
train_data = pd.DataFrame(train_data,columns=columns_names)
train_data['power'] = train_label
test_data = pd.DataFrame(test_data,columns=columns_names)
#test_data['power'] = test_label
###训练xgboost模型
submission, fi, metric,train_sub = xgmodel(train_data, test_data)
print('Baseline metrics')
print(metric)
fi.to_csv('xgboost_result.csv',index=False)
print('Feature Importance')
print(fi)


###训练lightgbm模型
submission, fi, metric,train_sub = gbmmodel(train_data, test_data)
print('Baseline metrics')
print(metric)
print('Feature Importance')
print(fi)
fi.to_csv('lightgbm_xgboost_result.csv',index=False)

