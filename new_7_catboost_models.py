# Import the required libraries
import math
from matplotlib import pyplot as plt
from scipy.io import savemat
from sklearn.metrics import r2_score
from model import BILSTM_AT
import catboost
from catboost import CatBoostRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow._api.v2.compat.v1 as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os
import glob
import datetime
import joblib


# Step 1: Load and split the data
# In[] 加载数据
def split_data(data,n,m):
    # out_:m组结果
    # out_single:单个结果
    in_, out_, out_single = [], [],[]
    n_samples = data.shape[0] - n - m
    for i in range(n_samples):
        input_data= []
        for j in range(i, i + n):
            for k in range(0, cols):
                    input_data.append(data[j, k])
        in_.append(input_data)
        output_data = []
        output_data_single = []
        output_data_single.append(data[i+n,1])
        for j in range(i + n , i + n + m):
            output_data.append(data[j, 1])
        out_.append(output_data)
        out_single.append(output_data_single)


    input_data_lstm = np.array(in_)
    output_data_lstm = np.array(out_)
    output_data_single = np.array(out_single)
    return input_data_lstm, output_data_lstm ,output_data_single

data=pd.read_csv("prepared_data.csv").values
cols = data.shape[1]
n_steps = 96
dimension = cols*n_steps
# 这是lstm预测的output的dimension
prediction_dimension = 1
m = prediction_dimension

in_,out_,out_single = split_data(data,n_steps,m)

n=range(in_.shape[0])
m=int(0.9 * in_.shape[0])#最后两天测试
train_data = in_[n[0:m],]
test_data = in_[n[m:],]
train_data_lstm = in_[n[0:m],]
test_data_lstm = in_[n[m:],]

train_label = out_[n[0:m],]
test_label = out_[n[m:],]
train_label_single = out_single[n[0:m],]
test_label_single = out_single[n[m:],]

# 变换为dataframe,列名为由'time','value','is_weekday','weather'组成的历史数据，共n_steps组，每组7列
columns_names = [j+'_'+str(i) for i in range(0,n_steps) for j in ['time','value','is_weekday','weather_cold','weather_warm','weather_hot','month']]
train_data = pd.DataFrame(train_data,columns=columns_names)
test_data = pd.DataFrame(test_data,columns=columns_names)
train_data_lstm = pd.DataFrame(train_data_lstm,columns=columns_names)
test_data_lstm = pd.DataFrame(test_data_lstm,columns=columns_names)

# Step 2: Train the CatBoost Regressor and evaluate feature importance
# Load the trained models from the "all model" folder

import glob

model_files = glob.glob("all model/predict_*.model")
models = []

# Sort the file names based on the sequence number
sorted_files = sorted(model_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))

# Load the models in the sorted order
for file in sorted_files:
    model = CatBoostRegressor().load_model(file)
    models.append(model)

# Generate predictions using the loaded models
test_predictions = []
train_predictions = []
new_test = test_data

# for j in range(2):
    # first model: do not need adjustment
model = models[0]
test_predictions = model.predict(test_data)
test_predictions_final = []
test_predictions_final.append(test_predictions[0])
feature_importance = model.get_feature_importance()

# for j in range(2):
# new_test = test_data[0,:]
for i in range(1,96):
    new_test = new_test.iloc[1:,:]
    ##追加的列名
    column_name_to_fill = 'predict_value_' + str(i)
    ##删除的列名
    column_name_to_delete = 'value_' + str(96-i)
    new_test.loc[:, column_name_to_fill] = test_predictions_final[-1]
    del new_test[column_name_to_delete]
    #in_.append(p_.tolist())
    out_ = test_label_single[i:]
    model = models[i]
    # train_predictions = model.predict(train_data)
    test_predictions_temp = model.predict(new_test)
    test_predictions_final.append(test_predictions_temp[i])
    if i == 95:
        feature_importance = model.get_feature_importance()

# Plot the train_predictions_final
plt.plot(test_predictions_final)
plt.plot(test_label_single[0:97],c='r',label = 'true')
plt.xlabel('Index')
plt.ylabel('Prediction')
plt.title('Test Predictions')
plt.show()

