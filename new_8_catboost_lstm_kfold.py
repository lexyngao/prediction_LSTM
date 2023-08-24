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
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold

def add_mmm(data,no_same_time=7):
    for i in range(data.shape[0]):
        ###找寻所有同期值的数据位置
        powerfilter_index = data[(data['time'] == data.loc[i,'time']) & (data['is_weekday'] == data.loc[i,'is_weekday'])].index
        ###筛选出其中的历史值
        powerfilter_index_previous = [j for j in powerfilter_index if j <= i]
        if powerfilter_index_previous == []:      #未找到同期值，用当期值替代
            data.loc[i,'pmean'] = data.loc[i,'value']
            data.loc[i,'pmax'] = data.loc[i,'value']
            data.loc[i,'pmin'] = data.loc[i, 'value']
        elif len(powerfilter_index_previous) > 0 and len(powerfilter_index_previous) < no_same_time: #未找到默认数量的同期值，用所有历史同期值替代
            data.loc[i, 'pmean'] = np.mean([data.loc[j, 'value'] for j in powerfilter_index_previous])
            data.loc[i, 'pmax'] = np.max([data.loc[j, 'value'] for j in powerfilter_index_previous])
            data.loc[i, 'pmin'] = np.min([data.loc[j, 'value'] for j in powerfilter_index_previous])
        else:   #找到默认数量的同期值，用所有历史同期值替代
            data.loc[i, 'pmean'] = np.mean([data.loc[j, 'value'] for j in powerfilter_index_previous[-no_same_time:]])
            data.loc[i, 'pmax'] = np.max([data.loc[j, 'value'] for j in powerfilter_index_previous[-no_same_time:]])
            data.loc[i, 'pmin'] = np.min([data.loc[j, 'value'] for j in powerfilter_index_previous[-no_same_time:]])
    return data

def split_future_data(data,m):
    # out_:m组结果
    # out_single:单个结果
    in_catboost, out_catboost_single = [], []
    for j in range(m,len(data)):
        input_data_catboost = []
        input_data_catboost.extend([data[i][1] for i in range(j-96,j)])
        input_data_catboost.extend(data[j][[0,2,3,4,5,6,7,8,9]])
        in_catboost.append(input_data_catboost)
        out_catboost_single.append([data[j][1]])
    return in_catboost,out_catboost_single

def split_time_weather_data(data):
    # out_:m组结果
    # out_single:单个结果
    in_catboost, out_catboost_single = [], []
    for j in range(len(data)):
        input_data_catboost = []
        #input_data_catboost.extend([data[i][1] for i in range(j-96,j)])
        input_data_catboost.extend(data[j][[0,2,3,4,5,6,7,8,9]])
        in_catboost.append(input_data_catboost)
        out_catboost_single.append([data[j][1]])
    return in_catboost,out_catboost_single

powerdata = pd.read_csv("prepared_data_0.csv")
data = add_mmm(powerdata) #条件筛选历史值
data = data.values
n_folds = 5


in_catboost,out_catboost_single = split_time_weather_data(data)
#columns_names = ['value' + '_'+str(i) for i in range(m)]
columns_names = []
columns_names.extend(['time','is_weekday','weather_cold','weather_warm','weather_hot','month','pmean','pmax','pmin'])
#################80%的训练数据,20%的预测数据
m=int(0.9 * len(in_catboost))
train_data = in_catboost[0:m]
test_data = in_catboost[m:]
train_label = out_catboost_single[0:m]
test_label = out_catboost_single[m:]
train_data = pd.DataFrame(train_data,columns=columns_names)
test_data = pd.DataFrame(test_data,columns=columns_names)

# 用batch数据放入CatBoost来得到一个预测值
catboost = CatBoostRegressor()
kf = KFold(n_splits=n_folds, shuffle=True)
X = train_data  # Your feature data
y = pd.DataFrame(train_label)# Your target variable
for train_index, val_index in kf.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    # Fit the model on the training data
    catboost.fit(X_train, y_train)
    
    # Evaluate the model on the validation data
    val_predictions = catboost.predict(X_val)
    val_r2 = r2_score(y_val, val_predictions)
    print("Validation R^2:", val_r2)
# catboost.fit(train_data, train_label)

# Step 4: Make predictions using CatBoost Regressor
test_predictions = catboost.predict(test_data)
train_predictions = catboost.predict(train_data)

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
predict_pairs = test_predictions.tolist()
real_pairs = [i[0] for i in test_label]
rsquare = r2_score(real_pairs,predict_pairs)
print(rsquare)

# plot test_set result
###获取每日零点时刻数据的位置
# test_data_start = test_data[test_data.time==0].index
# for i in test_data_start:
#     plt.figure()
#     ####获取每日零点到23点45分的数据
#     plt.plot(real_pairs[i:(i+96)], c='r', label='true')
#     plt.plot(predict_pairs[i:(i+96)], c='b', label='predict_catboost')
#     plt.legend()
#     plt.show()

# Step 5: Prepare the data for LSTM model
prediction_dimension = 1

X_train_lstm = train_data.values
X_train_lstm = np.hstack((X_train_lstm,train_predictions.reshape(-1,1))) # 添加预测值
X_test_lstm = test_data.values
X_test_lstm = np.hstack((X_test_lstm,test_predictions.reshape(-1,1))) # 添加预测值
y_train_lstm = train_label
y_test_lstm = test_label

# 归一化
ss_X = StandardScaler().fit(X_train_lstm)
ss_Y = StandardScaler().fit(y_train_lstm)
# n_steps are changed
dimension = X_train_lstm.shape[1] # 添加预测的数据
train_data_selected = ss_X.transform(X_train_lstm).reshape(X_train_lstm.shape[0], dimension)
test_data_selected = ss_X.transform(X_test_lstm).reshape(X_test_lstm.shape[0], dimension)
train_label = ss_Y.transform(y_train_lstm).reshape(-1,prediction_dimension) # 写死的预测的batch的大小
test_label = ss_Y.transform(y_test_lstm).reshape(-1,prediction_dimension)

# Step 6: Build and train the LSTM model

# In[]定义超参数
in_num = train_data_selected.shape[1]
out_num = train_label.shape[1]
num_epochs = 20  # 迭代次数
batch_size = 128  # batchsize
alpha = 0.0009  # 学习率
hidden_nodes0 = 200  # 第一隐含层神经元
hidden_nodes = 200  # 第二隐含层神经元
input_features = in_num
output_class = out_num
# placeholder
tf.disable_eager_execution()
X = tf.placeholder("float", [None, input_features])
Y = tf.placeholder("float", [None, output_class])
# In[] 初始化
logits = BILSTM_AT(X, hidden_nodes0, hidden_nodes, input_features, output_class)
loss = tf.losses.mean_squared_error(predictions=logits, labels=Y)
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(
    alpha,
    global_step,
    num_epochs, 0.99,
    staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-10).minimize(loss, global_step=global_step)
init = tf.global_variables_initializer()

# In[]训练
train = []
valid = []
with tf.Session() as sess:
    sess.run(init)
    N = train_data_selected.shape[0]
    for epoch in range(num_epochs):
        total_batch = int(math.ceil(N / batch_size))
        indices = np.arange(N)
        np.random.shuffle(indices)
        avg_loss = 0
        # 迭代训练，顺便计算训练集loss
        for i in range(total_batch):
            rand_index = indices[batch_size * i:batch_size * (i + 1)]
            x = train_data_selected[rand_index]
            y = train_label[rand_index] # 防溢出
            _, cost = sess.run([optimizer, loss],
                               feed_dict={X: x, Y: y})
            avg_loss += cost / total_batch

        # 计算测试集loss
        valid_data = test_data_selected.reshape(-1, input_features)
        valid_y = test_label.reshape(-1, output_class)
        valid_loss = sess.run(loss, feed_dict={X: valid_data, Y: valid_y})

        train.append(avg_loss)
        valid.append(valid_loss)
        print('epoch:', epoch, ' ,train loss ', avg_loss, ' ,valid loss: ', valid_loss)

    test_data_selected = test_data_selected.reshape(-1, input_features)
    test_pred = sess.run(logits, feed_dict={X: test_data_selected})
    test_pred = test_pred.reshape(-1, output_class)
# 对测试结果进行反归一化
test_label = ss_Y.inverse_transform(test_label)
test_pred = ss_Y.inverse_transform(test_pred)
savemat('result/bilstm_at_result.mat', {'true': test_label, 'pred': test_pred})

# In[] 画loss曲线
g = plt.figure()
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.plot(train, label='training')
plt.plot(valid, label='testing')
plt.title('loss curve')
plt.legend()
plt.show()
# In[]计算各种指标 batch预测的区间大小是1
test_pred1 = test_pred.reshape(-1, prediction_dimension)
test_label1 = test_label.reshape(-1, prediction_dimension)

# mape
test_mape = np.mean(np.abs((test_pred1 - test_label1) / test_label1))
# rmse
test_rmse = np.sqrt(np.mean(np.square(test_pred1 - test_label1)))
# mae
test_mae = np.mean(np.abs(test_pred1 - test_label1))
# R2
test_r2 = r2_score(test_label1, test_pred1)

print('LSTM利用CatBoost后测试集的mape:', test_mape, ' rmse:', test_rmse, ' mae:', test_mae, ' R2:', test_r2)

# plot test_set result
###获取每日零点时刻数据的位置
# test_data_start = test_data[test_data.time==0].index
# for i in test_data_start:
#     plt.figure()
#     ####获取每日零点到23点45分的数据
#     plt.plot(test_label1[i:(i+96)], c='r', label='true')
#     plt.plot(test_pred1[i:(i+96)], c='b', label='predict_lstm')
#     plt.legend()
#     plt.show()

#调和的部分
from scipy.optimize import minimize
from numpy.lib.stride_tricks import sliding_window_view

catboost_predictions = test_predictions.reshape(-1,1)
lstm_predictions = test_pred1
test_label_sliding = test_label

# Calculate the optimal weighted combination using the Lagrangian multiplier method
def objective(weights):
    combined_predictions = weights[0] * lstm_predictions + weights[1] * catboost_predictions
    return np.mean((combined_predictions - test_label_sliding) ** 2)

# Define the constraint
def constraint(weights):
    return np.sum(weights) - 1

# Initial weights (equal weights for both models)
initial_weights = [0.5, 0.5]

# Minimize the objective function subject to the constraints
result = minimize(objective, initial_weights, constraints={'type': 'eq', 'fun': constraint})

# Get the optimal weights
optimal_weights = result.x

# Calculate the final forecast results using the optimal weights
final_predictions = optimal_weights[0] * lstm_predictions + optimal_weights[1] * catboost_predictions

# plot test_set result
###获取每日零点时刻数据的位置
test_data_start = test_data[test_data.time==0].index
for i in test_data_start:
    plt.figure()
    ####获取每日零点到23点45分的数据
    plt.plot(test_label1[i:(i+96)], c='r', label='true')
    plt.plot(test_pred1[i:(i+96)], c='g', label='predict_lstm')
    plt.plot(catboost_predictions[i:(i + 96)], c='y', label='predict_catboost')
    plt.plot(final_predictions[i:(i + 96)], c='b', label='predict_final')
    plt.legend()
    plt.show()
# mape
test_mape = np.mean(np.abs((final_predictions - test_label_sliding) / test_label_sliding))
# rmse
test_rmse = np.sqrt(np.mean(np.square(final_predictions - test_label_sliding)))
# mae
test_mae = np.mean(np.abs(final_predictions - test_label_sliding))
# R2
test_r2 = r2_score(test_label_sliding, final_predictions)

print('最终调和后测试集的mape:', test_mape, ' rmse:', test_rmse, ' mae:', test_mae, ' R2:', test_r2)