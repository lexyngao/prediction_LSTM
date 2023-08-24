# Import the required libraries
import math
from matplotlib import pyplot as plt
from scipy.io import savemat
from sklearn.metrics import r2_score
# from model import BILSTM_AT
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


########################################################################################
######################用96个历史数据 + 实时的天气数据进行预测##################################

####增加预测点位同期（默认数量为7个同期值）的最大，最小，平均值
def add_mmm(data, no_same_time=7):
    for i in range(data.shape[0]):
        ###找寻所有同期值的数据位置
        powerfilter_index = data[
            (data['time'] == data.loc[i, 'time']) & (data['is_weekday'] == data.loc[i, 'is_weekday'])& (data['is_Sat'] == data.loc[i, 'is_Sat'])].index
        ###筛选出其中的历史值
        powerfilter_index_previous = [j for j in powerfilter_index if j <= i]
        if powerfilter_index_previous == []:  # 未找到同期值，用当期值替代
            data.loc[i, 'pmean'] = data.loc[i, 'value']
            data.loc[i, 'pmax'] = data.loc[i, 'value']
            data.loc[i, 'pmin'] = data.loc[i, 'value']
        elif len(powerfilter_index_previous) > 0 and len(
                powerfilter_index_previous) < no_same_time:  # 未找到默认数量的同期值，用所有历史同期值替代
            data.loc[i, 'pmean'] = np.mean([data.loc[j, 'value'] for j in powerfilter_index_previous])
            data.loc[i, 'pmax'] = np.max([data.loc[j, 'value'] for j in powerfilter_index_previous])
            data.loc[i, 'pmin'] = np.min([data.loc[j, 'value'] for j in powerfilter_index_previous])
        else:  # 找到默认数量的同期值，用所有历史同期值替代
            data.loc[i, 'pmean'] = np.mean([data.loc[j, 'value'] for j in powerfilter_index_previous[-no_same_time:]])
            data.loc[i, 'pmax'] = np.max([data.loc[j, 'value'] for j in powerfilter_index_previous[-no_same_time:]])
            data.loc[i, 'pmin'] = np.min([data.loc[j, 'value'] for j in powerfilter_index_previous[-no_same_time:]])
    return data


def split_time_weather_data(data):
    # out_:m组结果
    # out_single:单个结果
    # data = pd.DataFrame(data)
    in_catboost, out_catboost_single = [], []
    for j in range(len(data)):
        if data[j][2] == True: #weekday
            input_data_catboost = []
            # input_data_catboost.extend([data[i][1] for i in range(j-96,j)])
            input_data_catboost.extend(data[j][[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])
            in_catboost.append(input_data_catboost)
            out_catboost_single.append([data[j][1]])

    return in_catboost, out_catboost_single


powerdata = pd.read_csv("prepared_data_is_weekday.csv")
data = add_mmm(powerdata)
data = data.values
cols = data.shape[1]
n_steps = 96*7
lstm_dimension = 96
dimension = cols*n_steps

in_catboost, out_catboost_single = split_time_weather_data(data)
# columns_names = ['value' + '_'+str(i) for i in range(m)]
columns_names = []
columns_names.extend(
    ['time', 'is_weekday', 'is_Sat', 'is_Sun', 'weather_cold', 'weather_warm', 'weather_hot', 'month', 'pmean', 'pmax',
     'pmin'])
#################80%的训练数据,20%的预测数据
m = int(0.9 * len(in_catboost))
train_data = in_catboost[0:m]
test_data = in_catboost[m:]
train_label = out_catboost_single[0:m]
test_label = out_catboost_single[m:]
train_data = pd.DataFrame(train_data, columns=columns_names)
test_data = pd.DataFrame(test_data, columns=columns_names)


# 用batch数据放入CatBoost来得到一个预测值
catboost = CatBoostRegressor()
catboost.fit(train_data, train_label)

# Step 4: Make predictions using CatBoost Regressor
test_predictions = catboost.predict(test_data)
# train_predictions = catboost.predict(train_data)

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

predict_pairs = test_predictions.tolist()
real_pairs = [i[0] for i in test_label]
rsquare = r2_score(real_pairs, predict_pairs)

# plot test_set result
##获取每日零点时刻数据的位置
test_data_start = test_data[(test_data.time==0) & (test_data['is_weekday'] == True)].index
for i in test_data_start:
    plt.figure()
    ####获取每日零点到23点45分的数据
    plt.plot(real_pairs[i:(i+96)], c='r', label='true')
    plt.plot(predict_pairs[i:(i+96)], c='b', label='predict_weekday')
    plt.legend()
    plt.show()

print('CatBoost_weekday:', rsquare)

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


# Step 1: Load and split the data
# In[] 加载数据
def split_data(data,n,m):
    # out_:m组结果
    # out_single:单个结果
    in_,in_catboost, out_, out_single = [], [],[],[]
    n_samples = data.shape[0] - n - m - (m-1) #m-1: catboost的intervals
    for i in range(n_samples):
        # 只放入周末的数据
        if data[i][2] == True:
            continue
        input_data,input_data_catboost = [],[]
        for j in range(i, i + n + (m-1)):
            if j < i + n :
                for k in range(0, cols):
                    input_data_catboost.append(data[j, k])
            if j >= i + m - 1:
                for k in range(0, cols):
                    input_data.append(data[j, k])
        in_.append(input_data)
        in_catboost.append(input_data_catboost)
        output_data = []
        output_data_single = []
        output_data_single.append(data[i+n+95,1])
        for j in range(i + n +95 , i + n + m+95):
            output_data.append(data[j, 1])
        out_.append(output_data)
        out_single.append(output_data_single)


    input_data_lstm = np.array(in_)
    input_data_catboost = np.array(in_catboost)
    output_data_lstm = np.array(out_)
    output_data_single = np.array(out_single)
    return input_data_lstm,input_data_catboost, output_data_lstm ,output_data_single

data=pd.read_csv("prepared_data_0.csv").values
cols = data.shape[1]
n_steps = 96*7
dimension = cols*n_steps
# 这是lstm预测的output的dimension
prediction_dimension = 96
m = prediction_dimension

in_,in_catboost,out_,out_single = split_data(data,n_steps,m)

n=range(in_catboost.shape[0])
m=int(0.7 * in_catboost.shape[0])#最后两天测试
train_data = in_catboost[n[0:m],]
test_data = in_catboost[n[m:],]
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

# Drop columns with the name "time" except for "time_0"
train_data = train_data.drop(columns=[col for col in train_data.columns if col.startswith('time') and col != 'time_0'])
test_data = test_data.drop(columns=[col for col in test_data.columns if col.startswith('time') and col != 'time_0'])
train_data_lstm = train_data_lstm.drop(columns=[col for col in train_data_lstm.columns if col.startswith('time') and col != 'time_0'])
test_data_lstm = test_data_lstm.drop(columns=[col for col in test_data_lstm.columns if col.startswith('time') and col != 'time_0'])
columns_names = ['time_0']
columns_names += [j+'_'+str(i) for i in range(0,n_steps) for j in ['value','is_weekday','weather_cold','weather_warm','weather_hot','month']]

# Step 2: Train the CatBoost Regressor and evaluate feature importance
# 用batch数据放入CatBoost来得到一个预测值
catboost = CatBoostRegressor()
catboost.fit(train_data, train_label_single)
feature_importance = catboost.get_feature_importance()

# Step 3: Select the features with importance higher than 0.01
selected_features = [columns_names[i] for i, importance in enumerate(feature_importance) if importance > 0.01]

# np.savetxt('selected_features.csv', selected_features, delimiter=',')
# Step 4: Make predictions using CatBoost Regressor
test_predictions = catboost.predict(test_data)
train_predictions = catboost.predict(train_data)


# predictions = catboost.predict(test_data[selected_features])


# Step 5: Prepare the data for LSTM model
X_train_lstm = train_data_lstm[selected_features].values
X_train_lstm = np.hstack((X_train_lstm,train_predictions.reshape(-1,1))) # 添加预测值
X_test_lstm = test_data_lstm[selected_features].values
X_test_lstm = np.hstack((X_test_lstm,test_predictions.reshape(-1,1))) # 添加预测值
y_train_lstm = train_label
y_test_lstm = test_label

# 归一化
ss_X = StandardScaler().fit(X_train_lstm)
ss_Y = StandardScaler().fit(y_train_lstm)
# n_steps are changed
dimension =len(selected_features) + 1 # 添加预测的数据
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
# In[]计算各种指标 batch预测的区间大小是96
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
# plt.figure()
# if prediction_dimension == 1:
#     plt.plot(test_label1[:,:], c='r', label='true')
#     plt.plot(test_pred1[:,:], c='b', label='predict')
# else: # 预测太多维了 只展示预测的第一天
#     plt.plot(test_label1[0, :], c='r', label='true')
#     plt.plot(test_pred1[0, :], c='b', label='predict')
# plt.legend()
# plt.show()

from scipy.optimize import minimize
from numpy.lib.stride_tricks import sliding_window_view

catboost_predictions = test_predictions

# Calculate the CatBoost predictions in sliding window form
catboost_predictions_sliding = sliding_window_view(catboost_predictions, window_shape=(prediction_dimension,))

# 转换成sliding window之后行数有所减少
sliding_size = catboost_predictions_sliding.shape[0]

lstm_predictions = test_pred1[:sliding_size]  # Consider only the first sliding_size lines
test_label_sliding = test_label[:sliding_size]

# Calculate the optimal weighted combination using the Lagrangian multiplier method
def objective(weights):
    combined_predictions = weights[0] * lstm_predictions + weights[1] * catboost_predictions_sliding
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
final_predictions = optimal_weights[0] * lstm_predictions + optimal_weights[1] * catboost_predictions_sliding

# # Save the final predictions to a CSV file
# np.savetxt('final_predictions.csv', final_predictions, delimiter=',')
# np.savetxt('test_label_sliding.csv', test_label_sliding, delimiter=',')
# np.savetxt('lstm_predictions.csv', lstm_predictions, delimiter=',')
# np.savetxt('catboost_predictions_sliding.csv', catboost_predictions_sliding, delimiter=',')
# # plot test_set result
# plt.figure()
# if prediction_dimension == 1:
#     plt.plot(test_label_sliding[:,:], c='r', label='true')
#     plt.plot(test_pred1[:,:], c='b', label='predict_LSTM')
#     plt.plot(catboost_predictions[:, :], c='g', label='predict_Catboost')
#     plt.plot(final_predictions[:, :], c='m', label='predict_Catboost_LSTM')
# else: # 预测太多维了 只展示预测的第一天
#     plt.plot(test_label_sliding[0, :], c='r', label='true')
#     plt.plot(test_pred1[0, :], c='b', label='predict_LSTM')
#     plt.plot(catboost_predictions_sliding[0, :], c='g', label='predict_Catboost')
#     plt.plot(final_predictions[0, :], c='m', label='predict_Catboost_LSTM')
# plt.legend()
# plt.show()


# mape
test_mape = np.mean(np.abs((catboost_predictions_sliding - test_label_sliding) / test_label_sliding))
# rmse
test_rmse = np.sqrt(np.mean(np.square(catboost_predictions_sliding - test_label_sliding)))
# mae
test_mae = np.mean(np.abs(catboost_predictions_sliding - test_label_sliding))
# R2
test_r2 = r2_score(test_label_sliding, catboost_predictions_sliding)

print('CatBoost测试集的mape:', test_mape, ' rmse:', test_rmse, ' mae:', test_mae, ' R2:', test_r2)





# mape
test_mape = np.mean(np.abs((final_predictions - test_label_sliding) / test_label_sliding))
# rmse
test_rmse = np.sqrt(np.mean(np.square(final_predictions - test_label_sliding)))
# mae
test_mae = np.mean(np.abs(final_predictions - test_label_sliding))
# R2
test_r2 = r2_score(test_label_sliding, final_predictions)

print('最终调和后测试集的mape:', test_mape, ' rmse:', test_rmse, ' mae:', test_mae, ' R2:', test_r2)
# test_label_sliding =pd.read_csv("test_label_sliding.csv").values
# lstm_predictions = pd.read_csv("lstm_predictions.csv").values
# catboost_predictions_sliding = pd.read_csv("catboost_predictions_sliding.csv").values
# final_predictions = pd.read_csv("final_predictions.csv").values

# catboost_predictions_sliding = sliding_window_view(catboost_predictions, window_shape=(96,))

# Assuming catboost_predictions_sliding is the sliding window format matrix

# Now catboost_predictions_vector is a continuous vector without repetitive elements
# weekend_predictions = []
# weekend_real = []
# Get the rows that are multiples of 96 and lower than 500
for i in range(0,6):
    print_length = 96*(4+i*2)
    print_start = 96*(2+i*2)
    # print_length = 96*(8+i*7)
    # print_start = 96*(6+i*7)
    rows_to_plot_true = test_label_sliding[(np.arange(test_label_sliding.shape[0]) % 96 == 0) & (np.arange(test_label_sliding.shape[0]) < print_length) & (np.arange(test_label_sliding.shape[0]) >= print_start)]
    rows_to_plot_true = rows_to_plot_true.flatten()
    # weekend_real.append(rows_to_plot_true)
    plt.plot(rows_to_plot_true, c='r', label='true')

    rows_to_plot_catboost = catboost_predictions_sliding[(np.arange(catboost_predictions_sliding.shape[0]) % 96 == 0) & (np.arange(catboost_predictions_sliding.shape[0]) < print_length) & (np.arange(catboost_predictions_sliding.shape[0]) >= print_start)]
    rows_to_plot_catboost = rows_to_plot_catboost.flatten()
    plt.plot(rows_to_plot_catboost, c='y', label='predict_CatBoost')

    rows_to_plot_final = final_predictions[(np.arange(final_predictions.shape[0]) % 96 == 0) & (np.arange(final_predictions.shape[0]) < print_length) & (np.arange(final_predictions.shape[0]) >= print_start)]
    rows_to_plot_final = rows_to_plot_final.flatten()
    # weekend_predictions.append(rows_to_plot_final)
    plt.plot(rows_to_plot_final, c='b', label='predict_Final')

    rows_to_plot_lstm = lstm_predictions[(np.arange(lstm_predictions.shape[0]) % 96 == 0) & (np.arange(lstm_predictions.shape[0]) < print_length) & (np.arange(lstm_predictions.shape[0]) >= print_start)]
    rows_to_plot_lstm = rows_to_plot_lstm.flatten()
    plt.plot(rows_to_plot_lstm, c='m', label='predict_LSTM')
    plt.legend()
    plt.show()

    # mape
    test_mape = np.mean(np.abs((rows_to_plot_lstm - rows_to_plot_true - 1) / (rows_to_plot_true + 1)))
    # rmse
    test_rmse = np.sqrt(np.mean(np.square(rows_to_plot_lstm - rows_to_plot_true)))
    # mae
    test_mae = np.mean(np.abs(rows_to_plot_lstm - rows_to_plot_true))
    # R2
    test_r2 = r2_score(rows_to_plot_true, rows_to_plot_lstm)

    print('weekend-lstm测试集的mape:', test_mape, ' rmse:', test_rmse, ' mae:', test_mae, ' R2:', test_r2)

    # mape
    test_mape = np.mean(np.abs((rows_to_plot_catboost - rows_to_plot_true - 1) / (rows_to_plot_true + 1)))
    # rmse
    test_rmse = np.sqrt(np.mean(np.square(rows_to_plot_catboost - rows_to_plot_true)))
    # mae
    test_mae = np.mean(np.abs(rows_to_plot_catboost - rows_to_plot_true))
    # R2
    test_r2 = r2_score(rows_to_plot_true, rows_to_plot_catboost)

    print('weekend-catboost测试集的mape:', test_mape, ' rmse:', test_rmse, ' mae:', test_mae, ' R2:', test_r2)

    # mape
    test_mape = np.mean(np.abs((rows_to_plot_final - rows_to_plot_true - 1) / (rows_to_plot_true + 1)))
    # rmse
    test_rmse = np.sqrt(np.mean(np.square(rows_to_plot_final - rows_to_plot_true)))
    # mae
    test_mae = np.mean(np.abs(rows_to_plot_final - rows_to_plot_true))
    # R2
    test_r2 = r2_score(rows_to_plot_true, rows_to_plot_final)

    print('weekend最终调和后测试集的mape:', test_mape, ' rmse:', test_rmse, ' mae:', test_mae, ' R2:', test_r2)

# weekend_predictions = pd.DataFrame(weekend_predictions)
# weekend_real = pd.DataFrame(weekend_real)
# # mape
# test_mape = np.mean(np.abs((weekend_predictions - weekend_real ) / (weekend_real + 1)))
# # rmse
# test_rmse = np.sqrt(np.mean(np.square(weekend_predictions - weekend_real)))
# # mae
# test_mae = np.mean(np.abs(weekend_predictions - weekend_real))
# # R2
# test_r2 = r2_score(weekend_real, weekend_predictions)
#
# print('weekend所有所有最终调和后测试集的mape:', test_mape, ' rmse:', test_rmse, ' mae:', test_mae, ' R2:', test_r2)