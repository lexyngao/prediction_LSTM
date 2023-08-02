import numpy as np
import math

import tensorflow._api.v2.compat.v1 as tf
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
from scipy.io import savemat
from sklearn.metrics import r2_score
from model import BILSTM_AT

tf.reset_default_graph()
tf.set_random_seed(0)
np.random.seed(0)

# In[] 加载数据
def split_data(data,n,m):
    in_, out_ = [], []
    n_samples = data.shape[0] - n - m
    for i in range(n_samples):
        input_data = []
        for j in range(i, i + n):
            for k in range(0, cols):
                input_data.append(data[j, k])
        in_.append(input_data)
        output_data = []
        for j in range(i + n, i + n + m):
            output_data.append(data[j, 1])
        out_.append(output_data)

    input_data = np.array(in_)
    output_data = np.array(out_)
    return input_data, output_data
data=pd.read_csv("prepared_data.csv").values
cols = data.shape[1]
n_steps = 96*7
m = 96
dimension = cols*n_steps
# data = data.reshape(-1,)
in_,out_ = split_data(data,n_steps,m)

n=range(in_.shape[0])
m=int(0.7 * in_.shape[0])#最后两天测试
train_data = in_[n[0:m],]
test_data = in_[n[m:],]
train_label = out_[n[0:m],]
test_label = out_[n[m:],]
# 归一化
ss_X=StandardScaler().fit(train_data)
ss_Y=StandardScaler().fit(train_label)
# ss_X=MinMaxScaler(feature_range=(0,1)).fit(train_data)
# ss_Y=MinMaxScaler(feature_range=(0,1)).fit(train_label)
train_data = ss_X.transform(train_data)
train_label = ss_Y.transform(train_label)

test_data = ss_X.transform(test_data)
test_label = ss_Y.transform(test_label)

# print(train_data.shape)
# print(train_label.shape)
# print(test_data.shape)
# print(test_label.shape)

in_num = train_data.shape[1]
out_num = train_label.shape[1]

# In[]定义超参数
num_epochs = 20  # 迭代次数
batch_size = 128  # batchsize
alpha = 0.001  # 学习率
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
    N = train_data.shape[0]
    for epoch in range(num_epochs):
        total_batch = int(math.ceil(N / batch_size))
        indices = np.arange(N)
        np.random.shuffle(indices)
        avg_loss = 0
        # 迭代训练，顺便计算训练集loss
        for i in range(total_batch):
            rand_index = indices[batch_size * i:batch_size * (i + 1)]
            x = train_data[rand_index]
            y = train_label[rand_index] # 防溢出
            _, cost = sess.run([optimizer, loss],
                               feed_dict={X: x, Y: y})
            avg_loss += cost / total_batch

        # 计算测试集loss
        valid_data = test_data.reshape(-1, input_features)
        valid_y = test_label.reshape(-1, output_class)
        valid_loss = sess.run(loss, feed_dict={X: valid_data, Y: valid_y})

        train.append(avg_loss)
        valid.append(valid_loss)
        print('epoch:', epoch, ' ,train loss ', avg_loss, ' ,valid loss: ', valid_loss)

    test_data = test_data.reshape(-1, input_features)
    test_pred = sess.run(logits, feed_dict={X: test_data})
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
# In[]计算各种指标
test_pred1 = test_pred.reshape(-1, 96)
test_label1 = test_label.reshape(-1, 96)

# mape
test_mape = np.mean(np.abs((test_pred1 - test_label1) / test_label1))
# rmse
test_rmse = np.sqrt(np.mean(np.square(test_pred1 - test_label1)))
# mae
test_mae = np.mean(np.abs(test_pred1 - test_label1))
# R2
test_r2 = r2_score(test_label1, test_pred1)

print('测试集的mape:', test_mape, ' rmse:', test_rmse, ' mae:', test_mae, ' R2:', test_r2)

# plot test_set result
plt.figure()
plt.plot(test_label1[0,:], c='r', label='true')
plt.plot(test_pred1[0,:], c='b', label='predict')
plt.legend()
plt.show()