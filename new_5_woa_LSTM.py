# -*- coding: utf-8 -*-
# woa优化bilstm+attention
# glx:但是跑了40min才3个iteration
import numpy as np
# import tensorflow as tf
# glx:这段代码和我本机器的环境不太配合，改成一下的表述了
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow._api.v2.compat.v1 as tf

tf.enable_eager_execution

import math
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from model import BILSTM_AT


# In[]
def fitness(pop, P, T, Pt, Tt):
    tf.reset_default_graph()
    tf.set_random_seed(0)
    alpha = pop[0]  # 学习率
    num_epochs = int(pop[1])  # 迭代次数
    hidden_nodes0 = int(pop[2])  # 第一隐含层神经元
    hidden_nodes = int(pop[3])  # 第二隐含层神经元
    input_features = P.shape[1]
    output_class = T.shape[1]
    batch_size = 128  # batchsize
    # placeholder
    X = tf.placeholder("float", [None, input_features])
    Y = tf.placeholder("float", [None, output_class])
    logits = BILSTM_AT(X, hidden_nodes0, hidden_nodes, input_features, output_class)
    loss = tf.losses.mean_squared_error(predictions=logits, labels=Y)
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        alpha,
        global_step,
        num_epochs, 0.99,
        staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-10).minimize(loss,
                                                                                            global_step=global_step)
    init = tf.global_variables_initializer()

    # 训练
    with tf.Session() as sess:
        sess.run(init)
        N = P.shape[0]
        for epoch in range(num_epochs):
            total_batch = int(math.ceil(N / batch_size))
            indices = np.arange(N)
            np.random.shuffle(indices)
            avg_loss = 0
            # 迭代训练，顺便计算训练集loss
            for i in range(total_batch):
                rand_index = indices[batch_size * i:batch_size * (i + 1)]
                x = P[rand_index]
                y = T[rand_index]
                _, cost = sess.run([optimizer, loss],
                                   feed_dict={X: x, Y: y})
                avg_loss += cost / total_batch
        # 计算测试集的预测值

        test_pred = sess.run(logits, feed_dict={X: Pt})
        test_pred = test_pred.reshape(-1, output_class)

    F2 = np.mean(np.square((test_pred - Tt)))
    return F2


# 鲸鱼优化算法的python实现
def WOA(p_train, t_trian, p_test, t_test):
    '''
        noclus = 维度
        max_iterations = 迭代次数
        noposs 种群数
    '''
    noclus = 4
    max_iterations = 10
    noposs = 5
    lb = [0.0001, 10, 1, 1]
    ub = [0.001, 100, 200, 200]  # 学习率，训练次数，两个隐含层的节点数的寻优上下界
    poss_sols = np.zeros((noposs, noclus))  # whale positions
    gbest = np.zeros((noclus,))  # globally best whale postitions
    b = 2.0

    # 种群初始化，学习率为小数，其他的都是整数
    for i in range(noposs):
        for j in range(noclus):
            if j == 0:
                poss_sols[i][j] = (ub[j] - lb[j]) * np.random.rand() + lb[j]
            else:
                poss_sols[i][j] = np.random.randint(lb[j], ub[j])

    global_fitness = np.inf
    for i in range(noposs):
        cur_par_fitness = fitness(poss_sols[i, :], p_train, t_trian, p_test, t_test)
        if cur_par_fitness < global_fitness:
            global_fitness = cur_par_fitness
            gbest = poss_sols[i].copy()
    # 开始迭代
    trace, trace_pop = [], []
    for it in range(max_iterations):
        for i in range(noposs):
            a = 2.0 - (2.0 * it) / (1.0 * max_iterations)
            r = np.random.random_sample()
            A = 2.0 * a * r - a
            C = 2.0 * r
            l = 2.0 * np.random.random_sample() - 1.0
            p = np.random.random_sample()

            for j in range(noclus):
                x = poss_sols[i][j]
                if p < 0.5:
                    if abs(A) < 1:
                        _x = gbest[j].copy()
                    else:
                        rand = np.random.randint(noposs)
                        _x = poss_sols[rand][j]
                    D = abs(C * _x - x)
                    updatedx = _x - A * D
                else:
                    _x = gbest[j].copy()
                    D = abs(_x - x)
                    updatedx = D * math.exp(b * l) * math.cos(2.0 * math.acos(-1.0) * l) + _x
                # if updatedx < ground[0] or updatedx > ground[1]:
                #    updatedx = (ground[1]-ground[0])*np.random.rand()+ground[0]
                #   randomcount += 1

                poss_sols[i][j] = updatedx
            poss_sols[i, :] = boundary(poss_sols[i, :], lb, ub)  # 边界判断
            fitnessi = fitness(poss_sols[i], p_train, t_trian, p_test, t_test)
            if fitnessi < global_fitness:
                global_fitness = fitnessi
                gbest = poss_sols[i].copy()
        trace.append(global_fitness)
        print("iteration", it + 1, "=", global_fitness, [gbest[i] if i == 0 else int(gbest[i]) for i in range(len(lb))])

        trace_pop.append(gbest)
    return gbest, trace, trace_pop


def boundary(pop, lb, ub):
    # 防止粒子跳出范围,迭代数和节点数都应为整数
    pop = [pop[i] if i == 0 else int(pop[i]) for i in range(len(lb))]
    for j in range(len(lb)):
        if pop[j] > ub[j] or pop[j] < lb[j]:
            if j == 0:
                pop[j] = (ub[j] - lb[j]) * np.random.rand() + lb[j]
            else:
                pop[j] = np.random.randint(lb[j], ub[j])

    return pop


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


n = range(in_.shape[0])
m=int(0.7 * in_.shape[0])#最后两天测试
train_data = in_[n[0:m],]
test_data = in_[n[m:],]
train_label = out_[n[0:m],]
test_label = out_[n[m:],]
# 归一化

ss_X = StandardScaler().fit(train_data)
ss_Y = StandardScaler().fit(train_label)
# ss_X=MinMaxScaler(feature_range=(0,1)).fit(train_data)
# ss_Y=MinMaxScaler(feature_range=(0,1)).fit(train_label)
train_data = ss_X.transform(train_data)
train_label = ss_Y.transform(train_label)

test_data = ss_X.transform(test_data)
test_label = ss_Y.transform(test_label)
# In[]
tf.disable_eager_execution()
best, trace, result = WOA(train_data, train_label, test_data, test_label)
savemat('result/woa_result.mat', {'trace': trace, 'best': best, 'result': result})

# In[] 画图
data = loadmat('result/woa_result.mat')
trace = data['trace'].reshape(-1, )
result = data['result']
best = data['best'].reshape(-1, )

plt.figure()
plt.plot(trace)
plt.title('fitness curve')
plt.xlabel('iteration')
plt.ylabel('fitness value')
plt.savefig('woa_lstm图片保存/fitness curve.png')

plt.figure()
plt.plot(result[:, 0])
plt.title('learning rate optim')
plt.xlabel('iteration')
plt.ylabel('learning rate value')
plt.savefig('woa_lstm图片保存/learning rate optim curve.png')

plt.figure()
plt.plot(result[:, 1])
plt.title('itration optim')
plt.xlabel('iteration')
plt.ylabel('itration value')
plt.savefig('woa_lstm图片保存/itration optim curve.png')

plt.figure()
plt.plot(result[:, 2])
plt.title('first hidden nodes optim')
plt.xlabel('iteration')
plt.ylabel('first hidden nodes value')
plt.savefig('woa_lstm图片保存/first hidden nodes optim curve.png')

plt.figure()
plt.plot(result[:, 3])
plt.title('second hidden nodes optim')
plt.xlabel('iteration')
plt.ylabel('second hidden nodes value')
plt.savefig('woa_lstm图片保存/second hidden nodes optim curve.png')
plt.show()
