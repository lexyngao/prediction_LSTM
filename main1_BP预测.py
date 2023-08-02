# -*- coding: utf-8 -*-
# BP回归,bp就是多层感知器
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False

def data_split(data,n,m):
    # 前n个样本的所有值为输入，来预测未来m个功率值
    in_,out_=[],[]
    n_samples=data.shape[0]-n-m
    for i in range(n_samples):
        in_.append(data[i:i+n,:])
        out_.append(data[i+n:i+n+m,-1])
    input_data=np.array(in_).reshape(len(in_),-1)
    output_data=np.array(out_).reshape(len(out_),-1)
    return input_data,output_data

def result(real,pred,name):
    # ss_X = MinMaxScaler(feature_range=(-1, 1))
    # real = ss_X.fit_transform(real).reshape(-1,)
    # pred = ss_X.transform(pred).reshape(-1,)
    real=real.reshape(-1,)
    pred=pred.reshape(-1,)
    # mape
    test_mape = np.mean(np.abs((pred - real) / real))
    # rmse
    test_rmse = np.sqrt(np.mean(np.square(pred - real)))
    # mae
    test_mae = np.mean(np.abs(pred - real))
    # R2
    test_r2 = r2_score(real, pred)

    print(name,'的mape:%.4f,rmse:%.4f,mae：%.4f,R2:%.4f'%(test_mape ,test_rmse, test_mae, test_r2))

# In[] 加载数据
data = pd.read_excel('数据.xlsx').iloc[:10000,1:].values.astype(np.float32)
n_steps=100
m=1
input_data,output_data=data_split(data,n_steps,m)
# 数据划分 前70%作为训练集 后30%作为测试集    
n=range(input_data.shape[0])
m1=int(0.7*input_data.shape[0])
train_data=input_data[n[0:m1],:]
train_label=output_data[n[0:m1]]
test_data=input_data[n[m1:],:]
test_label=output_data[n[m1:]]
# 归一化
ss_X=MinMaxScaler().fit(train_data)
ss_Y=MinMaxScaler().fit(train_label)

train_data = ss_X.transform(train_data).reshape(train_data.shape[0],-1)
test_data = ss_X.transform(test_data).reshape(test_data.shape[0],-1)
train_label = ss_Y.transform(train_label).reshape(train_data.shape[0],-1)
test_label = ss_Y.transform(test_label).reshape(test_data.shape[0],-1)

out_num=test_label.shape[-1]
clf = MLPRegressor(max_iter=100,hidden_layer_sizes=(50))
clf.fit(train_data,train_label)
test_pred=clf.predict(test_data).reshape(-1,out_num)
# In[] 画出测试集的值

# 对测试结果进行反归一化
test_pred = ss_Y.inverse_transform(test_pred)
test_label = ss_Y.inverse_transform(test_label)

plt.figure()
plt.plot(test_label[:,-1], c='r', label='real')
plt.plot(test_pred[:,-1], c='b', label='pred')
plt.legend()
plt.xlabel('test set')
plt.ylabel('power/MW')
plt.savefig('result/bp_result.jpg')
plt.show()
np.savez('result/bp_result.npz', true=test_label, pred=test_pred)

# In[]计算各种指标
test_label = test_label.reshape(-1, )
test_pred = test_pred.reshape(-1, )
result(test_label,test_pred,'bp')










