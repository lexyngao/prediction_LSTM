# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False

def mape(y_true, y_pred):
    #评价指标MAPE
    record=[]
    for index in range(len(y_true)):
        if abs(y_true[index])>10:
            temp_mape=np.abs((y_pred[index] - y_true[index]) / y_true[index])
            record.append(temp_mape)
    return np.mean(record) * 100
def result(real,pred,name):
    # ss_X = MinMaxScaler(feature_range=(-1, 1))
    # real = ss_X.fit_transform(real).reshape(-1,)
    # pred = ss_X.transform(pred).reshape(-1,)
    real=real.reshape(-1,)
    pred=pred.reshape(-1,)
    # mape
    test_mape = mape(real, pred)
    # rmse
    test_rmse = np.sqrt(np.mean(np.square(pred - real)))
    # mae
    test_mae = np.mean(np.abs(pred - real))
    # R2
    test_r2 = r2_score(real, pred)

    print(name,'的mape:%.4f,rmse:%.4f,mae：%.4f,R2:%.4f'%(test_mape ,test_rmse, test_mae, test_r2))

data0=np.load('result/bp_result.npz')['true']
data1=np.load('result/bp_result.npz')['pred']
data2=np.load('result/lstm_result.npz')['pred']
data3=np.load('result/transformer.npz')['pred']

result(data0,data1,'bp')
result(data0,data2,'lstm')
result(data0,data3,'transformer')


plt.figure()
plt.plot(data0[-900:,-1], c='g', label='real')
plt.plot(data1[-900:,-1], c='b', label='BP')
plt.plot(data2[-900:,-1], c='c', label='LSTM')
plt.plot(data3[-900:,-1], c='r', label='Transformer')
plt.legend()
plt.xlabel('样本点')
plt.ylabel('值')
plt.savefig('figure/结果对比.png')
plt.show()


















