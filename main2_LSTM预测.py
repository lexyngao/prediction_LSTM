# -*- coding: utf-8 -*-
#lstm
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from tensorflow.keras.layers import Input,Dense, LSTM,Flatten
from tensorflow.keras.models import Model
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
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

class LSTM_(object):
    def __init__(self, input_dim,feat_dim,output_dim,hidden_unit):
        self.feat_dim = feat_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.filters = hidden_unit
        self.units = hidden_unit
        self._estimators = {}
        
    def build_model(self):
        inp = Input(shape=(self.input_dim, self.feat_dim))
        x = LSTM(units=self.units,
                  kernel_initializer="glorot_uniform",
                  bias_initializer="zeros",
                  return_sequences=False)(inp)
        x=Flatten()(x)
        out = Dense(self.output_dim, activation=None)(x)
        model = Model(inputs=inp, outputs=out)
        return model
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
ss_X=StandardScaler().fit(train_data)
ss_Y=StandardScaler().fit(train_label)
train_data = ss_X.transform(train_data).reshape(train_data.shape[0],n_steps,-1)
test_data = ss_X.transform(test_data).reshape(test_data.shape[0],n_steps,-1)
train_label = ss_Y.transform(train_label).reshape(train_data.shape[0],-1)
test_label = ss_Y.transform(test_label).reshape(test_data.shape[0],-1)
# In[]定义超参数
num_epochs = 20#迭代次数
batch_size = 256# batchsize
lr = 0.001# 学习率
hidden = 20#lstm节点数量
sequence,feature=train_data.shape[-2:]
output_node=train_label.shape[1]
model=LSTM_(sequence,feature,output_node,hidden).build_model()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')
# model.summary()
train_again=True  #为 False 的时候就直接加载训练好的模型进行测试
#训练模型
if train_again:
    history=model.fit(train_data,train_label, epochs=num_epochs,validation_data=(test_data,test_label),batch_size=batch_size, verbose=1)
    model.save_weights('model/LSTM_model.h5')
    # 画loss曲线
    plt.figure()
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.plot( history.history['loss'], label='train_loss')
    plt.plot( history.history['val_loss'], label='valid_loss')
    plt.title('loss curve')
    plt.legend()
    plt.savefig('model/LSTM_model_loss.jpg')
    
    
else:#加载模型
    model.load_weights('model/LSTM_model.h5')
test_pred=model.predict(test_data)
  

# 对测试结果进行反归一化
test_label1 = ss_Y.inverse_transform(test_label)
test_pred1 = ss_Y.inverse_transform(test_pred)


np.savez('result/lstm_result.npz',true=test_label1,pred=test_pred1)
    
plt.figure()
plt.plot(test_label[:,-1], c='r', label='real')
plt.plot(test_pred[:,-1], c='b', label='pred')
plt.legend()
plt.xlabel('test set')
plt.ylabel('power/MW')
plt.savefig('result/lstm_result.jpg')
plt.show()

# In[]计算各种指标
test_label = test_label1.reshape(-1, )
test_pred = test_pred1.reshape(-1, )
result(test_label,test_pred,'lstm')
