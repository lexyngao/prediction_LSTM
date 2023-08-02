import pandas as pd
from sklearn.model_selection import KFold
import gc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import datetime
from sklearn import metrics
import xgboost as xgb
import tensorflow as tf
import time
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
            for k in range(0,cols):
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
    input_data = np.array(in_).reshape(-1,dimension)
    output_data = np.array(out_).reshape(len(out_), -1)
    return input_data, output_data

def result(real, pred, name):
    # ss_X = MinMaxScaler(feature_range=(-1, 1))
    # real = ss_X.fit_transform(real).reshape(-1,)
    # pred = ss_X.transform(pred).reshape(-1,)
    real = real.reshape(-1, )
    pred = pred.reshape(-1, )
    # mape
    test_mape = np.mean(np.abs((pred - real) / real))
    # rmse
    test_rmse = np.sqrt(np.mean(np.square(pred - real)))
    # mae
    test_mae = np.mean(np.abs(pred - real))
    # R2
    test_r2 = r2_score(real, pred)

    print(name, '的mape:%.4f,rmse:%.4f,mae：%.4f,R2:%.4f' % (test_mape, test_rmse, test_mae, test_r2))

# In[2] 加载数据
data = pd.read_csv('prepared_data.csv').iloc[:, :].values
cols = data.shape[1]
n_steps = 96  # 基于前6小时的数据
m = 1
dimension = cols*n_steps #标准的全套数据
input_data, output_data = data_split(data, n_steps,m)
# 数据划分 前70%作为训练集 后30%作为测试集
n = range(input_data.shape[0])
m1 = int(0.7 * input_data.shape[0])
train_data = input_data[n[0:m1], :]
train_label = output_data[n[0:m1]]
test_data = input_data[n[m1:], :]
test_label = output_data[n[m1:]]


# 直观表示的阶段
# plt.figure()
# # plt.plot(output_data[:, -1], c="r", label='real-all')
# plt.plot(test_label[:, -1], c="r", label='real')
# # plt.plot(predict[:, -1], c="b", label='pred')
# # plt.plot(truth[:, -1], c="r", label='real')
# # plt.xticks(range(0, len(test_label), 2928), range(0, 26))
# plt.grid()
# plt.legend()
# plt.title('test_situation')
# # plt.title('train_situation')
# # plt.savefig('result/feature_transformer_result.jpg')
# plt.show()


# 变换为dataframe,列名为由'time','value','is_weekday','weather'组成的历史数据，共96组，96*4列
columns_names = [j+'_'+str(i) for i in range(0,96) for j in ['time','value','is_weekday','weather_cold','weather_warm','weather_hot','month']]
train_data = pd.DataFrame(train_data,columns=columns_names)
train_data['power'] = train_label
test_data = pd.DataFrame(test_data,columns=columns_names)
#test_data['power'] = test_label

feature_train = False
if feature_train == True:
    ###训练xgboost模型
    submission, fi, metric,train_sub = xgmodel(train_data, test_data)
    print('Baseline metrics')
    print(metric)
    fi.to_csv('xgboost_result.csv',index=False)
    print('Feature Importance')
    print(fi)
else:
    fi = pd.read_csv('xgboost_result.csv')

## Train the XGBoost model and obtain the feature importance dataframe (fi)
# In[]
def get_angles(pos, i, d_model):
    # 这里的i等价与公式中的2i和2i+1
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(maximum_position_encoding, d_model):
    angle_rads = get_angles(np.arange(maximum_position_encoding)[:, np.newaxis], np.arange(d_model)[np.newaxis, :],
                            d_model)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :, 0]  # (batch_size, 1, 1, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.
    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)  ## 64
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits = scaled_attention_logits + (mask * -1e9)
        # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0  # d_model self.num_heads 要能够整除
        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        return out2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Dense(d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        # print('------------------\n',seq_len)
        #    x=tf.tile(tf.expand_dims(x,2),self.d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # print(x.shape)
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        # print(x.shape)
        # print(mask.shape)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        return x  # (batch_size, input_seq_len, d_model)


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 input_vocab_size, target_vocab_size, maximum_position_encoding, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, training, enc_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
        final_output = self.final_layer(enc_output)  # (batch_size, inp_seq_len, target_vocab_size)
        return final_output[:, -1, :]

def train_step(inp, tar):
    enc_padding_mask = create_padding_mask(inp)
    # enc_padding_mask = None
    with tf.GradientTape() as tape:
        predictions = transformer(inp, True, enc_padding_mask)
        loss = loss_function(tar, predictions)
    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    train_loss(loss)


# In[]
EPOCHS = 50  # 训练次数 100->50
BATCH_SIZE = 256  ## batchsize
VAL_BATCH_SIZE = 64  # valid batchsize
num_layers = 2  ## encoder 层数 ->2
d_model = 32  # dmodel要是num_heads的整数倍 ->32
dff = 100  # 编码层中前向网络的节点数
num_heads = 32  # 多头注意力的头数量 -> 8
dropout_rate = 0.5  # dropout
input_en_vocab_size = train_data.shape[-1]  # 输入节点数
target_sp_vocab_size = train_label.shape[-1]  # 输出节点数

optimizer = tf.keras.optimizers.Adam(0.01, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
loss_function = tf.keras.losses.MeanSquaredError()
train_loss = tf.keras.metrics.Mean(name='train_loss')
valid_loss = tf.keras.metrics.Mean(name='valid_loss')



transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_en_vocab_size, target_sp_vocab_size,
                          maximum_position_encoding=5000,
                          rate=dropout_rate)
checkpoint_path = "./checkpoints/"
ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=10)


# Filter the feature importance dataframe to keep only the important features
selected_features = fi[fi['importance'] > 0.01]['feature'].tolist()
# selected_features.append('weather_95')

# Extract the selected features from the training and test datasets
train_data_selected = train_data[selected_features]
test_data_selected = test_data[selected_features]

# 归一化
# ss_X=MinMaxScaler().fit(train_data)
# ss_Y=MinMaxScaler().fit(train_label)
ss_X = StandardScaler().fit(train_data_selected)
ss_Y = StandardScaler().fit(train_label)
# n_steps are changed
dimension =len(selected_features)
train_data_selected = ss_X.transform(train_data_selected).reshape(train_data_selected.shape[0], dimension, -1)
test_data_selected = ss_X.transform(test_data_selected).reshape(test_data_selected.shape[0], dimension, -1)
train_label = ss_Y.transform(train_label).reshape(train_data_selected.shape[0], -1)
test_label = ss_Y.transform(test_label).reshape(test_data_selected.shape[0], -1)


# Extract the selected features from the training and test datasets
# train_data_selected = train_data[selected_features]
# test_data_selected = test_data[selected_features]

# Convert train_data_selected dataframe to numpy array
# train_data_selected_array = train_data_selected.values
# Convert test_data_selected dataframe to numpy array
# test_data_selected_array = test_data_selected.values
# Remove all columns from train_data->别真的remove了
# train_data_selected = train_data_selected.drop(train_data_selected.columns, axis=1)
# Remove all columns from test_data
# test_data_selected = test_data_selected.drop(test_data_selected.columns, axis=1)


# Train the transformer model using only the selected features
# transformer_model = YourTransformerModel()
# transformer.fit(train_data_selected, train_label)

# Make predictions using the transformer model
# predictions = transformer.predict(test_data_selected)


###训练lightgbm模型
# submission, fi, metric,train_sub = gbmmodel(train_data, test_data)
# print('Baseline metrics')
# print(metric)
# print('Feature Importance')
# print(fi)
# fi.to_csv('lightgbm_xgboost_result.csv',index=False)
#
# n_steps are changed to 403

train_again = True  # 为 False 的时候就直接加载训练好的模型进行测试
if train_again:
    num_samples, num_valid_samples = train_data_selected.shape[0], test_data_selected.shape[0]
    num_batches, num_valid_batches = int(num_samples / BATCH_SIZE), int(num_valid_samples / VAL_BATCH_SIZE)
    trainloss, validloss = [], []

    for epoch_n in range(EPOCHS):
        start_time = time.time()
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        cost, cost2 = 0, 0
        for batch_n in range(num_batches):
            rand_index = indices[BATCH_SIZE * batch_n:BATCH_SIZE * (batch_n + 1)]
            inp = train_data_selected[rand_index]
            tar = train_label[rand_index]

            train_loss.reset_states()
            inp_tf = tf.convert_to_tensor(inp, dtype=tf.float32)
            tar_tf = tf.convert_to_tensor(tar, dtype=tf.float32)
            train_step(inp_tf, tar_tf)
            cost += train_loss.result().numpy() / num_batches
        end1_time = time.time()
        # # compute validationset loss
        for batch_n in range(num_valid_batches):
            inp = test_data_selected[batch_n * VAL_BATCH_SIZE:(batch_n + 1) * VAL_BATCH_SIZE]
            tar = test_label[batch_n * VAL_BATCH_SIZE:(batch_n + 1) * VAL_BATCH_SIZE]
            valid_loss.reset_states()
            inp_tf = tf.convert_to_tensor(inp, dtype=tf.float32)
            tar_tf = tf.convert_to_tensor(tar, dtype=tf.float32)
            enc_padding_mask = create_padding_mask(inp_tf)
            predictions = transformer(inp_tf, False, enc_padding_mask)
            loss = loss_function(tar_tf, predictions)
            valid_loss(loss)
            cost2 += valid_loss.result().numpy() / num_valid_batches

        end2_time = time.time()
        if (epoch_n + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
        print(epoch_n + 1, 'train time:%.2f s,valid time:%.2f s,train loss:%.5f, valid loss:%.5f.' % (
        end1_time - start_time, end2_time - end1_time, cost, cost2))
        trainloss.append(cost)
        validloss.append(cost2)

    plt.figure
    plt.plot(trainloss, label='train_loss')
    plt.plot(validloss, label='valid_loss')
    plt.legend()
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.title('loss curve')
    plt.savefig('model/feature_transformer loss curve.jpg')

else:
    # 加载最后一个模型
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
    else:
        ckpt.restore('checkpoints/ckpt-20')

# ckpt.restore('./checkpoints/ckpt-16')
# In[] 测试
pred = np.zeros((0, test_label.shape[1]))
batchsize = 64  #
n_samples = test_data.shape[0]
n_baches = np.ceil(n_samples / batchsize)
for i in range(int(n_baches)):
    start = i * batchsize
    end = min((i + 1) * batchsize, n_samples)
    inp_tf = tf.convert_to_tensor(test_data_selected[start:end, :, :], dtype=tf.float32)
    enc_padding_mask = create_padding_mask(inp_tf)
    predictions = transformer(inp_tf, False, enc_padding_mask)
    pred = np.vstack([pred, predictions.numpy()])
predict = ss_Y.inverse_transform(pred)
truth = ss_Y.inverse_transform(test_label)

plt.figure()
plt.plot(truth[:, -1], c="r", label='real')
plt.plot(predict[:, -1], c="b", label='pred')
# plt.plot(truth[:, -1], c="r", label='real')
plt.grid()
plt.legend()
plt.title('test set')
plt.savefig('result/feature_transformer_result.jpg')
plt.show()
np.savez('result/feature_transformer.npz', true=truth, pred=predict)
#
# # In[]计算各种指标
test_label = truth.reshape(-1, )
test_pred = predict.reshape(-1, )
result(test_label, test_pred, 'transformer')