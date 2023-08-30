# -*- coding: utf-8 -*-
# 双向LSTM+attention
# import tensorflow as tf

# glx:这段代码和我本机器的环境不太配合，改成一下的表述了
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow._api.v2.compat.v1 as tf

def weight_variables(shape):
    """偏置"""
    w=tf.Variable(tf.random_normal(shape=shape,mean=0.0,stddev=1.0))
    return w
  
def bias_variables(shape):
    """偏置"""
    b=tf.Variable(tf.constant(0.001,shape=shape))
    return b
def attention(inputs, attention_size, time_major=False):  
    if isinstance(inputs, tuple):  
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.  
        inputs = tf.concat(inputs, 2)  
  
    if time_major:  
        # (T,B,D) => (B,T,D)  
        inputs = tf.transpose(inputs, [1, 0, 2])  
  
    inputs_shape = inputs.shape
    # glx:不知道为啥报错没有value属性,所以这里直接给人家删除了
    sequence_length = inputs_shape[1]  # the length of sequences processed in the antecedent RNN layer
    hidden_size = inputs_shape[2]  # hidden size of the RNN layer
  
    # Attention mechanism  
    W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))  
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))  
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))  
  
    v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))  
    vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))  
    exps = tf.reshape(tf.exp(vu), [-1, sequence_length])  
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])  
  
    # Output of Bi-RNN is reduced with attention vector  
    output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)  
  
    return output   

# 单纯的lstm
def LSTM(x,hidden_nodes0,hidden_nodes,input_features,output_class):
    x_reshape= tf.reshape(x , [-1, 1,input_features])
    #
    with tf.variable_scope("LSTM"):
        rnn_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(hidden_nodes0), tf.nn.rnn_cell.LSTMCell(hidden_nodes0)])
        
        outputs,_=tf.nn.dynamic_rnn(rnn_cell,x_reshape,dtype=tf.float32)
        x_at_reshape=tf.reshape(outputs,[-1,hidden_nodes0])
    #

    with tf.variable_scope("OUTPUT"):
        w_fc2=weight_variables([hidden_nodes0,output_class])
        b_fc2=bias_variables([output_class])
        y_predict=tf.matmul(x_at_reshape,w_fc2)+b_fc2

    return tf.reshape(y_predict, [-1, output_class])

#双向LSTM加注意力机制
def BILSTM_AT(x,hidden_nodes0,hidden_nodes,input_features,output_class):
    x_reshape= tf.reshape(x , [-1, 1,input_features])
    # 双向lstm层
    with tf.variable_scope("BILSTM"):
        # glx:前向的隐含层数为hidden_nodes0
        rnn_cellforword = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(hidden_nodes0), tf.nn.rnn_cell.LSTMCell(hidden_nodes0)])
        # glx:后向的隐含层数为hidden_nodes
        rnn_cellbackword = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(hidden_nodes), tf.nn.rnn_cell.LSTMCell(hidden_nodes)])
        
        outputs,_=tf.nn.bidirectional_dynamic_rnn(rnn_cellforword,rnn_cellbackword,x_reshape,dtype=tf.float32)
        rnn_out=outputs
    #注意力层
    with tf.variable_scope("Attention"):
        attention_size = 64
        attention_out = attention(rnn_out, attention_size, False)
        pool_shape = attention_out.get_shape().as_list()
        nodes = pool_shape[1]
        x_at_reshape=tf.reshape(attention_out,[-1,nodes])
    # 输出
    with tf.variable_scope("OUTPUT"):
        w_fc2=weight_variables([nodes,output_class])
        b_fc2=bias_variables([output_class])
        y_predict=tf.matmul(x_at_reshape,w_fc2)+b_fc2
    return tf.reshape(y_predict, [-1, output_class])


# lstm_attention
def LSTM_AT(x, hidden_nodes0, hidden_nodes, input_features, output_class):
    x_reshape = tf.reshape(x, [-1, 1, input_features])
    #
    with tf.variable_scope("LSTM"):
        rnn_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.LSTMCell(hidden_nodes0), tf.nn.rnn_cell.LSTMCell(hidden_nodes0)])

        outputs, _ = tf.nn.dynamic_rnn(rnn_cell, x_reshape, dtype=tf.float32)
        x_at_reshape = tf.reshape(outputs, [-1, hidden_nodes0])
        rnn_out = outputs
    #
    # 注意力层
    with tf.variable_scope("Attention"):
        attention_size = 64
        attention_out = attention(rnn_out, attention_size, False)
        pool_shape = attention_out.get_shape().as_list()
        nodes = pool_shape[1]
        x_at_reshape = tf.reshape(attention_out, [-1, nodes])

    with tf.variable_scope("OUTPUT"):
        w_fc2 = weight_variables([hidden_nodes0, output_class])
        b_fc2 = bias_variables([output_class])
        y_predict = tf.matmul(x_at_reshape, w_fc2) + b_fc2

    return tf.reshape(y_predict, [-1, output_class])
