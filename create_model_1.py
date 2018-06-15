import tensorflow as tf
import numpy as np

#create placeholder
#根据变量节点个数建立
#输入层和输出层的节点个数相同
def create_placeholders(columns):
    X = tf.placeholder(dtype=tf.float64, shape=(None, columns), name='channel_input')
    Y = tf.placeholder(dtype=tf.float64, shape=(None, columns), name='codeword_output')
    return X,Y

#初始化权值
#应有几个掩码矩阵，建立非全连接网络
def initialize_weights(parameters):
    weights = {}
    rows = parameters['rows']
    columns = parameters['columns']
    cn_num = parameters['cn_num']
    vn_num = parameters['vn_num']
    itera_num = parameters['itera_num']

    #输入层到第一层（cn）的权值矩阵
    Wi = tf.get_variable("Wi", [columns,cn_num], initializer = tf.ones_initializer(), dtype=tf.float64,
                         trainable=False)
    # tf.random_normal_initializer()
    # tf.ones_initializer()
    # Wi = Wi / 10

    #cn层到vn层
    W_c2v = tf.get_variable("W_c2v", [cn_num,vn_num], initializer = tf.random_normal_initializer(), dtype=tf.float64)
    #####
    # W_c2v = tf.get_variable("W_c2v", [cn_num,vn_num,itera_num], initializer = tf.random_normal_initializer(), dtype=tf.float64)
    W_c2v = W_c2v / 10

    #vn层到cn层
    W_v2c = tf.get_variable("W_v2c", [vn_num,cn_num,itera_num], initializer = tf.ones_initializer(), dtype=tf.float64,
                            trainable=False)
    # tf.random_normal_initializer()
    # tf.ones_initializer()
    # W_v2c = W_v2c / 10

    #cn层到输出层
    Wo = tf.get_variable("Wo", [cn_num,columns], initializer = tf.random_normal_initializer(), dtype=tf.float64)
    Wo = Wo / 10

    #channel到vn节点的权值
    # Wc2v = tf.get_variable("Wc2v", [1,vn_num,itera_num], initializer = tf.random_normal_initializer(), dtype=tf.float64)
    Wc2v = tf.get_variable("Wc2v", [1,vn_num], initializer = tf.random_normal_initializer(), dtype=tf.float64)
    Wc2v = Wc2v / 10

    #channel到out的权值
    Wc2o = tf.get_variable("Wc2o", [1,columns], initializer = tf.random_normal_initializer(), dtype=tf.float64)
    Wc2o = Wc2o / 10

    weights['Wi'] = Wi
    weights['Wo'] = Wo
    weights['W_c2v'] = W_c2v
    weights['W_v2c'] = W_v2c
    weights['Wc2o'] = Wc2o
    weights['Wc2v'] = Wc2v

    return weights

# 建立网络结构，前向传播
def forward_propagation(X, parameters, weights):
    Wi = weights['Wi']
    Wo = weights['Wo']
    W_c2v = weights['W_c2v']
    W_v2c = weights['W_v2c']
    Wc2o = weights['Wc2o']
    Wc2v = weights['Wc2v']

    mask1 = parameters['mask1']
    mask2 = parameters['mask2']
    mask3 = parameters['mask3']
    mask4 = parameters['mask4']
    mask5 = parameters['mask5']
    cn_num = parameters['cn_num']
    vn_num = parameters['vn_num']
    columns = parameters['columns']
    rows = parameters['rows']


    itera_num = parameters['itera_num']
    value = {}

    inv_mask1 = 1. - mask1
    inv_mask3 = 1. - mask3

    # print(mask1)
    # print(mask3)
    help_mask1 = (mask1.sum(axis=0) == 0).astype(np.int)
    help_mask3 = (mask3.sum(axis=0) == 0).astype(np.int)

    # print("help_mask1", help_mask1)
    # print("help_mask3", help_mask3)
    with tf.variable_scope("Model"):
        with tf.variable_scope("input2cn_layer"):
            # 输入节点到校验节点的信息汇总
            # 先对输入层中每个输入除以二后求tanh
            Z = tf.tanh(X / 2)
            value['input2cn_layer_tanh_out'] = Z
            # print(Z)
            # 由于是行向量的堆叠，扩展每个行向量的长度，扩展cn_num次，与mask1.shape[1]相同
            Z = tf.tile(Z, [1,cn_num])
            value['input2cn_layer_tile_out'] = Z
            # print(Z)
            # 然后与权值及reshaped的mask1按位相乘
            Z = tf.multiply(tf.multiply(mask1.T.reshape(1,-1), tf.reshape(Wi, [1,-1])),
                            Z)
            value['input2cn_layer_multiply_out'] = Z
            # print(Z)
            # 这样multiply之后会有好多0的，别忘了加上inv_mask1!!!!!!!
            Z = Z + inv_mask1.T.reshape(1,-1)

            # 对每个行向量，将其reshape为一个矩阵，这里要注意reshape的顺序
            Z = tf.reshape(Z, [-1,cn_num,columns])
            value['input2cn_layer_reshape_out'] = Z
            # print(Z)

            # 对于该矩阵，要在列上相乘，因为reshape的顺序！！！
            Z = tf.reduce_prod(Z, axis=2)
            value['input2cn_layer_prod_out'] = Z
            # print(Z)

            # 在atanh之前还有一步要处理，因为可能存在掩码矩阵全为0的行
            # 那么经过上面的运算后，现在Z的行中可能存在为1的值，这样是不对，正确的做法
            # 是将该1转为0

            Z = Z - help_mask1
            value['input2cn_layer_subhelpmask_out'] = Z

            A = 2 * tf.atanh(Z)
            value['input2cn_layer_atanh_out'] = A
            # print(A)

        # X_ex = lambda t: tf.multiply(tf.matmul(X, mask5), Wc2v[:,:,it])
        X_ex = lambda : tf.multiply(tf.matmul(X, mask5), Wc2v)

        for it in range(itera_num):
            # print("###########itnum : %d##################"%(it))
            with tf.variable_scope("cn2vn_layer_%d"%it):
                #校验节点到变量节点的信息汇总
                # Z = tf.matmul(A, tf.multiply(mask2, W_c2v[:,:,it])) + X_ex(it)
                Z = tf.matmul(A, tf.multiply(mask2, W_c2v)) + X_ex()
                value["cn2vn_layer_%d_matmul_out"%it] = Z
                # print(Z)
                A = tf.tanh(Z / 2)
                value["cn2vn_layer_%d_tanh_out"%it] = A
                # print(A)

            with tf.variable_scope("vn2cn_layer_%d"%it):
                #变量节点到校验节点的信息汇总
                Z = tf.tile(A, [1,cn_num])
                value["vn2cn_layer_%d_tile_out"%it] = Z
                # print(Z)
                Z = tf.multiply(tf.multiply(mask3.T.reshape(1,-1), tf.reshape(W_v2c[:,:,it], [1,-1])),
                                Z)
                value["vn2cn_layer_%d_multiply_out"%it] = Z
                # print(Z)
                # 这样multiply之后会有好多0的，别忘了加上inv_mask1!!!!!!!
                Z = Z + inv_mask3.T.reshape(1,-1)

                Z = tf.reshape(Z, [-1,vn_num,cn_num])
                value["vn2cn_layer_%d_reshape_out"%it] = Z
                # print(Z)
                Z = tf.reduce_prod(Z, axis=2)
                value["vn2cn_layer_%d_prod_out"%it] = Z
                # print(Z)

                Z = Z - help_mask3
                value["vn2cn_layer_%d_subhelpmask_out"%it] = Z

                A = 2 * tf.atanh(Z)
                value["vn2cn_layer_%d_atanh_out"%it] = A
                # print(Z)

        with tf.variable_scope("cn2output_layer"):
            #LLR总和，输出节点的输出信息
            Z = tf.matmul(A, tf.multiply(mask4, Wo)) + tf.multiply(X, Wc2o)
            value["cn_2_outlayer"] = Z
            # print(Z)

            # Z = tf.sigmoid(-Z)
    return Z, value

def compute_cost(Z, Y):
    # logits = tf.transpose(Z)
    # labels= tf.transpose(Y)
    # logits = tf.sigmoid(Z)
    logits = Z
    labels = Y
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=-logits, labels=labels))
    # cost = -1 * tf.reduce_mean(tf.multiply(labels, tf.log(logits)) +
                               # tf.multiply((1 - labels), tf.log(1-logits)))
    return cost
