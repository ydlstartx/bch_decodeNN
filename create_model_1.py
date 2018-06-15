import tensorflow as tf
import numpy as np

#create placeholder
def create_placeholders(columns):
    X = tf.placeholder(dtype=tf.float64, shape=(None, columns), name='channel_input')
    Y = tf.placeholder(dtype=tf.float64, shape=(None, columns), name='codeword_output')
    return X,Y

def initialize_weights(parameters):
    weights = {}
    rows = parameters['rows']
    columns = parameters['columns']
    cn_num = parameters['cn_num']
    vn_num = parameters['vn_num']
    itera_num = parameters['itera_num']

    Wi = tf.get_variable("Wi", [columns,cn_num], initializer = tf.ones_initializer(), dtype=tf.float64,
                         trainable=False)

    W_c2v = tf.get_variable("W_c2v", [cn_num,vn_num], initializer = tf.random_normal_initializer(), dtype=tf.float64)
    W_c2v = W_c2v / 10

    W_v2c = tf.get_variable("W_v2c", [vn_num,cn_num,itera_num], initializer = tf.ones_initializer(), dtype=tf.float64,
                            trainable=False)

    Wo = tf.get_variable("Wo", [cn_num,columns], initializer = tf.random_normal_initializer(), dtype=tf.float64)
    Wo = Wo / 10

    Wc2v = tf.get_variable("Wc2v", [1,vn_num], initializer = tf.random_normal_initializer(), dtype=tf.float64)
    Wc2v = Wc2v / 10

    Wc2o = tf.get_variable("Wc2o", [1,columns], initializer = tf.random_normal_initializer(), dtype=tf.float64)
    Wc2o = Wc2o / 10

    weights['Wi'] = Wi
    weights['Wo'] = Wo
    weights['W_c2v'] = W_c2v
    weights['W_v2c'] = W_v2c
    weights['Wc2o'] = Wc2o
    weights['Wc2v'] = Wc2v

    return weights

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

    help_mask1 = (mask1.sum(axis=0) == 0).astype(np.int)
    help_mask3 = (mask3.sum(axis=0) == 0).astype(np.int)

    with tf.variable_scope("Model"):
        with tf.variable_scope("input2cn_layer"):

            Z = tf.tanh(X / 2)
            value['input2cn_layer_tanh_out'] = Z

            Z = tf.tile(Z, [1,cn_num])
            value['input2cn_layer_tile_out'] = Z

            Z = tf.multiply(tf.multiply(mask1.T.reshape(1,-1), tf.reshape(Wi, [1,-1])),
                            Z)
            value['input2cn_layer_multiply_out'] = Z

            Z = Z + inv_mask1.T.reshape(1,-1)

            Z = tf.reshape(Z, [-1,cn_num,columns])
            value['input2cn_layer_reshape_out'] = Z
 

            Z = tf.reduce_prod(Z, axis=2)
            value['input2cn_layer_prod_out'] = Z

            Z = Z - help_mask1
            value['input2cn_layer_subhelpmask_out'] = Z

            A = 2 * tf.atanh(Z)
            value['input2cn_layer_atanh_out'] = A
            # print(A)

        X_ex = lambda : tf.multiply(tf.matmul(X, mask5), Wc2v)

        for it in range(itera_num):

            with tf.variable_scope("cn2vn_layer_%d"%it):
                Z = tf.matmul(A, tf.multiply(mask2, W_c2v)) + X_ex()
                value["cn2vn_layer_%d_matmul_out"%it] = Z

                A = tf.tanh(Z / 2)
                value["cn2vn_layer_%d_tanh_out"%it] = A

            with tf.variable_scope("vn2cn_layer_%d"%it):

                Z = tf.tile(A, [1,cn_num])
                value["vn2cn_layer_%d_tile_out"%it] = Z

                Z = tf.multiply(tf.multiply(mask3.T.reshape(1,-1), tf.reshape(W_v2c[:,:,it], [1,-1])),
                                Z)
                value["vn2cn_layer_%d_multiply_out"%it] = Z

                Z = Z + inv_mask3.T.reshape(1,-1)

                Z = tf.reshape(Z, [-1,vn_num,cn_num])
                value["vn2cn_layer_%d_reshape_out"%it] = Z

                Z = tf.reduce_prod(Z, axis=2)
                value["vn2cn_layer_%d_prod_out"%it] = Z

                Z = Z - help_mask3
                value["vn2cn_layer_%d_subhelpmask_out"%it] = Z

                A = 2 * tf.atanh(Z)
                value["vn2cn_layer_%d_atanh_out"%it] = A

        with tf.variable_scope("cn2output_layer"):

            Z = tf.matmul(A, tf.multiply(mask4, Wo)) + tf.multiply(X, Wc2o)
            value["cn_2_outlayer"] = Z

    return Z, value

def compute_cost(Z, Y):

    logits = Z
    labels = Y
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=-logits, labels=labels))

    return cost
