
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
import time

from set_mask_matrix_1 import set_mask_matrix_1
from create_model_1 import create_placeholders, initialize_weights, forward_propagation, compute_cost

parser = argparse.ArgumentParser(prog ="train model",
                                 usage="train the model and save params and model")

parser.add_argument('-n', default=15, type=int, help="the codeword length n")
parser.add_argument('-k', default=7,  type=int, help="the information length k")
parser.add_argument('-t', default=2,  type=int, help="the error tolerance t")
parser.add_argument('--learning_rate', default=0.05, type=float)
parser.add_argument('--epochs', default=15000, type=int)
parser.add_argument('--inum', default=5, type=int)
parser.add_argument('--batchs', default=32, type=int)
parser.add_argument('--have_test', default=0, type=int, choices=[0,1])

args = parser.parse_args()

n = args.n
k = args.k
t = args.t
learning_rate_0 = args.learning_rate
num_epochs = args.epochs
itera_num = args.inum
batchs = args.batchs
print_cost = True

bch_type = 'bch_%d_%d_%d'%(n,k,t)
H_matrix = 'H_%d_%d_%d.txt'%(n,k,t)

datasets_folder = '../datasets/%s/'%(bch_type)
parameter_folder = '../model_out/%s/parameters/'%(bch_type)
model_folder = '../model_out/%s/model/'%(bch_type)
hmatrix_folder = '../h_matrix/'
weights_params_folder = '../model_out/%s/weights_params/'%(bch_type)

tmp_out_folder = '../model_out/%s/'%(bch_type)

h_matrix = np.loadtxt(hmatrix_folder+H_matrix, delimiter=' ').astype(np.int32)

fd_trainx = open(datasets_folder+'train_x.txt','rb')
fd_trainy = open(datasets_folder+'train_y.txt','rb')
fd_valx   = open(datasets_folder+'val_x.txt','rb')
fd_valy   = open(datasets_folder+'val_y.txt','rb')
if args.have_test:
    fd_testx = open(datasets_folder+'test_x.txt','rb')
    fd_testy = open(datasets_folder+'test_y.txt','rb')
    test_m = len(['' for line in fd_testx])


train_m = len(['' for line in fd_trainx])
trainy_m = len(['' for line in fd_trainy])
assert train_m == trainy_m

val_m = len(['' for line in fd_valx])
valy_m = len(['' for line in fd_valy])
assert val_m == valy_m

fd_trainx.seek(0)
fd_trainy.seek(0)
fd_valx.seek(0)
fd_valy.seek(0)

def generator(fd, batchs=batchs):
    def reader(fd):
        while 1:
            line = fd.readline()
            if line != b'':
                line = line.split()
                data = []
                for l in line:
                    l = float(l)
                    data.append(l)
                return data
            else:
                fd.seek(0)

    while 1:
        data_set = []
        for i in range(batchs):
            data_set.append(reader(fd))
        yield np.array(data_set)

trainx_generator = generator(fd_trainx)
trainy_generator = generator(fd_trainy)
valx_generator = generator(fd_valx)
valy_generator = generator(fd_valy)

parameters = set_mask_matrix_1(h_matrix)

parameters['itera_num'] = itera_num

costs = []

columns = parameters['columns']

X, Y = create_placeholders(columns)

weights = initialize_weights(parameters)

X_p = tf.div(X, np.log((1-t/n) / (t/n)))

Z, value = forward_propagation(X_p, parameters, weights)

cost = compute_cost(Z, Y)

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = learning_rate_0
decay_steps = 5000

decay_rate = 0.8
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, decay_steps, decay_rate)

optimizer = (
    tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)
)

init = tf.global_variables_initializer()
saver = tf.train.Saver()


def eval_on_data(sess, Z, fdx, fdy, datanum, batchs):
    fdx.seek(0)
    fdy.seek(0)
    count = 0
    genx = generator(fdx, batchs)
    geny = generator(fdy, batchs)
    i = 0
    for batch in range(0, datanum, batchs):
        data_x = next(genx)
        data_y = next(geny)
        model_out = sess.run(Z, feed_dict={X:data_x})
        model_out = (model_out < 0)

        tmp = np.bitwise_xor(model_out.astype(np.int), data_y.astype(np.int))
        tmp = np.sum(tmp, axis=1)
        tmp = (tmp == 0).astype('int')
        tmp = np.sum(tmp)
        count += tmp
        i += 1
    fdx.seek(0)
    fdy.seek(0)
    return count/(i*batchs)

def cost_on_data(sess, fdx, fdy, datanum, batchs):
    fdx.seek(0)
    fdy.seek(0)
    count = 0
    genx = generator(fdx, batchs)
    geny = generator(fdy, batchs)
    i = 0
    for batch in range(0, datanum, batchs):
        data_x = next(genx)
        data_y = next(geny)
        cost_on_batch = sess.run(cost, feed_dict={X:data_x, Y:data_y})

        count += cost_on_batch * batchs
        i += 1
    fdx.seek(0)
    fdy.seek(0)
    return count/(i*batchs)

def inference_on_data(sess, Z, fdx, fdo, savenum, batchs):
    fdx.seek(0)
    genx = generator(fdx)
    data = []
    for batch in range(0,savenum,batchs):
        data_x = next(genx)
        tmp = sess.run(Z, feed_dict={X:data_x})
        tmp = (tmp < 0).astype(np.int)
        data.append(tmp)
    data = np.concatenate(data, axis=0)
    np.savetxt(fdo, data, fmt='%1d')
    fdx.seek(0)

print()
print("BCH(%d,%d,%d) start training!"%(n,k,t))
print("itera_num = %d"%(itera_num))
print("train_x shape: (%d,%d)"%(train_m,n))
print("train_y shape: (%d,%d)"%(train_m,n))
print("val_x shape: (%d,%d)"%(val_m,n))
print("val_y shape: (%d,%d)"%(val_m,n))
print("eopchs = %d"%(num_epochs))
print("batch size = %d"%(batchs))
print("learning_rate = %.2f"%(learning_rate_0))
print()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_epochs):
        if epoch == 0:
            t1 = time.time()
        batch_cost = 0

        for batch in range(0, train_m, batchs):
            train_x = next(trainx_generator)
            train_y = next(trainy_generator)
            _ , batch_cost = sess.run([optimizer, cost], feed_dict={X:train_x, Y:train_y})

        if print_cost == True and epoch % 5 == 0:
            print("#####################################")
            print("learn rate: ", sess.run(learning_rate))
            print("epoch_cost after epoch %i: %.6f" % (epoch, cost_on_data(sess, fd_trainx, fd_trainy, train_m, batchs)))
            print("val cost: %.6f"%(cost_on_data(sess, fd_valx, fd_valy, val_m, batchs)))

            print("trian_accuracy: %.5f"%(eval_on_data(sess, Z, fd_trainx, fd_trainy, train_m, batchs)))
            print("val_accuracy: %.5f"%(eval_on_data(sess, Z, fd_valx, fd_valy, val_m, batchs)))
            print("time : %.5f s"%(time.time() - t1))
            t1 = time.time()

        if print_cost == True and epoch % 5 == 0:
            costs.append(batch_cost)

        if eval_on_data(sess, Z, fd_valx, fd_valy, val_m, batchs) == 1.0:
            break

    print()
    print("The end!")
    print("epoch_cost : %.6f" % (cost_on_data(sess, fd_trainx, fd_trainy, train_m, batchs)))
    print("val cost: %.6f"%(cost_on_data(sess, fd_valx, fd_valy, val_m, batchs)))

    print("trian_accuracy: %.6f"%(eval_on_data(sess, Z, fd_trainx, fd_trainy, train_m, batchs)))
    print("val_accuracy: %.6f"%(eval_on_data(sess, Z, fd_valx, fd_valy, val_m, batchs)))
    if args.have_test:
        print("test set accuracy: %.6f"%(eval_on_data(sess, Z, fd_testx, fd_testy, test_m, batchs)))

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per thousands)')
    plt.title("learning rate : %d, %d, %d"%(learning_rate_0, decay_steps, decay_rate))
    plt.show()

    save_path = saver.save(sess, model_folder+'model_bch_%d_%d_%d'%(n,k,t))
    print("Model saved in path: %s"%(save_path))

    fdo_train = open('./train_out.txt', 'wb')
    fdo_val   = open('./val_out.txt', 'wb')
    inference_on_data(sess, Z, fd_trainx, fdo_train, min(batchs*500, train_m), batchs)
    inference_on_data(sess, Z, fd_valx, fdo_val, min(batchs*500, val_m), batchs)

    np.save(weights_params_folder+'parameters.npy', parameters)
    np.save(weights_params_folder+'weights.npy', sess.run(weights))

fd_trainx.close()
fd_trainy.close()
fd_valx.close()
fd_valy.close()
fdo_train.close()
fdo_val.close()
