
import numpy as np
import argparse

parser = argparse.ArgumentParser(prog="preprocess the dataset",
                                 usage="used to convert the binary codeword to initial value")

parser.add_argument('-n', default=15, type=int, help="the codeword length n")
parser.add_argument('-k', default=7,  type=int, help="the information length k")
parser.add_argument('-t', default=2,  type=int, help="the error tolerance t")
parser.add_argument('-odset', default="train_set_x", help="the original datasets will be preprocessed, should be one of:test_set_x, train_set_x, val_set_x",
                    metavar="original_datasets", choices=["test_set_x", "train_set_x", "val_set_x"])

args = parser.parse_args()

n = args.n
k = args.k
t = args.t

original_data_set = args.odset

processed_data_set = args.odset.split('_')[0] + '_' +args.odset.split('_')[2]

folder_path = "../datasets/bch_%d_%d_%d/"%(n,k,t)

itrainx_path = "original/" + original_data_set + ".txt"

otrainx_path = processed_data_set + ".txt"

def initial_value(err_rate, y):
    initial_v = -1 * np.ones((y.size))
    initial_v = np.power(initial_v, y) * np.log((1-err_rate)/err_rate)
    return initial_v

def process_data(err_rate, idata):
    rows, columns = idata.shape
    odata = np.zeros((rows,columns))
    for r in range(rows):
        odata[r,:] = initial_value(err_rate, idata[r,:])
    return odata

def process_datasets(folder_path, itrainx_path, otrainx_path, err_rate, delimiter=' '):
    itrainx = np.loadtxt(folder_path+itrainx_path)
    otrainx = process_data(err_rate, itrainx)
    np.savetxt(folder_path+otrainx_path, otrainx)

process_datasets(folder_path, itrainx_path, otrainx_path, t/n)
