import math
import os
import numpy as np
import pandas as pd
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn, rnn
import sklearn
from sklearn.utils import shuffle
import matplotlib.pyplot as plt



def params_ave(weights_pred, k, emd_ave, W_i2h1_ave, W_h2h1_ave, b_i2h1_ave, b_h2h1_ave, W_i2h2_ave, W_h2h2_ave, b_i2h2_ave, b_h2h2_ave, W_dense_ave, b_dense_ave):

	"""
	Taking moveing average of all parameters.

	Dispose of 10 samples at the begining of collecting 50 training samples.
	"""
    
    #W_emd
    weight0 = weights_pred[0].asnumpy()
    #W_i2h1
    weight1 = weights_pred[1].asnumpy()
    #W_h2h1
    weight2 = weights_pred[2].asnumpy()
    #b_i2h1
    weight3 = weights_pred[3].asnumpy()
    #b_h2h1
    weight4 = weights_pred[4].asnumpy()
    #W_i2h2
    weight5 = weights_pred[5].asnumpy()
    #W_h2h2
    weight6 = weights_pred[6].asnumpy()
    #b_i2h2
    weight7 = weights_pred[7].asnumpy()
    #b_h2h2
    weight8 = weights_pred[8].asnumpy()
    #W_dense
    weight9 = weights_pred[9].asnumpy()
    #b_dense_bias
    weight10 = weights_pred[10].asnumpy()
    
    #averaging
    num_samples = k - 10

    emd_ave = ((num_samples - 1)*emd_ave + weight0)/num_samples
    W_i2h1_ave = ((num_samples - 1)*W_i2h1_ave + weight1)/num_samples
    W_h2h1_ave = ((num_samples - 1)*W_h2h1_ave + weight2)/num_samples
    b_i2h1_ave = ((num_samples - 1)*b_i2h1_ave + weight3)/num_samples
    b_h2h1_ave = ((num_samples - 1)*b_h2h1_ave + weight4)/num_samples
    W_i2h2_ave = ((num_samples - 1)*W_i2h2_ave + weight5)/num_samples
    W_h2h2_ave = ((num_samples - 1)*W_h2h2_ave + weight6)/num_samples
    b_i2h2_ave = ((num_samples - 1)*b_i2h2_ave + weight7)/num_samples
    b_h2h2_ave = ((num_samples - 1)*b_h2h2_ave + weight8)/num_samples
    W_dense_ave = ((num_samples - 1)*W_dense_ave + weight9)/num_samples
    b_dense_ave = ((num_samples - 1)*b_dense_ave + weight10)/num_samples

    return emd_ave, W_i2h1_ave, W_h2h1_ave, b_i2h1_ave, b_h2h1_ave, W_i2h2_ave, W_h2h2_ave, b_i2h2_ave, b_h2h2_ave, W_dense_ave, b_dense_ave 




def get_ave_param(params_pred, weights_pred, emd_ave, W_i2h1_ave, W_h2h1_ave, b_i2h1_ave, b_h2h1_ave, W_i2h2_ave, W_h2h2_ave, b_i2h2_ave, b_h2h2_ave, W_dense_ave, b_dense_ave):

	"""
	Obtain moving average at each 50th sample, and then update the predictive parameters.
	"""
    
    #W_emd
    param0 = params_pred[0]
    weight0 = weights_pred[0].asnumpy()
    #W_i2h1
    param1 = params_pred[1]
    weight1 = weights_pred[1].asnumpy()
    #W_h2h1
    param2 = params_pred[2]
    weight2 = weights_pred[2].asnumpy()
    #b_i2h1
    param3 = params_pred[3]
    weight3 = weights_pred[3].asnumpy()
    #b_h2h1
    param4 = params_pred[4]
    weight4 = weights_pred[4].asnumpy()
    #W_i2h2
    param5 = params_pred[5]
    weight5 = weights_pred[5].asnumpy()
    #W_h2h2
    param6 = params_pred[6]
    weight6 = weights_pred[6].asnumpy()
    #b_i2h2
    param7 = params_pred[7]
    weight7 = weights_pred[7].asnumpy()
    #b_h2h2
    param8 = params_pred[8]
    weight8 = weights_pred[8].asnumpy()
    #W_dense
    param9 = params_pred[9]
    weight9 = weights_pred[9].asnumpy()
    #b_dense_bias
    param10 = params_pred[10]
    weight10 = weights_pred[10].asnumpy()
    
    #averaging
    emd_ave = ((50 - 10 - 1)*emd_ave + weight0)/(50 - 10)
    W_i2h1_ave = ((50 - 10 - 1)*W_i2h1_ave + weight1)/(50 - 10)
    W_h2h1_ave = ((50 - 10 - 1)*W_h2h1_ave + weight2)/(50 - 10)
    b_i2h1_ave = ((50 - 10 - 1)*b_i2h1_ave + weight3)/(50 - 10)
    b_h2h1_ave = ((50 - 10 - 1)*b_h2h1_ave + weight4)/(50 - 10)
    W_i2h2_ave = ((50 - 10 - 1)*W_i2h2_ave + weight5)/(50 - 10)
    W_h2h2_ave = ((50 - 10 - 1)*W_h2h2_ave + weight6)/(50 - 10)
    b_i2h2_ave = ((50 - 10 - 1)*b_i2h2_ave + weight7)/(50 - 10)
    b_h2h2_ave = ((50 - 10 - 1)*b_h2h2_ave + weight8)/(50 - 10)
    W_dense_ave = ((50 - 10 - 1)*W_dense_ave + weight9)/(50 - 10)
    b_dense_ave = ((50 - 10 - 1)*b_dense_ave + weight10)/(50 - 10)

    param0.set_data(emd_ave)
    params_pred[0] = param0

    param1.set_data(W_i2h1_ave)
    params_pred[1] = param1

    param2.set_data(W_h2h1_ave)
    params_pred[2] = param2

    param3.set_data(b_i2h1_ave)
    params_pred[3] = param3

    param4.set_data(b_h2h1_ave)
    params_pred[4] = param4

    param5.set_data(W_i2h2_ave)
    params_pred[5] = param5

    param6.set_data(W_h2h2_ave)
    params_pred[6] = param6

    param7.set_data(b_i2h2_ave)
    params_pred[7] = param7

    param8.set_data(b_h2h2_ave)
    params_pred[8] = param8

    param9.set_data(W_dense_ave)
    params_pred[9] = param9

    param10.set_data(b_dense_ave)
    params_pred[10] = param10

    return params_pred
  



