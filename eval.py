import math
import os
import numpy as np
import pandas as pd
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn, rnn



def train_eval_loss(data, target, model, pr, rho_1, scale, seq_len, args_batch_size):
    total_L = mean_L = 0.0
    hidden = model.begin_state(func = mx.nd.zeros, batch_size = args_batch_size, ctx=context)
    
    output, hidden = model(data, hidden)
    L = loss(output, target)
    total_L = (mx.nd.sum(L).asscalar())*scale + rho_1*0.5*pr
    mean_L = total_L / 29049 / args_batch_size
    
    return total_L, mean_L



def train_eval_ppl(data, target, model, seq_len, args_batch_size):
    total_L = 0.0
    hidden = model.begin_state(func = mx.nd.zeros, batch_size = args_batch_size, ctx=context)
    
    output, hidden = model(data, hidden)
    L = loss(output, target) 
    total_L = mx.nd.sum(L).asscalar()
    cur_L = total_L/seq_len/args_batch_size
    
    return cur_L



def test_eval_loss(data, target, model, pr, rho_1, scale_test, seq_len_test, args_batch_size):
    total_L = mean_L = 0.0
    hidden = model.begin_state(func = mx.nd.zeros, batch_size = args_batch_size, ctx=context)
    
    output, hidden = model(data, hidden)
    L = loss(output, target)
    total_L = (mx.nd.sum(L).asscalar())*scale_test + rho_1*0.5*pr
    mean_L = total_L / 2575 / args_batch_size
    
    return total_L, mean_L
    
    

def test_eval_ppl(data, target, model):
    total_L = 0.0
    hidden = model.begin_state(func = mx.nd.zeros, batch_size = args_batch_size, ctx=context)
    
    output, hidden = model(data, hidden)
    L = loss(output, target) 
    total_L = mx.nd.sum(L).asscalar()
    return total_L 

