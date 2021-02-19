import math
import os
import numpy as np
import pandas as pd
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn, rnn
import matplotlib.pyplot as plt



context = mx.gpu() 
corpus = Corpus(args_data)



def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [i.detach() for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden



def sparsify_params(params, weights, param_struct, sparse_emb, sparse_h2h1, sparse_h2h2):

	if sparse_emb == 1:
	    #W_emb
	    param = params[0]
	    weight1 = weights[0].asnumpy()
	    param.set_data(weight1*param_struct['W_emb'])
	    params[0] = param

	if sparse_h2h1 == 1:
	    #W_h2h1
	    param = params[2]
	    weight1 = weights[2].asnumpy()
	    param.set_data(weight1*param_struct['W_h2h1'])
	    params[2] = param

	if sparse_h2h2 == 1:
	    #W_h2h2
	    param = params[6]
	    weight1 = weights[6].asnumpy()
	    param.set_data(weight1*param_struct['W_h2h2'])
	    params[6] = param
	   
    return params



def update_model(model, params):
	
    for k,v in model.collect_params().items():
        if 'embedding0_weight' in k:
            v = params[0]
        
        elif 'lstm0_l0_i2h_weight' in k:
            v = params[1]
        
        elif 'lstm0_l0_h2h_weight' in k:
            v = params[2]
        
        elif 'lstm0_l0_i2h_bias' in k:
            v = params[3]
        
        elif 'lstm0_l0_h2h_bias' in k:
            v = params[4]
        
        elif 'lstm0_l1_i2h_weight' in k:
            v = params[5]
        
        elif 'lstm0_l1_h2h_weight' in k:
            v = params[6]

        elif 'lstm0_l1_i2h_bias' in k:
            v = params[7]

        elif 'lstm0_l1_h2h_bias' in k:
            v = params[8]

        elif 'dense0_weight' in k:
            v = params[9]

        elif 'dense0_bias' in k:
            v = params[10]     
    return model



def update_structure(weights, param_struct, grads, sel_emb, scale, rho_0, rho_1, const_q, sparse_emb, sparse_h2h1, sparse_h2h2):

	if sparse_emb == 1:
	    #W_emb
	    weight = weights[0]
	    weight_np = weight.asnumpy()
	    weight1 = np.reshape(weight_np,-1)[sel_emb]
	    
	    grad = grads[0]
	    grad_np = grad.asnumpy() 
	    grad1 = np.reshape(grad_np,-1)[sel_emb]
	    
	    G = scale * grad1
	    G_sq = G**2
	    zz1 = 0.5 * (rho_1-rho_0) * (weight1**2)
	    zz2 = -weight1 * G
	    zz3 = -0.5 * (weight1**2) * G_sq
	    zz = zz1 + zz2 + zz3
	    
	    prob = 1/(1 + np.exp(const_q + zz))
	    # print(prob)
	    a = np.random.uniform(0,1,sel_emb.shape[0]) <= prob
	    np.reshape(param_struct['W_emb'],-1)[sel_emb] = [int(elem) for elem in a]
	
	if sparse_h2h1 == 1:
	    #W_h2h1
	    weight = weights[2]
	    weight_np = weight.asnumpy()
	    weight1 = np.reshape(weight_np,-1)[sel_h2h1]
	    grad = grads[2]
	    grad_np = grad.asnumpy() 
	    grad1 = np.reshape(grad_np,-1)[sel_h2h1]
	    
	    G = scale * grad1
	    G_sq = G**2
	    zz1 = 0.5 * (rho_1-rho_0) * (weight1**2)
	    zz2 = -weight1 * G
	    zz3 = -0.5 * (weight1**2) * G_sq
	    zz = zz1 + zz2 + zz3
	    
	    prob = 1/(1 + np.exp(const_q + zz))
	    a = np.random.uniform(0,1,sel_h2h1.shape[0]) <= prob
	    np.reshape(param_struct['W_h2h1'],-1)[sel_h2h1] = [int(elem) for elem in a]
	    
    if sparse_h2h2 == 1:
	    #W_h2h2
	    weight = weights[6]
	    weight_np = weight.asnumpy()
	    weight1 = np.reshape(weight_np,-1)[sel_h2h2]
	    grad = grads[6]
	    grad_np = grad.asnumpy() 
	    grad1 = np.reshape(grad_np,-1)[sel_h2h2]
	    
	    G = scale * grad1
	    G_sq = G**2
	    zz1 = 0.5 * (rho_1-rho_0) * (weight1**2)
	    zz2 = -weight1 * G
	    zz3 = -0.5 * (weight1**2) * G_sq
	    zz = zz1 + zz2 + zz3
	    
	    prob = 1/(1 + np.exp(const_q + zz))
	    a = np.random.uniform(0,1,sel_h2h2.shape[0]) <= prob
	    np.reshape(param_struct['W_h2h2'],-1)[sel_h2h2] = [int(elem) for elem in a]

    return param_struct
    


def update_params(params, param_struct, weights, grads, gamma, rho_0, rho_1, scale, sparse_emb, sparse_h2h1, sparse_h2h2):
    
    #W_emb 
    param = params[0]
    weight = weights[0]
    weight1 = weight.asnumpy()
    grad = grads[0]
	grad_np = grad.asnumpy() 

    if sparse_emb == 1:
	    L_0 = np.sum(param_struct['W_emb']==0)
	    if L_0 > 0:
	        weight1[param_struct['W_emb']==0] = np.random.normal(0, np.sqrt(1/rho_0), L_0)

	    L_1 = np.sum(param_struct['W_emb']==1)
	    if L_1 > 0:
	        sub_grad = grad_np[param_struct['W_emb']==1]
	        sub_grad = rho_1*weight1[param_struct['W_emb']==1] + scale*sub_grad 
	        sub_weights = weight1[param_struct['W_emb']==1] - gamma*sub_grad + np.sqrt(2*gamma)*np.random.normal(0,1,L_1)
	        weight1[param_struct['W_emb']==1] = sub_weights 

	    param.set_data(weight1)
	    params[0] = param
	
	else:
		grad1 = rho_1*weight1 + scale*grad_np
	    weight1 = weight1 - gamma*grad1 + np.sqrt(2*gamma)*np.random.normal(0.0, 1.0, size = (weight1.shape[0],weight1.shape[1]))
	    param.set_data(weight1)
	    params[1] = param



    #W_i2h1
    param = params[1]
    weight = weights[1]
    weight1 = weight.asnumpy()
    grad = grads[1]
    grad_np = grad.asnumpy() 
    
    
    grad1 = rho_1*weight1 + scale*grad_np
    weight1 = weight1 - gamma*grad1 + np.sqrt(2*gamma)*np.random.normal(0.0, 1.0, size = (weight1.shape[0],weight1.shape[1]))
    param.set_data(weight1)
    params[1] = param


    #W_h2h1
    
    param = params[2]
    weight = weights[2]
    weight1 = weight.asnumpy()
    grad = grads[2]
    grad_np = grad.asnumpy() 

    if sparse_h2h1 == 1:
	    L_0 = np.sum(param_struct['W_h2h1']==0)
	    if L_0 > 0:
	        weight1[param_struct['W_h2h1']==0] = np.random.normal(0, np.sqrt(1/rho_0), L_0)

	    L_1 = np.sum(param_struct['W_h2h1']==1)
	   
	    if L_1 > 0:
	        sub_grad = grad_np[param_struct['W_h2h1']==1]
	        sub_grad = rho_1*weight1[param_struct['W_h2h1']==1] + scale*sub_grad 
	        sub_weights = weight1[param_struct['W_h2h1']==1] - gamma*sub_grad + np.sqrt(2*gamma)*np.random.normal(0,1,L_1)
	        weight1[param_struct['W_h2h1']==1] = sub_weights 

	    param.set_data(weight1)
	    params[2] = param
    
   else:
	    grad1 = rho_1*weight1 + scale*grad_np
	    weight1 = weight1 - gamma*grad1 + np.sqrt(2*gamma)*np.random.normal(0.0, 1.0, size = (weight1.shape[0],weight1.shape[1]))
	    param.set_data(weight1)
	    params[2] = param


    #update b_i2h1
    param = params[3]
    weight = weights[3]
    weight1 = weight.asnumpy()
    grad = grads[3]
    grad_np = grad.asnumpy() 
    
    grad1 = rho_1*weight1 + scale*grad_np
    weight1 = weight1 - gamma*grad1 + np.sqrt(2*gamma)*np.random.normal(0.0,1.0, size = (weight1.shape[0], ))
    param.set_data(weight1)
    params[3] = param


    #update b_h2h1
    param = params[4]
    weight = weights[4]
    weight1 = weight.asnumpy()
    grad = grads[4]
    grad_np = grad.asnumpy() 
    
    grad1 = rho_1*weight1 + scale*grad_np
    weight1 = weight1 - gamma*grad1 + np.sqrt(2*gamma)*np.random.normal(0.0,1.0, size = (weight1.shape[0],))
    param.set_data(weight1)
    params[4] = param

    
    #update W_i2h2
    param = params[5]
    weight = weights[5]
    weight1 = weight.asnumpy()
    grad = grads[5]
    grad_np = grad.asnumpy() 
    
    grad1 = rho_1*weight1 + scale*grad_np
    weight1 = weight1 - gamma*grad1 + np.sqrt(2*gamma)*np.random.normal(0.0,1.0, size = (weight1.shape[0],weight1.shape[1]))
    param.set_data(weight1)
    params[5] = param
    
    
    #W_h2h2
    param = params[6]
    weight = weights[6]
    weight1 = weight.asnumpy()
    grad = grads[6]
    grad_np = grad.asnumpy() 

    if sparse_h2h2 == 1:
	    L_0 = np.sum(param_struct['W_h2h2']==0)
	    if L_0 > 0:
	        weight1[param_struct['W_h2h2']==0] = np.random.normal(0, np.sqrt(1/rho_0), L_0)

	    L_1 = np.sum(param_struct['W_h2h2']==1)
	    grad = grads[6]
	    grad_np = grad.asnumpy() 
	    if L_1 > 0:
	        sub_grad = grad_np[param_struct['W_h2h2']==1]
	        sub_grad = rho_1*weight1[param_struct['W_h2h2']==1] + scale*sub_grad 
	        sub_weights = weight1[param_struct['W_h2h2']==1] - gamma*sub_grad + np.sqrt(2*gamma)*np.random.normal(0,1,L_1)
	        weight1[param_struct['W_h2h2']==1] = sub_weights 

	    param.set_data(weight1)
	    params[6] = param
    
    else:
	    grad1 = rho_1*weight1 + scale*grad_np
	    weight1 = weight1 - gamma*grad1 + np.sqrt(2*gamma)*np.random.normal(0.0,1.0, size = (weight1.shape[0],weight1.shape[1]))
	    param.set_data(weight1)
	    params[6] = param


    #update b_i2h2
    param = params[7]
    weight = weights[7]
    weight1 = weight.asnumpy()
    grad = grads[7]
    grad_np = grad.asnumpy() 
    
    grad1 = rho_1*weight1 + scale*grad_np
    weight1 = weight1 - gamma*grad1 + np.sqrt(2*gamma)*np.random.normal(0.0,1.0, size = (weight1.shape[0],))
    param.set_data(weight1)
    params[7] = param


    #update b_h2h2
    param = params[8]
    weight = weights[8]
    weight1 = weight.asnumpy()
    grad = grads[8]
    grad_np = grad.asnumpy() 
    
    grad1 = rho_1*weight1 + scale*grad_np
    weight1 = weight1 - gamma*grad1 + np.sqrt(2*gamma)*np.random.normal(0.0,1.0, size = (weight1.shape[0],))
    param.set_data(weight1)
    params[8] = param


    #update W_dense
    param = params[9]
    weight = weights[9]
    weight1 = weight.asnumpy()
    grad = grads[9]
    grad_np = grad.asnumpy() 
    
    grad1 = rho_1*weight1 + scale*grad_np
    weight1 = weight1 - gamma*grad1 + np.sqrt(2*gamma)*np.random.normal(0.0,1.0, size = (weight1.shape[0],weight1.shape[1]))
    param.set_data(weight1)
    params[9] = param


    #update b_dense
    param = params[10]
    weight = weights[10]
    weight1 = weight.asnumpy()
    grad = grads[10]
    grad_np = grad.asnumpy() 
    
    grad1 = rho_1*weight1 + scale*grad_np
    weight1 = weight1 - gamma*grad1 + np.sqrt(2*gamma)*np.random.normal(0.0,1.0, size = (weight1.shape[0],))
    param.set_data(weight1)
    params[10] = param
    
    return params



def grad_loss(model, data, target, hidden):
    with autograd.record():
        output, hidden = model(data, hidden)
        L = loss(output, target)
        L.backward()
    grads = [i.grad(context) for i in model.collect_params().values()]
    return grads, L
    


def prior(weights_pred):
    pr = 0.0
    
    #W_emd
    weight0 = weights_pred[0].asnumpy()
    pr += np.sum(weight0**2)
    
    #W_i2h1
    weight1 = weights_pred[1].asnumpy()
    pr += np.sum(weight1**2)
    
    #W_h2h1
    weight2 = weights_pred[2].asnumpy()
    pr += np.sum(weight2**2)
    
    #b_i2h1
    weight3 = weights_pred[3].asnumpy()
    pr += np.sum(weight3**2)
    
    #b_h2h1
    weight4 = weights_pred[4].asnumpy()
    pr += np.sum(weight4**2)
    
    #W_i2h2
    weight5 = weights_pred[5].asnumpy()
    pr += np.sum(weight5**2)
    
    #W_h2h2
    weight6 = weights_pred[6].asnumpy()
    pr += np.sum(weight6**2)
    
    #b_i2h2
    weight7 = weights_pred[7].asnumpy()
    pr += np.sum(weight7**2)
    
    #b_h2h2
    weight8 = weights_pred[8].asnumpy()
    pr += np.sum(weight8**2)
    
    #W_dense
    weight9 = weights_pred[9].asnumpy()
    pr += np.sum(weight9**2)
    
    #b_dense
    weight10 = weights_pred[10].asnumpy()
    pr += np.sum(weight10**2)
    
    return pr



