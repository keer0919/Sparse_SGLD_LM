import math
import os
import numpy as np
import pandas as pd
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn, rnn
import sklearn
from sklearn.utils import shuffle


from language_model import get_batch, detach, update_model, update_structure, update_params, sparsify_params, grad_loss, prior
from eval import train_eval_loss, train_eval_ppl, test_eval_loss, test_eval_ppl
from prediction import params_ave, get_ave_param


args_data = "./data/ptb."
args_emsize = 100
args_nhid = 200
args_nlayers = 2
args_clip = 0.5
args_batch_size = 32
args_bptt = 100


#define shape of parameters
vocab_size = ntokens = len(corpus.dictionary)
#embedding layer shape
shape1, shape2 = vocab_size, args_emsize 

#i2h1 shape
shape3, shape4 = args_nhid*4, args_emsize 
#h2h1 shape
shape5, shape6 = args_nhid*4, args_nhid
#i2h1 bias shape
shape7 = args_nhid*4
#h2h1 bias shape
shape8 = args_nhid*4

#i2h2 shape
shape9, shape10 = args_nhid*4, args_nhid
#h2h2 shape
shape11, shape12 = args_nhid*4, args_nhid
#i2h2 bias shape
shape13 = args_nhid*4
#h2h2 bias shape
shape14 = args_nhid*4

#dense weight shape
shape15, shape16 = vocab_size, args_nhid
#dense bias shape
shape17 = vocab_size

p = shape1*shape2+shape3*shape4+shape5*shape6+shape7+shape8+shape9*shape10+shape11*shape12+shape13+shape14+shape15*shape16+shape17
sub_p1 = shape3*shape4 + shape7 + shape8 + shape9*shape10 + shape13 + shape14 + shape15*shape16 + shape17 

sparse_emb = 1
sparse_h2h1 = sparse_h2h2 = 0


#prior parameters
rho_0 = 100
rho_1 = 1
u_0 = 0.5
gamma_0 = 5e-7
decay = 0.001
scale = 29049/100
scale_test = 2575/100
const_q = (u_0 + 1) * np.log(p) + 0.5 * np.log(rho_0/rho_1)



#Initialize the averaged parameters
emd_ave = np.zeros((shape1, shape2))
W_i2h1_ave = np.zeros((shape3, shape4))
W_h2h1_ave = np.zeros((shape5, shape6))
b_i2h1_ave = np.zeros((shape7, ))
b_h2h1_ave = np.zeros((shape8, ))
W_i2h2_ave = np.zeros((shape9, shape10))
W_h2h2_ave = np.zeros((shape11, shape12))
b_i2h2_ave = np.zeros((shape13, ))
b_h2h2_ave = np.zeros((shape14, ))
W_dense_ave = np.zeros((shape15, shape16))
b_dense_ave = np.zeros((shape17, ))


n_iter = 40000
n_samples = int(n_iter/50)
res_train = np.zeros((n_iter, 6))
res_test = np.zeros((n_samples, 5))



def psub(sparse_emb, sparse_h2h1, sparse_h2h2, shape5, shape6, shape11, shape12, shape1, shape2):

	if sparse_emb + sparse_h2h1 + sparse_h2h2 == 1:
		sub_p = sub_p1 + shape5*shape6 + shape11*shape12

	if sparse_emb + sparse_h2h1 + sparse_h2h2 == 2:
		sub_p = sub_p1 + shape1*shape2

	if sparse_emb + sparse_h2h1 + sparse_h2h2 == 3:
		sub_p = sub_p1 

	return sub_p



#define sparsity structure
def init_param_struct(sparse_emb, sparse_h2h1, sparse_h2h2):

	if sparse_emb + sparse_h2h1 + sparse_h2h2 == 3:
		param_struct = {
	    'W_emb' : np.ones(shape = (shape1, shape2)),
	    'W_h2h1' : np.ones(shape = (shape5, shape6)),
	    'W_h2h2' : np.ones(shape = (shape11, shape12)),
	    }

	if sparse_emb + sparse_h2h1 + sparse_h2h2 == 2:
		param_struct = {
	    'W_h2h1' : np.ones(shape = (shape5, shape6)),
	    'W_h2h2' : np.ones(shape = (shape11, shape12)),
	    }

	if sparse_emb + sparse_h2h1 + sparse_h2h2 == 1:
		param_struct = {
	    'W_emb' : np.ones(shape = (shape1, shape2)),
	    }

	return param_struct



class RNNModel(gluon.Block):
    """
    A LSTM model with an embedding layer, recurrent layer, and a desen layer.

    """
    def __init__(self, vocab_size, num_embed, num_hidden, num_layers, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        
        with self.name_scope():
            self.encoder = nn.Embedding(vocab_size, num_embed, weight_initializer = mx.init.Uniform(0.1))
            self.rnn = rnn.LSTM(num_hidden, num_layers, input_size=num_embed)
            self.decoder = nn.Dense(vocab_size, in_units = num_hidden, weight_initializer = mx.init.Uniform(0.1))
            self.num_hidden = num_hidden
        

    def forward(self, inputs, hidden):
        emb = self.encoder(inputs)
        output, hidden = self.rnn(emb, hidden)
        decoded = self.decoder(output.reshape((-1, self.num_hidden)))
        return decoded, hidden

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)



#Initialize model
model = RNNModel(ntokens, args_emsize, args_nhid, args_nlayers)
model.collect_params().initialize(mx.init.Xavier(), ctx=context)
file_name = "ssgld.params"
model.save_parameters(file_name)

model_pred = RNNModel(ntokens, args_emsize, args_nhid, args_nlayers)
model_pred.collect_params().initialize(mx.init.Xavier(), ctx=context)
model_pred.load_parameters(file_name, ctx=context)
file_name_pred = "ssgld_pred.params"
model_pred.save_parameters(file_name_pred)


loss = gluon.loss.SoftmaxCrossEntropyLoss()



#Randomly draw a minibatch from test set
pos = np.random.choice(2475, 1)                  
seq_len_test = 100
test = test_data[pos : pos + seq_len_test]
test_target = test_data[pos + 1 : pos + 1 + seq_len_test]
test_target = test_target.reshape((-1,)) 



def train():

	for epoch in range(n_iter):
    total_L = 0
    gamma = gamma_0/(1 + decay*epoch)
    
    if epoch == 0:
        train = train_data
    else:
        train = sklearn.utils.shuffle(train_data)    
    
    hidden = model.begin_state(func = mx.nd.zeros, batch_size = args_batch_size, ctx = context) 
    
    #Randomly draw a minibatch
    pos = np.random.choice(28949, 1)
    seq_len = args_bptt
    data = train[pos : pos + seq_len]
    target = train[pos + 1 : pos + 1 + seq_len]
    target = target.reshape((-1,))
    
    hidden = detach(hidden)
    
    #sparsity structure
    if sparse_emb == 1:
    	sel_emb = np.random.choice(a = shape1*shape2, size=100, replace=False)
    	np.reshape(param_struct['W_emb'],-1)[sel_emb] = 0
    
    if sparse_h2h1 == 1:
    	sel_h2h1 = np.random.choice(a = shape5*shape6, size=50, replace=False)
    	np.reshape(param_struct['W_h2h1'],-1)[sel_h2h1] = 0
    
    if sparse_h2h2 == 1:
		sel_h2h2 = np.random.choice(a = shape11*shape12, size=50, replace=False)
		np.reshape(param_struct['W_h2h2'],-1)[sel_h2h2] = 0
    
  
    params = [i for i in model.collect_params().values()]
    weights = [i.data() for i in model.collect_params().values()]
    model.save_parameters(file_name)
    

    params_sparse = sparsify_params(params, weights, param_struct)                 
    model_sparse = update_model(model, params_sparse)                                     
    grads_sparse, L_sparse = grad_loss(model_sparse, data, target, hidden)                              
    gluon.utils.clip_global_norm(grads_sparse, args_clip * args_bptt * args_batch_size)
    
    model.load_parameters(file_name, ctx=context)
    params = [i for i in model.collect_params().values()]
    weights = [i.data() for i in model.collect_params().values()]
  
    #update sparsity structure 
    param_struct = update_structure(weights, param_struct, grads_sparse, sel_emb, scale, rho_0, rho_1, const_q)  #new struct (100 positions)
    
    params_sparse = sparsify_params(params, weights, param_struct)
    model_sparse = update_model(model, params_sparse)
    weights_try = [i.data() for i in model_sparse.collect_params().values()]
    grads_sparse, L_sparse = grad_loss(model_sparse, data, target, hidden)
    gluon.utils.clip_global_norm(grads_sparse, args_clip * args_bptt * args_batch_size)
    
    model.load_parameters(file_name, ctx=context)
    params = [i for i in model.collect_params().values()]
    weights = [i.data() for i in model.collect_params().values()]

    #update parameters 
    params = update_params(params, param_struct, weights, grads_sparse, gamma, rho_0, rho_1, scale)
    model = update_model(model, params)
    weights_after = [i.data() for i in model.collect_params().values()]
    
    model.save_parameters(file_name)
    params = [i for i in model.collect_params().values()]
    weights = [i.data() for i in model.collect_params().values()]
 
    
    #predictive model
    params_pred = sparsify_params(params, weights, param_struct)  
    model_pred = update_model(model, params_pred)
    weights_pred = [i.data() for i in model_pred.collect_params().values()]
    
    pr_train = prior(weights_pred)
    train_total_L, train_mean_L = train_eval_loss(data, target, model_pred, pr_train, rho_1, scale, seq_len, args_batch_size)
    batch_L_train = train_eval_ppl(data, target, model, seq_len, args_batch_size)


    model.load_parameters(file_name, ctx=context)
    params = [i for i in model.collect_params().values()]
    weights = [i.data() for i in model.collect_params().values()]
    
    weight_temp = weights[0]
    
    #compute ratio of nonzero elements
    non_zero = 0
    if sparse_emb == 1:
    	num_emb = np.count_nonzero(param_struct['W_emb'])
    	emb_ratio = num_emb / (shape1*shape2)
    	non_zero = num_emb
    	res_train[epoch, 5] = emb_ratio

    if sparse_h2h1 == 1:
		num_h2h1 = np.count_nonzero(param_struct['W_h2h1'])
		h2h1_ratio = num_h2h1 / (shape5*shape6)
		non_zero += num_h2h1
		res_train[epoch, 6] = h2h1_ratio

	if sparse_h2h2 == 1:
		num_h2h2 = np.count_nonzero(param_struct['W_h2h2'])
		h2h2_ratio = num_h2h2 / (shape11*shape12)
		non_zero += num_h2h2
		res_train[epoch, 7] = h2h2_ratio
 
    ratio = (non_zero + sub_p)/p
    
    res_train[epoch, 0] = train_total_L
    res_train[epoch, 1] = train_mean_L
    res_train[epoch, 2] = pr_train
    res_train[epoch, 3] = ratio
    res_train[epoch, 4] = batch_L_train
    
    
    ssgld = pd.DataFrame(res_train)
    ssgld.to_csv("0215_ssgld2.csv")
    print('[Epoch %d] Training loss %.4f, training mean loss %.2f, training prior %.2f, non-zero ratio %.4f' % (epoch, train_total_L, train_mean_L, pr_train, ratio))
    print('nonzero of embedding %.4f' % (emb_ratio))
    
    model.save_parameters(file_name)
    
    #average sparsified params
    k = (epoch+1)%50
    params_pred = [i for i in model.collect_params().values()]
    weights_pred = [i.data() for i in model.collect_params().values()]
    
    if k != 0 and k > 10:
        emd_ave, W_i2h1_ave, W_h2h1_ave, b_i2h1_ave, b_h2h1_ave, W_i2h2_ave, W_h2h2_ave, b_i2h2_ave, b_h2h2_ave, W_dense_ave, b_dense_ave = params_ave(weights_pred, k, emd_ave, W_i2h1_ave, W_h2h1_ave, b_i2h1_ave, b_h2h1_ave, W_i2h2_ave, W_h2h2_ave, b_i2h2_ave, b_h2h2_ave, W_dense_ave, b_dense_ave)
        
    if k == 0:
        sample_id = int(np.floor(epoch / 50))
        
        params_average = get_ave_param(params_pred, weights_pred, emd_ave, W_i2h1_ave, W_h2h1_ave, b_i2h1_ave, b_h2h1_ave, W_i2h2_ave, W_h2h2_ave, b_i2h2_ave, b_h2h2_ave, W_dense_ave, b_dense_ave)
        model_pred = update_model(model_pred, params_average)
        weights_average = [i.data() for i in model_pred.collect_params().values()]
        
        pr_test = prior(weights_average)
        
        test_total_L, test_mean_L = test_eval_loss(test, test_target, model_pred, pr_test, rho_1, scale_test, seq_len_test, args_batch_size)
        total_L_test2 = test_eval_ppl(test, test_target, model_pred)
        batch_L_test2 = total_L_test2/ seq_len_test / args_batch_size
        res_test[sample_id, 0] = test_total_L
        res_test[sample_id, 1] = test_mean_L
        res_test[sample_id, 2] = pr_test
        res_test[sample_id, 3] = batch_L_test2
        res_test[sample_id, 4] = math.exp(batch_L_test2)

        ssgld_test = pd.DataFrame(res_test)
        ssgld_test.to_csv("0215_ssgld_test2.csv") 

        emd_ave = np.zeros((shape1, shape2))
        W_i2h1_ave = np.zeros((shape3, shape4))
        W_h2h1_ave = np.zeros((shape5, shape6))
        b_i2h1_ave = np.zeros((shape7, ))
        b_h2h1_ave = np.zeros((shape8, ))
        W_i2h2_ave = np.zeros((shape9, shape10))
        W_h2h2_ave = np.zeros((shape11, shape12))
        b_i2h2_ave = np.zeros((shape13, ))
        b_h2h2_ave = np.zeros((shape14, ))
        W_dense_ave = np.zeros((shape15, shape16))
        b_dense_ave = np.zeros((shape17, ))

        print('[Epoch %d] Test loss %.4f, test mean loss %.4f, test prior %.4f' % (epoch, test_total_L, test_mean_L, pr_test))
        print('[Epoch %d] Test ce loss %.4f, test perplexity %.4f' % (epoch, batch_L_test2, math.exp(batch_L_test2)))

        
        





