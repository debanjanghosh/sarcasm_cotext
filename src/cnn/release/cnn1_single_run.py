"""
Sample code for
Convolutional Neural Networks for Sentence Classification
http://arxiv.org/pdf/1408.5882v2.pdf

Much of the code is modified from
- deeplearning.net (for ConvNet classes)
- https://github.com/mdenil/dropout (for dropout)
- https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)

Much of the code is modified from
Kim's CNN implementation
"""


import random
import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
import re
import warnings
import sys
import pandas as pd
import logging
import math
from conv_net_classes import LeNetConvPoolLayer
from conv_net_classes import MLPDropout
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support as score
from mercurial.wireproto import batch


warnings.filterwarnings("ignore")   

logger = logging.getLogger('myapp')
hdlr = logging.FileHandler('log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.WARNING)
time_stamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
log_file = open("./data/output/logs/log_file_{}".format(time_stamp), "w+")


#different non-linearities
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
def Tanh(x):
    y = T.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)
       

'''
main function to build a training/test model for the 2-CNN
'''

def build_model(U,
                img_h1,
                img_w=110, 
                x1_filter_hs=[1,2,3],
                hidden_units=[110,2], 
                dropout_rate=0.5,
                batch_size=50, 
                conv_non_linear="relu",
                activation=Iden,
                sqr_norm_lim=9,
                non_static=True):

    rng = np.random.RandomState(3435)
    filter_w = img_w
    feature_maps = hidden_units[0]
    x1_filter_shapes = []
    pool_x1_sizes = []
    
    ''' for different filters set the pools for both CNN'''
    ''' (note - this code creates the structures in a way to be handled easily
        in the lenet code) '''
    
    for filter_h in x1_filter_hs:
        x1_filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_x1_sizes.append((img_h1-filter_h+1, img_w-filter_w+1))
    
    parameters = [("image x1 shape",img_h1,img_w), ("x1 filter shape",x1_filter_shapes), ("pool x1 size", pool_x1_sizes), ("hidden_units",hidden_units),
                  ("dropout", dropout_rate), ("batch_size",batch_size),("non_static", non_static),
                    ("conv_non_linear", conv_non_linear), 
                    ("sqr_norm_lim",sqr_norm_lim)]
    
    print parameters
    
    logger.error("start")
    logger.error('Records: %s', parameters)

    
    #define model architecture
    x1 = T.imatrix('x1')
    y = T.ivector('y')
    
    Words = theano.shared(value = U, name = "Words")
    
    '''for the first layer input'''
    
    layer0_x1_input = Words[x1.flatten()].reshape((x1.shape[0],1,x1.shape[1],Words.shape[1]))
    
    conv_layers = []
    conv_layers1 = []

    x1_layer1_inputs = []
    
    '''creating two LeNetConvPoolLayers - for both CNN'''
    
    for i in xrange(len(x1_filter_hs)):
        x1_conv_layer = LeNetConvPoolLayer(rng, input=layer0_x1_input,image_shape=(batch_size, 1, img_h1, img_w),
                                filter_shape=x1_filter_shapes[i], poolsize=pool_x1_sizes[i], non_linear=conv_non_linear)
        x1_layer1_input = x1_conv_layer.output.flatten(2)
        conv_layers1.append(x1_conv_layer)
        x1_layer1_inputs.append(x1_layer1_input)
    
    
    layer1_input = T.concatenate(x1_layer1_inputs, 1)
    hidden_units[0] = feature_maps * ( len(x1_filter_hs))
  #  conv_layers = conv_layers1 + conv_layers2
    
    #TODO - instead of concat, try another function to combine the layers? 
    
    #x1_layer1_input = T.concatenate(x1_layer1_inputs,1)
    #x2_layer1_input = T.concatenate(x2_layer1_inputs,1)
    #outer_prod = x1_layer1_input.dimshuffle(0,1,'x') * x2_layer1_input.dimshuffle(0,'x',1)
    #layer1_input = outer_prod.flatten(2)
    #hidden_units[0] = feature_maps*len(x1_filter_hs) * feature_maps*len(x2_filter_hs)
    
    #layer1_input = x1_layer1_input * x2_layer1_input
    classifier = MLPDropout(rng, input=layer1_input, layer_sizes=hidden_units, activations=[activation], dropout_rates=[dropout_rate])
    
    return x1, y, Words, conv_layers1, classifier

def shared_dataset(data, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x1, data_y = data
    shared_x1 = theano.shared(np.asarray(data_x1,
                                               dtype='int32'),
                                 borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y,
                                               dtype='int32'),
                                 borrow=borrow)
    return shared_x1, shared_y

''''''
        
def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9,word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        '''
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param
        '''
        updates[param] = stepped_param
    return updates 

def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)
    
def word_2_index(word, word_idx_map):
    if word in word_idx_map:
        return word_idx_map[word]
    else:
        return 0


'''
indexing the sentences into a list of indices based on the words

'''

def index_x2(sent, word_idx_map, max_l=30, filter_h=3):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    ''' based on the length of filter we add pad. for a filter = 3, the first ngram would be <pad> <pad> I'''
    for i in xrange(pad):
        x.append(0)
        
    '''we are using split (no NLTK since tweets are already tokenized using CMU tool)'''    
    words = sent.split()
    for word in words :
        
        if len(x) >= max_l+2*pad :
            break
        
        if word in word_idx_map:
            x.append(word_idx_map[word])
        else:
            x.append(1) # unknown word
    while len(x) < max_l+2*pad:
        x.append(0)
    
    return x
'''
sentence * words into matrix format transformation
'''

def make_idx_data(data, word_idx_map, max_x1, x1_filter_h):
    """
    Transforms sentences into a 2-d matrix.
    """
    x1 = []
    y = []
    for d in data:
        idx_x1 = index_x2(d["text"], word_idx_map,max_x1, x1_filter_h)
        x1.append(idx_x1)
        y.append(d['y'])
        
    x1_data = np.array(x1,dtype="int")
    y_data = np.array(y,dtype="int")
    return [x1_data, y_data]


def getTest2Sets(test_data,batch_size):
    
    n_batches = int(math.ceil(test_data[0].shape[0]/float(batch_size)))
    n_test_batches = int(np.round(n_batches*1.))
    print 'n_batches: ', n_batches
  #  print 'n_train_batches: ', n_test_batches
    #for data in train_data:
    #    print data.shape 
    test_set = [data[:n_test_batches*batch_size] for data in test_data]
  #  val_set = [data[n_train_batches*batch_size:] for data in train_data]
    
    ''' need fixed # of data for each batch - so slight manipulation'''
    if test_set[0].shape[0] % batch_size > 0:
        extra_data_num = batch_size - test_set[0].shape[0] % batch_size
        new_set = [np.append(data, data[:extra_data_num],axis=0) for data in test_set]
        test_set = new_set
    
    print 'test size =', len(test_set[0])

    return test_set,n_test_batches


def getTrainValSets(train_data,batch_size):
    
    n_batches = int(math.ceil(train_data[0].shape[0]/float(batch_size)))
    n_train_batches = int(np.round(n_batches*0.9))
    print 'n_batches: ', n_batches
    print 'n_train_batches: ', n_train_batches
    #for data in train_data:
    #    print data.shape 
    train_set = [data[:n_train_batches*batch_size] for data in train_data]
    val_set = [data[n_train_batches*batch_size:] for data in train_data]
    
    ''' need fixed # of data for each batch - so slight manipulation'''
    if val_set[0].shape[0] % batch_size > 0:
        extra_data_num = batch_size - val_set[0].shape[0] % batch_size
        new_set = [np.append(data, data[:extra_data_num],axis=0) for data in val_set]
        val_set = new_set
    #print 'train size =', train_set.shape, ' val size =', val_set.shape 

    return train_set,val_set

'''main test function'''

def getTestSet(test_data,batch_size,img_h1,img_h2):

    ### 2. organize data
    minibatches = []
    minibatch_start = 0
 
    '''we still need some fake data for the final batch '''
    
    for i in range(test_data[0].shape[0] // batch_size):
        minibatches.append((minibatch_start,minibatch_start+batch_size))
        minibatch_start += batch_size
    if (minibatch_start != test_data[0].shape[0]):
        fake_size = batch_size + minibatch_start - test_data[0].shape[0]
        
        #CNN1
        target = test_data[0] 
        #CNN2
        situation = test_data[1]
        labels = test_data[2]
                              
        extra_data_target = target[0:fake_size]
        extra_data_situation = situation[0:fake_size]
        extra_data_labels = labels[0:fake_size]
        
       
        fake_data = np.zeros((fake_size, img_h1+1)),np.zeros((fake_size, img_h2+1))
        minibatches.append((minibatch_start,minibatch_start+batch_size))
        
        target = np.concatenate((target,extra_data_target),axis=0)
        situation = np.concatenate((situation,extra_data_situation),axis=0)
        labels = np.concatenate((labels,extra_data_labels),axis=0)

        test_data = [ target, situation, labels ]
        
    return test_data,minibatches
        
def train_test(train_data,test_data,model_file,pred_file,U):

    hidden_units=[100,2]
    batch_size=1000
    img_w=100
    dropout_rate=0.5
                
    n_epochs=25
    lr_decay = 0.95
    conv_non_linear="relu"
    activation=Iden
    sqr_norm_lim=9
    non_static=True            
    
  #  print 'hello'
    shuffle_batch=True
    x1_filter_hs = [1,2,3]
    batch_size = 25
    non_static = False
    hidden_units = [100,2]

    img_h1 = len(train_data[0][0])
    print train_data[0].shape
    print "img_h1=" + str(img_h1)
    
    x1 = T.imatrix('x1')
    y = T.ivector('y')
    
    Words = theano.shared(value = U, name = "Words")
    zero_vec_tensor = T.vector()
    zero_vec = np.zeros(img_w)
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0,:], zero_vec_tensor))], allow_input_downcast=True)

    x1, y, Words, conv_layers_1, classifier = build_model(
                U,
                img_h1,
                img_w=img_w, 
                x1_filter_hs=x1_filter_hs,
                hidden_units=hidden_units, 
                dropout_rate=dropout_rate,
                batch_size=batch_size, 
                conv_non_linear=conv_non_linear,
                activation=activation,
                sqr_norm_lim=sqr_norm_lim,
                non_static=non_static)
    index = T.lscalar()
        
         #define parameters of the model and update functions using adadelta
    params = classifier.params
    for conv_layer in conv_layers_1:
        params += conv_layer.params
        
    if non_static:
        #if word vectors are allowed to change, add them as model parameters
        params += [Words]

    cost = classifier.negative_log_likelihood(y)
    dropout_cost = classifier.dropout_negative_log_likelihood(y)
    grad_updates = sgd_updates_adadelta(params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)
    
    
    #shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate 
    #extra data (at random)
    # !! currently assume input is randomized !!
    
    np.random.seed(3435)

    n_batches = int(math.ceil(train_data[0].shape[0]/float(batch_size)))
    n_train_batches = int(np.round(n_batches*0.9))


    train_set,val_set = getTrainValSets(train_data,batch_size)
  #  test_set,minibatches = getTestSet(test_data,batch_size,img_h1,img_h2)
    
    test_set,n_test_batches = getTest2Sets(test_data,batch_size)

    train_set_x1, train_set_y = shared_dataset(train_set)
    val_set_x1, val_set_y = shared_dataset(val_set)    
    test_set_shared_x1, test_set_shared_y = shared_dataset(test_set)
    test_set_x1, test_set_y = test_set
   
    n_val_batches = n_batches - n_train_batches
    val_model = theano.function([index], classifier.errors(y),
                                givens={
                                        x1: val_set_x1[index * batch_size: (index + 1) * batch_size],
                                        y: val_set_y[index * batch_size: (index + 1) * batch_size]})

    train_model = theano.function([index], cost, updates=grad_updates,
                                  givens={
                                          x1: train_set_x1[index * batch_size: (index + 1) * batch_size],
                                          y: train_set_y[index * batch_size: (index + 1) * batch_size]})
    

    s_idx = T.iscalar()
    e_idx = T.iscalar()
    
   
    test_model = theano.function([index], classifier.errors(y),
                                givens={
                                        x1: test_set_shared_x1[index * batch_size: (index + 1) * batch_size],
                                        y: test_set_shared_y[index * batch_size: (index + 1) * batch_size]})


    test_pred_layers_x1 = []

    test_size = test_set_x1.shape[0]
  #  test_layer0_input = U[T.cast(x.flatten(),dtype="int32")].reshape((test_size,1,img_h,U.shape[1]))
  #  test_layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((test_size,1,img_h,Words.shape[1]))

  #  test_layer0_input_x1 = Words[x1.flatten()].reshape((x1.shape[0],1,x1.shape[1],Words.shape[1]))
  #  test_layer0_input_x2 = Words[x2.flatten()].reshape((x2.shape[0],1,x2.shape[1],Words.shape[1]))

    test_layer0_input_x1 = Words[T.cast(x1.flatten(),dtype="int32")].reshape((test_size,1,img_h1,Words.shape[1]))
    for i in xrange(len(conv_layers_1)):
        conv_layer_1 = conv_layers_1[i]
        test_layer0_output_x1 = conv_layer_1.predict(test_layer0_input_x1, test_size)
        test_pred_layers_x1.append(test_layer0_output_x1.flatten(2))
    
    test_layer1_input = T.concatenate(test_pred_layers_x1, 1)
    
    '''
    for i, conv_layer_1 in enumerate(conv_layers_1):
        conv_layer_2 = conv_layers_2[i]
        
        test_layer0_input_x1 = Words[x1.flatten()].reshape((x1.shape[0],1,x1.shape[1],Words.shape[1]))
        test_layer0_input_x2 = Words[x2.flatten()].reshape((x2.shape[0],1,x2.shape[1],Words.shape[1]))

        test_layer0_output_x1 = conv_layer_1.predict(test_layer0_input_x1, test_size)
        test_layer0_output_x2 = conv_layer_2.predict(test_layer0_input_x2, test_size)
            
        test_layer0_output_x1+=test_layer0_output_x2
            
        test_pred_layers.append(test_layer0_output_x1.flatten(2))
    '''    
    test_y_pred = classifier.predict(test_layer1_input)
    test_error = T.mean(T.neq(test_y_pred, y))
    test_model_all = theano.function([x1,y], test_error, allow_input_downcast = True)   

    print 'start training....'
    
    epoch = 0
    best_val_perf = 0
    val_perf = 0
    cost_epoch = 0    
    
    
    while (epoch < n_epochs):
        epoch = epoch + 1
        if shuffle_batch:
            j = 1
            for minibatch_index in np.random.permutation(range(n_train_batches)):
                cost_epoch = train_model(minibatch_index)
                set_zero(zero_vec)

        else:
            for minibatch_index in xrange(n_train_batches):
                cost_epoch = train_model(minibatch_index)  
                set_zero(zero_vec)
                
     #   train_losses = [test_model(i) for i in xrange(n_train_batches)]
     #   train_perf = 1 - np.mean(train_losses)
        val_losses = [val_model(i) for i in xrange(n_val_batches)]
        val_perf = 1- np.mean(val_losses)
        print('epoch %i, val perf %f %%' % (epoch, val_perf*100.))
        logger.error('epoch %i, val perf %f %%' % (epoch, val_perf*100.))
        
        if val_perf >= best_val_perf:
            best_val_perf = val_perf
            
            test_losses = test_model_all(test_set_x1, test_set_y)        
            test_perf = 1- test_losses

            ''' check the next two lines'''
#            test_losses = [test_model(i) for i in xrange(n_test_batches)]
#            test_perf = 1- np.mean(test_losses)
 
       #     test_preds = [test_result(minibatches[i][0], minibatches[i][1]) for i in xrange(len(minibatches))]

#            test_loss = test_model(test_set_x,test_set_y)        
#            losses = [test_result(minibatches[i][0], minibatches[i][1]) for i in xrange(len(minibatches))-1]
           # losses = [test_result(minibatches[0][0], minibatches[0][1])]# for i in xrange(len(minibatches))-1]

            #total_test_losses = 0
            #for i in xrange(len(losses)):
             #   total_test_losses += losses[i] * (minibatches[i][1]-minibatches[i][0])
    
            #test_error = T.mean(T.neq(test_y_pred, y))
           # total_test_losses /= test_data[0].shape[0]
           # test_perf = 1- total_test_losses
    
            print('test size ' + str(test_data[0].shape[0]))
            print('test perf %f %%' % (test_perf * 100.))
            logger.error('test perf %f %%' % (test_perf * 100.))

    '''the following is for a control on measuring P/R/F1 (avg, SD) via external code'''
    '''
    test_preds = [test_result(minibatches[i][0], minibatches[i][1]) for i in xrange(len(minibatches))]
    with open(output_file, 'wb') as fout:
        for pred in test_preds:
            for p in pred:
                fout.write(str(p) + '\n')
    '''

    
def getTrainTestData(target,path):

    path ='/Users/dg513/work/eclipse-workspace/compose-workspace/CompositionModels/data/output/wsd/pkl/1_cnn/'
    
    train_file = path + 'tweet.' + target + '.target.TRAIN' + '.pkl'
    test_file =  path + 'tweet.' + target + '.target.TEST' + '.pkl'
    
    model_file = path + 'tweet.' + target + '.target.TRAIN' + '.model.pkl'
    output_file = path + target + '.1cnn.pred'

    batch_size = 25
    
    '''non_static parameter handles the static/dynamic nature of embedding. 
        whether we need to learn task/specific embedding or not'''
    non_static = True
    
    print "loading data...",
    logger.error("loading data...");
    

    x = cPickle.load(open(train_file,"rb"))
    train_data, W, word_idx_map, max_x1 = x[0], x[1], x[2], x[3]
    print 'size=', len(W), len(W[0])
    img_w = len(W[0])
    print "data loaded!"
    U = W.astype(theano.config.floatX)
    
    test_data = cPickle.load(open(test_file,'rb'))

    print "max x1 length = " + str(max_x1)
    
    x1_filter_hs = [1,2,3]
    
    '''this is coming from the original LENET'''
    execfile("./src/com/rutgers/cnn/conv_net_classes.py")
    
    '''this converts the training data in format for CNN'''
    idx_train_data = make_idx_data(train_data, word_idx_map, max_x1, max(x1_filter_hs))
    idx_test_data = make_idx_data(test_data, word_idx_map, max_x1, max(x1_filter_hs))
    
    return idx_train_data,idx_test_data,model_file,output_file,U

def listTargets():
    file = open('./data/config/targets.txt')
    targets = [ line.strip() for line in file.readlines()]
    return targets

''' path is the input for .pkl files (resulted after preprocessing)
   hidden_units = # of hidden nodes
   targets = target words for which we are running all the experiments 
   [for "targets" check EMNLP2015 paper] 
'''

def main(args):

    path  = args[0]
    hidden_units = int(args[1])
    targets = listTargets()
    targets = ['always']

    ''' fix the parameter on one target '''
    random.shuffle(targets)
    random_target = targets[0]

    '''drop_outs,epsilons, epochs are fixed after running on the dev data'''
    drop_outs = [0.5]
    hidden_dims = [hidden_units]
    epsilons = [1e-5]
    epochs = [2]

    for target in targets:
        'selected target is ' + target

        for drop_out in drop_outs:
                for hidden_dim in hidden_dims:
                        for epsilon in epsilons:
                                for epoch in epochs:
                                    
                                    train_data,test_data,model_file,pred_file,U = getTrainTestData(target,path)
                                    train_test(train_data,test_data,model_file,pred_file,U)
                                      #  test(params,target,path,hidden_dim,non_static=True,both=False)



    log_file.close()

    
if __name__=="__main__":
    main(sys.argv[1:])



        
