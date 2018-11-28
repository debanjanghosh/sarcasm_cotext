import numpy as np
import cPickle
from collections import defaultdict
import sys, re,os
#import pandas as pd
import math
from boto.dynamodb.condition import NULL
from utils import *

from collections import Counter
from collections import defaultdict

import ConfigParser



windowSize = 1
begin = 2

def range(index1,index2,total):
    
    ''' check if index2 in the beginning or the end of the utterance'''
    if index2 <= begin:
        if index1 <=begin:
            return True
        else:
            return False
    
    if index2 >= total-begin-1:
        if index1 >=total-begin-1:
            return True
        else:
            return False
    
    ''' index2 is somewhere in the middle. return +-1 '''
    if math.fabs(index1-index2) <=windowSize:
        return True
    else:
        return False


"""
The input file format is: 1st column is label (int), 2rd column is keyword, and 4nd column is query (string)
"""
def read_data_file(data_file,target, max_x1, max_x2, is_train):
    queries = []
    all_ids = []
    change_count = 0
    with open(data_file, "r") as fin:
        for line in fin:
            line = line.strip()
            [label, kw, _, text] = line.split('\t');
            text = text.lower()
            # delete the keywords from the text
            #text = re.sub(r'\s*\b%s\b\s*' % kw, ' ', text, 1).strip()

            #x2_words = text.split()
            #x1_words = kw.split()
            
            # we create an one word window for the hashtag
            kwPosn = -1
            words = text.split()
            for index, word in enumerate(words):
                if word.lower() == kw.lower() or word.lower() == '#' + kw.lower():
                    kwPosn = index
                    break
            if kwPosn == -1:
                print 'the target: ' + target + ' is absent - returning'
                continue
            
            x1_words = []
            x2_words = []
            for index, word in enumerate(words):
                ''' 2 heuristics - from both end, if word is in 3 words then it goes to x1'''
                ''' otherwise +-1 word goes to x1'''
                if range(index,kwPosn,len(words)):
                    x1_words.append(word)
                else:
                    x2_words.append(word)
             
            
            
            # update vocab only when it is training data
            '''
            if is_train == 1:
                for word in set(x1_words):
                    vocab[word] += 1
                for word in set(x2_words):
                    vocab[word] += 1
            '''
                    
            if len(x1_words) > max_x1:
                x1_words = x1_words[:max_x1]
                change_count += 1
            if len(x2_words) > max_x2:
                x2_words = x2_words[:max_x2]
                change_count += 1
            
            datum = {"y": int(label),
                    "x1": " ".join(x1_words),
                    "x2": " ".join(x2_words),
                    "x1_len": len(x1_words),
                    "x2_len": len(x2_words)}
            queries.append(datum)
            all_ids.append(label)
     
    print ("length more than %i: %i" % (max_x2, change_count)) 
    
    return queries,all_ids

"""
The input file format is: 1st column is label (int), 2rd column is keyword, and 4nd column is query (string)
"""
def read_data_file_notarget(data_file, max_x1, max_x2, is_train):
    queries = []
    all_ids = []
    change_count = 0
    with open(data_file, "r") as fin:
        for line in fin:
            line = line.strip()
            [label, kw, _, text] = line.split('\t');
            text = text.lower()
            # delete the keywords from the text
            #text = re.sub(r'\s*\b%s\b\s*' % kw, ' ', text, 1).strip()

            #x2_words = text.split()
            #x1_words = kw.split()
            
            # we create an one word window for the hashtag
            kwPosn = -1
            words = text.split()
            for index, word in enumerate(words):
                if word.lower() == kw.lower() or word.lower() == '#' + kw.lower():
                    kwPosn = index
                    break
            if kwPosn == -1:
                print 'the target: ' + target + ' is absent - returning'
                continue
            
            x1_words = []
            x2_words = []
            for index, word in enumerate(words):
                ''' 2 heuristics - from both end, if word is in 3 words then it goes to x1'''
                ''' otherwise +-1 word goes to x1'''
                if range(index,kwPosn,len(words)):
                    x1_words.append(word)
                else:
                    x2_words.append(word)
             
            
            
            # update vocab only when it is training data
            '''
            if is_train == 1:
                for word in set(x1_words):
                    vocab[word] += 1
                for word in set(x2_words):
                    vocab[word] += 1
            '''
                    
            if len(x1_words) > max_x1:
                x1_words = x1_words[:max_x1]
                change_count += 1
            if len(x2_words) > max_x2:
                x2_words = x2_words[:max_x2]
                change_count += 1
            
            datum = {"y": int(label),
                    "x1": " ".join(x1_words),
                    "x2": " ".join(x2_words),
                    "x1_len": len(x1_words),
                    "x2_len": len(x2_words)}
            queries.append(datum)
            all_ids.append(label)
     
    print ("length more than %i: %i" % (max_x2, change_count)) 
    
    return queries,all_ids


def read_all_data(allData, output_id_file,target, max_x1, max_x2, is_train):
    queries = []
    change_count = 0
    writer = open(output_id_file,'w')
    for line in allData:
        line = line.strip()
        [label, kw, _, text] = line.split('\t');
        text = text.lower()
                    
        # we create an one word window for the hashtag
        kwPosn = -1
        for index, word in enumerate(text.split()):
            if word.lower() == kw.lower() or word.lower() == '#' + kw.lower():
                kwPosn = index
                break
        if kwPosn == -1:
            print 'the target: ' + target + ' is absent - returning'
            continue
            
        x1_words = []
        x2_words = []
        for index, word in enumerate(text.split()):
            if range(index,kwPosn):
                x1_words.append(word)
            else:
                x2_words.append(word)
             
            
            
            # update vocab only when it is training data
            '''
            if is_train == 1:
                for word in set(x1_words):
                    vocab[word] += 1
                for word in set(x2_words):
                    vocab[word] += 1
            '''
                    
        if len(x1_words) > max_x1:
            x1_words = x1_words[:max_x1]
            change_count += 1
        if len(x2_words) > max_x2:
            x2_words = x2_words[:max_x2]
            change_count += 1
            
        datum = {"y": int(label),
                    "x1": " ".join(x1_words),
                    "x2": " ".join(x2_words),
                    "x1_len": len(x1_words),
                    "x2_len": len(x2_words)}
        queries.append(datum)
        writer.write((label))
        writer.write('\n')
    
    print ("length more than %i: %i" % (max_x2, change_count)) 
    
    writer.close()
    return queries
    

"""
input word_vecs is a dict from string to vector.
this function will generate two dicts: from string to index, from index to vector.
Also, a random vector is initialized for all unknown words with index 0.
""" 
def get_W_POS(word_vecs,wordPosMap,pos_vecs):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    k = len(word_vecs.values()[0])
    p = len(pos_vecs.values()[0])
    print 'k =', k
    print 'p =', p
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+2, k+p), dtype='float32')
    #W[0] = np.zeros(k)
    np.random.seed(4321) # first word is UNK
    W[0] = np.random.normal(0,0.17,k+p)
    W[1] = np.random.normal(0,0.17,k+p)
    p_vec = np.random.normal(0,0.17,p)

    i = 2
    for word in word_vecs:
        word_vec = word_vecs[word]
        pos_map =  wordPosMap.get(word)
        if pos_map is not None:
            p_vec = pos_vecs[pos_map]
     #   W[i] = np.concatenate(word_vec,p_vec)
        W[i] = np.hstack((word_vec,p_vec))
        
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

"""
input word_vecs is a dict from string to vector.
this function will generate two dicts: from string to index, from index to vector.
Also, a random vector is initialized for all unknown words with index 0.
""" 
def get_W_POS_LIWC(word_vecs,wordPosMap,wordLIWCMap,LIWCDictSize,pos_vecs):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    k = len(word_vecs.values()[0])
    p = len(pos_vecs.values()[0])
    n = LIWCDictSize
    print 'k =', k
    print 'p =', p
    print 'n =', n
    
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+2, k+p+n), dtype='float32')
    #W[0] = np.zeros(k)
    np.random.seed(4321) # first word is UNK
    W[0] = np.random.normal(0,0.17,k+p+n)
    W[1] = np.random.normal(0,0.17,k+p+n)
    i = 2
    p_vec = np.random.normal(0,0.17,p)
    liwc_vec = np.random.normal(0,0.17,n)
    for word in word_vecs:
        word_vec = word_vecs[word]
        pos_map =  wordPosMap.get(word)
        if pos_map is not None:
            p_vec = pos_vecs[pos_map]
        
        liwc_map =  wordLIWCMap.get(word)
        if liwc_map is not None:
            liwc_vec = wordLIWCMap.get(word)
        W[i] = np.hstack((word_vec,p_vec,liwc_vec))
        
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

"""
input word_vecs is a dict from string to vector.
this function will generate two dicts: from string to index, from index to vector.
Also, a random vector is initialized for all unknown words with index 0.
""" 
def get_W_LIWC(word_vecs,wordPosMap,wordLIWCMap,LIWCDictSize,pos_vecs):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    k = len(word_vecs.values()[0])
    n = LIWCDictSize
    print 'k =', k
    print 'n =', n
    
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+2, k+n), dtype='float32')
    #W[0] = np.zeros(k)
    np.random.seed(4321) # first word is UNK
    W[0] = np.random.normal(0,0.17,k+n)
    W[1] = np.random.normal(0,0.17,k+n)
    i = 2
    liwc_vec = np.random.normal(0,0.17,n)
    for word in word_vecs:
        word_vec = word_vecs[word]
        
        liwc_map =  wordLIWCMap.get(word)
        if liwc_map is not None:
            liwc_vec = wordLIWCMap.get(word)
        W[i] = np.hstack((word_vec,liwc_vec))
        
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


"""
input word_vecs is a dict from string to vector.
this function will generate two dicts: from string to index, from index to vector.
Also, a random vector is initialized for all unknown words with index 0.
""" 
def get_W(word_vecs):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    k = len(word_vecs.values()[0])
    print 'k =', k
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+2, k), dtype='float32')
    #W[0] = np.zeros(k)
    np.random.seed(4321) # first word is UNK
    W[0] = np.random.normal(0,0.17,k)
    W[1] = np.random.normal(0,0.17,k)
    i = 2
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def load_bin_pos_vec(fname):
    
    pos_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            pos_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            ##   f.read(binary_len)
    return pos_vecs


def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs



def train_senti(target):
    
    path = './data/input/sarcasm_senti_training/'
    
    train_file = path + 'tweet.' + target + '.target.SENTI.BOTH.TRAIN'
    output_train_file = './data/output/samelm/pkl/senti/2_cnn/' + 'tweet.' + target + '.target.SENTI.BOTH.TRAIN'   + '.pkl'
    output_train_id_file = './data/output/samelm/ids/senti/2_cnn/' + 'tweet.' + target + '.target.SENTI.BOTH.TRAIN' + '.id'
    
    max_x1 = 1
    max_x2 = 100
    x1_filter_h = [3]
    x2_filter_h = [1,2,3]
    
    np.random.seed(4321)
    print "loading training data...",
    train_data,train_id = read_data_file(train_file,target, max_x1, max_x2, 1)
    
    max_x1 = np.max(pd.DataFrame(train_data)["x1_len"])
    max_x2 = np.max(pd.DataFrame(train_data)["x2_len"])
    
    print "data loaded!"
    print "vocab size: " + str(len(vocab))
    print "max keyword length: " + str(max_x1)
    print "max sentence length: " + str(max_x2)
    
    #w2v_file = "/Users/wguo/projects/nn/data/w2v/GoogleNews-vectors-negative300.bin"
    w2v_file = "../data/w2v/sdata-vectors.bin"
    w2v_file = "./data/config/tweet.all.05032015.sg.model.bin"

    print "loading word2vec vectors...",
    
    w2v = load_bin_vec(w2v_file, vocab)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    #add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    
    cPickle.dump([train_data, W, word_idx_map, max_x1, max_x2], open(output_train_file, "wb"))
    
    writer1 = open(output_train_id_file,'w')
    for id in train_id:
        writer1.write(id)
        writer1.write('\n')
        
    writer1.close()
    
   
    
 
def train_test_allData(targets,vocab):
    
    path = '/Users/dg513/work/eclipse-workspace/sarcasm-workspace/SarcasmDetection/data/twitter_corpus/wsd/sentiment/samelm2/weiwei/'
    
    allTraining = []
    allTesting = []
    
    output_train_file = './data/output/samelm/pkl/2_cnn/' + 'tweet.' + 'ALLTARGETS' + '.target.TRAIN' + '.pkl'
    output_test_file = './data/output/samelm/pkl/2_cnn/' + 'tweet.' + 'ALLTARGETS' + '.target.TEST'   + '.pkl'
    
    output_train_id_file = './data/output/samelm/ids/2_cnn/' + 'tweet.' + 'ALLTARGETS' + '.target.TRAIN' + '.id'
    output_test_id_file = './data/output/samelm/ids/2_cnn/' + 'tweet.' + 'ALLTARGETS' + '.target.TEST' + '.id'

    
    max_x1 = 3
    max_x2 = 100
    w2v_file = "./data/config/tweet.all.05032015.sg.model.bin"
    print "loading word2vec vectors..."

    all_train_data = []
    all_test_data = []
    
    all_train_id = []
    all_test_id = []
    
    print "loading training data...",
  
    for target in targets:
        train_file = path + 'tweet.' + target + '.target.TRAIN'
        test_file = path + 'tweet.' + target + '.target.TEST'
        
        train_data,train_id = read_data_file(train_file,target, max_x1,max_x2, 1)
        all_train_data.extend(train_data)
        all_train_id.extend(train_id)
        
        test_data,test_id = read_data_file(test_file,target, max_x1,max_x2, 0)
        all_test_data.extend(test_data)
        all_test_id.extend(test_id)
        
    
    
    np.random.seed(4321)
 #   target = 'ALLTARGETS'
    cPickle.dump(all_test_data, open(output_test_file, "wb"))
    
   
    max_x1_train = np.max(pd.DataFrame(all_train_data)["x1_len"])
    max_x2_train = np.max(pd.DataFrame(all_train_data)["x2_len"])
    
    max_x1_test = np.max(pd.DataFrame(all_test_data)["x1_len"])
    max_x2_test = np.max(pd.DataFrame(all_test_data)["x2_len"])
    
    
    print "data loaded!"
    print "vocab size: " + str(len(vocab))
    print "max keyword length: " + str(max(max_x1_train,max_x1_test))
    print "max sentence length: " + str(max(max_x2_train,max_x2_test))
    
    #w2v_file = "/Users/wguo/projects/nn/data/w2v/GoogleNews-vectors-negative300.bin"
    w2v_file = "../data/w2v/sdata-vectors.bin"
    w2v_file = "./data/config/tweet.all.05032015.sg.model.bin"

    print "loading word2vec vectors...",
    
    w2v = load_bin_vec(w2v_file, vocab)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    #add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    
    cPickle.dump([all_train_data, W, word_idx_map,max(max_x1_train,max_x1_test), max(max_x2_train,max_x2_test)], open(output_train_file, "wb"))
    
    writer1 = open(output_train_id_file,'w')
    for id in all_train_id:
        writer1.write(id)
        writer1.write('\n')
        
    writer1.close()
    
    writer2 = open(output_test_id_file,'w')
    for id in all_test_id:
        writer2.write(id)
        writer2.write('\n')
        
    writer2.close()
    
    
    print "dataset created!"
    
def train_test_single(vocab,vocabPos,vocabLIWC,LIWCDictSize,path):
    
    '''we have two versions. text+extra and only text'''

    train_file = path + 'incongruence_cotext_training.txt'
  #  test_file = path + 'marker_nonmarker_cotext_test.txt'
    test_file = path  + 'sentiment_change_cotext_test.txt'

    output_train_word_file = './data/output/samelm/pkl/2_cnn/' + 'incongruence_cotext_training.word' + '.pkl'

  #  output_test_word_file = './data/output/samelm/pkl/2_cnn/' + 'marker_nonmarker_cotext_test.word' + '.pkl'
  #  output_test_word_file = './data/output/samelm/pkl/2_cnn/' + 'sentiment_change_cotext_test.word' + '.pkl'
    output_test_word_file = './data/output/samelm/pkl/2_cnn/' + 'incongruence_cotext_test.word' + '.pkl'

#    output_train_wordpos_file = './data/output/samelm/pkl/2_cnn/' + 'tweet.' + target + '.wordpos.TRAIN' + '.pkl'
#   output_test_wordpos_file = './data/output/samelm/pkl/2_cnn/' + 'tweet.' + target + '.wordpos.TEST' + '.pkl'

#    output_train_wordposliwc_file = './data/output/samelm/pkl/2_cnn/' + 'tweet.' + target + '.wordposliwc.TRAIN' + '.pkl'
#    output_test_wordposliwc_file = './data/output/samelm/pkl/2_cnn/' + 'tweet.' + target + '.wordposliwc.TEST' + '.pkl'

    output_train_wordliwc_file = './data/output/samelm/pkl/2_cnn/' + 'incongruence_cotext_training' + '.wordliwc.TRAIN' + '.pkl'

    #output_test_wordliwc_file = './data/output/samelm/pkl/2_cnn/' + 'marker_nonmarker_cotext_test' + '.wordliwc.TEST' + '.pkl'
   # output_test_wordliwc_file = './data/output/samelm/pkl/2_cnn/' + 'sentiment_change_cotext_test' + '.wordliwc.TEST' + '.pkl'
    output_test_wordliwc_file = './data/output/samelm/pkl/2_cnn/' + 'incongruence_cotext_test' + '.wordliwc.TEST' + '.pkl'


    output_train_id_file = './data/output/samelm/ids/2_cnn/' + 'incongruence_cotext_training' + '.word.TRAIN' + '.id'

    #output_test_id_file = './data/output/samelm/ids/2_cnn/' + 'marker_nonmarker_cotext_test' + '.word.TEST' + '.id'
 #   output_test_id_file = './data/output/samelm/ids/2_cnn/' + 'sentiment_change_cotext_test' + '.word.TEST' + '.id'
    output_test_id_file = './data/output/samelm/ids/2_cnn/' + 'incongruence_cotext_test' + '.word.TEST' + '.id'
    
    max_x1 = 3
    max_x2 = 100
    x1_filter_h = [3]
    x2_filter_h = [1,2,3]
    
    np.random.seed(4321)

    print "loading training data...",
    train_data,train_id = read_data_file_notarget(train_file, max_x1, max_x2, 1)
    test_data,test_id = read_data_file_notarget(test_file, max_x1,max_x2, 0)
    cPickle.dump(test_data, open(output_test_word_file, "wb"))
 
    '''
    max_x1 = np.max(pd.DataFrame(train_data)["x1_len"])
    max_x2 = np.max(pd.DataFrame(train_data)["x2_len"])
    '''
    
    print "data loaded!"
    print "vocab size: " + str(len(vocab))
    print "max keyword length: " + str(max_x1)
    print "max sentence length: " + str(max_x2)
    
    w2v_file = "./data/config/tweet.all.05032015.sg.model.bin"
    print "loading word2vec vectors...",
    
    w2v = load_bin_vec(w2v_file, vocab)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    
    W, word_idx_map = get_W(w2v)
    np.random.shuffle(train_data)
    cPickle.dump([train_data, W, word_idx_map, max_x1, max_x2], 
                 open(output_train_word_file, "wb"))
    
    writer1 = open(output_train_id_file,'w')
    for id in train_id:
        writer1.write(id)
        writer1.write('\n')
        
    writer1.close()
    
    writer2 = open(output_test_id_file,'w')
    for id in test_id:
        writer2.write(id)
        writer2.write('\n')
        
    writer2.close()
    '''
    posPath = '/Users/dg513/work/eclipse-workspace/distrib-workspace/wordVectorLucene/data/input/'
    posFile = 'tweet.all.targets.12202017.pos.sg.model.bin'
    p2v = load_bin_pos_vec(posPath+posFile)
    W, word_idx_map = get_W_POS(w2v,vocabPos,p2v)
    np.random.shuffle(train_data)
    cPickle.dump([train_data, W, word_idx_map, max_x1, max_x2], 
                 open(output_train_wordpos_file, "wb"))

    W, word_idx_map = get_W_LIWC(w2v,vocabPos,vocabLIWC,LIWCDictSize,p2v)
    np.random.shuffle(train_data)
    cPickle.dump([train_data, W, word_idx_map, max_x1, max_x2], 
                 open(output_train_wordposliwc_file, "wb"))
    '''
    W, word_idx_map = get_W_LIWC(w2v,vocabPos,vocabLIWC,LIWCDictSize,None)
    np.random.shuffle(train_data)
    cPickle.dump([train_data, W, word_idx_map, max_x1, max_x2], 
                 open(output_train_wordliwc_file, "wb"))
 
    print "dataset created!"

def train_test(target,vocab,vocabPos,vocabLIWC,LIWCDictSize,path):
    
    '''we have two versions. text+pos and only text'''

    train_file = path + 'tweet.' + target + '.target.TRAIN'
    test_file = path + 'tweet.' + target + '.target.TEST'

    output_train_word_file = './data/output/samelm/pkl/2_cnn/' + 'tweet.' + target + '.word.TRAIN' + '.pkl'
    output_test_word_file = './data/output/samelm/pkl/2_cnn/' + 'tweet.' + target + '.word.TEST' + '.pkl'

    output_train_wordpos_file = './data/output/samelm/pkl/2_cnn/' + 'tweet.' + target + '.wordpos.TRAIN' + '.pkl'
    output_test_wordpos_file = './data/output/samelm/pkl/2_cnn/' + 'tweet.' + target + '.wordpos.TEST' + '.pkl'

    output_train_wordposliwc_file = './data/output/samelm/pkl/2_cnn/' + 'tweet.' + target + '.wordposliwc.TRAIN' + '.pkl'
    output_test_wordposliwc_file = './data/output/samelm/pkl/2_cnn/' + 'tweet.' + target + '.wordposliwc.TEST' + '.pkl'

    output_train_wordliwc_file = './data/output/samelm/pkl/2_cnn/' + 'tweet.' + target + '.wordliwc.TRAIN' + '.pkl'
    output_test_wordliwc_file = './data/output/samelm/pkl/2_cnn/' + 'tweet.' + target + '.wordliwc.TEST' + '.pkl'


    output_train_id_file = './data/output/samelm/ids/2_cnn/' + 'tweet.' + target + '.word.TRAIN' + '.id'
    output_test_id_file = './data/output/samelm/ids/2_cnn/' + 'tweet.' + target + '.word.TEST' + '.id'
    
    max_x1 = 3
    max_x2 = 100
    x1_filter_h = [3]
    x2_filter_h = [1,2,3]
    
    np.random.seed(4321)

    print "loading training data...",
    train_data,train_id = read_data_file(train_file,target, max_x1, max_x2, 1)
    test_data,test_id = read_data_file(test_file,target, max_x1,max_x2, 0)
    cPickle.dump(test_data, open(output_test_word_file, "wb"))
 
    '''
    max_x1 = np.max(pd.DataFrame(train_data)["x1_len"])
    max_x2 = np.max(pd.DataFrame(train_data)["x2_len"])
    '''
    
    print "data loaded!"
    print "vocab size: " + str(len(vocab))
    print "max keyword length: " + str(max_x1)
    print "max sentence length: " + str(max_x2)
    
    w2v_file = "./data/config/tweet.all.05032015.sg.model.bin"
    print "loading word2vec vectors...",
    
    w2v = load_bin_vec(w2v_file, vocab)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    
    W, word_idx_map = get_W(w2v)
    np.random.shuffle(train_data)
    cPickle.dump([train_data, W, word_idx_map, max_x1, max_x2], 
                 open(output_train_word_file, "wb"))
    
    writer1 = open(output_train_id_file,'w')
    for id in train_id:
        writer1.write(id)
        writer1.write('\n')
        
    writer1.close()
    
    writer2 = open(output_test_id_file,'w')
    for id in test_id:
        writer2.write(id)
        writer2.write('\n')
        
    writer2.close()
    '''
    posPath = '/Users/dg513/work/eclipse-workspace/distrib-workspace/wordVectorLucene/data/input/'
    posFile = 'tweet.all.targets.12202017.pos.sg.model.bin'
    p2v = load_bin_pos_vec(posPath+posFile)
    W, word_idx_map = get_W_POS(w2v,vocabPos,p2v)
    np.random.shuffle(train_data)
    cPickle.dump([train_data, W, word_idx_map, max_x1, max_x2], 
                 open(output_train_wordpos_file, "wb"))

    W, word_idx_map = get_W_LIWC(w2v,vocabPos,vocabLIWC,LIWCDictSize,p2v)
    np.random.shuffle(train_data)
    cPickle.dump([train_data, W, word_idx_map, max_x1, max_x2], 
                 open(output_train_wordposliwc_file, "wb"))
    '''
    W, word_idx_map = get_W_LIWC(w2v,vocabPos,vocabLIWC,LIWCDictSize,None)
    np.random.shuffle(train_data)
    cPickle.dump([train_data, W, word_idx_map, max_x1, max_x2], 
                 open(output_train_wordliwc_file, "wb"))


 
    print "dataset created!"


def test_senti(target):
 
    path = './data/input/sarcasm_senti_testing/'
    test_file = path + 'tweet.' + target + '.target.SENTI.BOTH.TEST'
    output_test_file = './data/output/samelm/pkl/senti/2_cnn/' + 'tweet.' + target + '.target.SENTI.BOTH.TEST'   + '.pkl'
    output_test_id_file = './data/output/samelm/ids/senti/2_cnn/' + 'tweet.' + target + '.target.SENTI.BOTH.TEST' + '.id'

    max_x1 = 3
    max_x2 = 100

    test_data,test_id = read_data_file(test_file,target, max_x1,max_x2, 0)
    cPickle.dump(test_data, open(output_test_file, "wb"))

    writer2 = open(output_test_id_file,'w')
    for id in test_id:
        writer2.write(id)
        writer2.write('\n')
        
    writer2.close()
    
    
def test(target):
    
    path = './data/input/sarcasm_senti_training/'
    
    test_file = path + 'tweet.' + target + '.target.SENTI.BOTH.TEST'
    output_test_file = './data/output/samelm/pkl/senti/2_cnn/' + 'tweet.' + target + '.target.SENTI.BOTH.TEST'   + '.pkl'
    output_test_id_file = './data/output/samelm/ids/senti/2_cnn/' + 'tweet.' + target + '.target.SENTI.BOTH.TEST' + '.id'

    
    max_l = 50
    vocab = defaultdict(float) # empty 
    test_data = read_data_file(test_file, vocab, max_l, 0)
    cPickle.dump(test_data, open(output_file, "wb"))
    
    
def listTargets():
    file = open('./data/config/targets.txt')
    targets = [ line.strip() for line in file.readlines()]
    return targets

def createVocabLIWCMap(vocab,liwcDirPath):
    
    vocabLIWCMap = {}
    files = os.listdir(liwcDirPath)
    index = 0
    for file in files:
        if 'Store' in file:
            continue
        f = open(liwcDirPath+file)
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        for word in vocab:
            word = word.lower()
            dicts = vocabLIWCMap.get(word,[0] * len(files))
            if word in lines:
                dicts[index] = 1.0
            vocabLIWCMap[word] = dicts
                
                
        f.close()
        index+=1
    
    return vocabLIWCMap,len(files)

def vocab_pos_mapping(vocab,posFile):
    
    #aren't friend always there for -.-    V N R R P E
    wordPosMap ={}
    done = 0
    f = open(posFile)
    lines = f.readlines()
    f.close()
    for line in lines:
        elements = line.split('\t')
        words,pos = elements[0],elements[1]
        words = words.split()
        pos = pos.split()
        for index,word in enumerate(words):
            if word.strip().lower() in vocab:
                wordPosMap[word.strip().lower()] =pos[index]
            
            
            
        
    return wordPosMap


def parseArguments(argv):
    
    config_file = './data/config/'+argv[0].strip()
    config = ConfigParser.ConfigParser()
    config.read(config_file)
    kwargs = {}
    try:
        header = 'CNN_HEADER'
        kwargs['mainPath'] = config.get(header, 'main_path')
        pos_present = config.get(header, 'pos_present')
        if pos_present == 'True':
           # kwargs['posPath'] = config.get(header, 'pos_path')
            kwargs['posFile'] = config.get(header, 'pos_file')
        target_present = config.get(header,'target_present')
        if target_present == 'True':
            kwargs['targetFile'] = config.get(header,'target_file') 
        kwargs['LIWCPath'] = config.get(header,'liwc_path')
    except:
        print("check the parameters that you entered in the config file")
        exit()
        
    return kwargs

def main_targets(argv):
    
    kwargs = parseArguments(argv)
    
    path = kwargs.get('mainPath')
    posFile = kwargs.get('posFile')
    liwcDictPath = kwargs.get('LIWCPath')
  #  posFile = argv[2]
 
 #   posPath = '/Users/dg513/work/eclipse-workspace/distrib-workspace/wordVectorLucene/data/input/'
 #   posFile = 'tweet.all.targets.words.pos'

    targets = listTargets()
  #  vocab = create_all_vocab(targets)
  #  train_test_allData(targets,vocab)

  #  targets = ['always', 'amazing', 'attractive', 'awesome']
    vocab = create_all_vocab(targets,path)
    vocab_pos = vocab_pos_mapping(vocab,posFile)
    liwc_word_map,LIWCDictSize = createVocabLIWCMap(vocab,liwcDictPath)

    for target in targets:
   #     print 'working on target: ' + target
     #   vocab = create_vocab(target,path)
        train_test(target,vocab,vocab_pos,liwc_word_map,LIWCDictSize,path)
     #   train(target)
     #   train_senti(target)
     #   test(target)
     #   test_senti(target)

def main(argv):
    
    kwargs = parseArguments(argv)
    
    path = kwargs.get('mainPath')
    posFile = kwargs.get('posFile')
    liwcDictPath = kwargs.get('LIWCPath')
  #  posFile = argv[2]
 
 #   posPath = '/Users/dg513/work/eclipse-workspace/distrib-workspace/wordVectorLucene/data/input/'
 #   posFile = 'tweet.all.targets.words.pos'

  #  vocab = create_all_vocab(targets)
  #  train_test_allData(targets,vocab)

  #  targets = ['always', 'amazing', 'attractive', 'awesome']
    vocab = create_all_vocab_file(path)
    vocab_pos = vocab_pos_mapping(vocab,posFile)
    liwc_word_map,LIWCDictSize = createVocabLIWCMap(vocab,liwcDictPath)

    train_test_single(vocab,vocab_pos,liwc_word_map,LIWCDictSize,path)
     #   train(target)
     #   train_senti(target)
     #   test(target)
     #   test_senti(target)


if __name__=="__main__":

    main(sys.argv[1:])
    
