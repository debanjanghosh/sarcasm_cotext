import numpy as np
import cPickle
from collections import defaultdict
import sys, re
#import pandas as pd
import math
from boto.dynamodb.condition import NULL
from utils import *

from collections import Counter
from collections import defaultdict



windowSize = 1
begin = 2

'''helper function that acts on heuristics to select the range for 
    the first and the second CNN'''

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
The input file format is: 1st column is label (int), 2rd column is target, 
and 4nd column is utterance
we create two datum. (1) a target with its immediate neighbour [for first CNN]
and (2) remaining words [for second CNN]

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

            kwPosn = -1
            words = text.split()
            '''finding the target indexing'''
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
In this function we are also loading the POS vector
We horizontally stack the two vectors for use

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
    i = 2
    for word in word_vecs:
        word_vec = word_vecs[word]
        pos_map =  wordPosMap.get(word)
        p_vec = np.random.normal(0,0.17,p)
        if pos_map is not None:
            p_vec = pos_vecs[pos_map]

        W[i] = np.hstack((word_vec,p_vec))
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

'''loading POS vectors'''
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

'''loading binary vector (word2vec)'''

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



    

'''generating the training/test data for the experiments'''

def train_test(target,vocab,vocabPos,path,posPath,posFile):
    

    train_file = path + 'tweet.' + target + '.target.TRAIN'
    test_file = path + 'tweet.' + target + '.target.TEST'

    output_train_file = './data/output/samelm/pkl/2_cnn/' + 'tweet.' + target + '.target.TRAIN' + '.pkl.temp'
    output_test_file = './data/output/samelm/pkl/2_cnn/' + 'tweet.' + target + '.target.TEST' + '.pkl.temp'

    output_train_id_file = './data/output/samelm/ids/2_cnn/' + 'tweet.' + target + '.target.TRAIN' + '.id'
    output_test_id_file = './data/output/samelm/ids/2_cnn/' + 'tweet.' + target + '.target.TEST' + '.id'

    
    max_x1 = 3
    max_x2 = 100
    x1_filter_h = [3]
    x2_filter_h = [1,2,3]
    
    np.random.seed(4321)

    print "loading training data...",
    '''this is a setup for dual CNN. First CNN is applied over only the 
        target and its neighbor while the second CNN is applied over rest
        of the utterance'''
    
    train_data,train_id = read_data_file(train_file,target, max_x1, max_x2, 1)
    
    test_data,test_id = read_data_file(test_file,target, max_x1,max_x2, 0)
    cPickle.dump(test_data, open(output_test_file, "wb"))
 
    
    max_x1 = np.max(pd.DataFrame(train_data)["x1_len"])
    max_x2 = np.max(pd.DataFrame(train_data)["x2_len"])
    
    
    print "data loaded!"
    print "vocab size: " + str(len(vocab))
    print "max keyword length: " + str(max_x1)
    print "max sentence length: " + str(max_x2)
    
    '''we are using twitter embedding'''
    w2v_file = "./data/config/tweet.all.05032015.sg.model.bin"

    print "loading word2vec vectors...",
    
    w2v = load_bin_vec(w2v_file, vocab)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    
    '''pos to vector mapping'''    
    p2v = load_bin_pos_vec(posPath+posFile)

    '''loading the word vector and its mapping'''    
    W, word_idx_map = get_W_POS(w2v,vocabPos,p2v)
    np.random.shuffle(train_data)

    cPickle.dump([train_data, W, word_idx_map, max_x1, max_x2], open(output_train_file, "wb"))
    
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
    
    
    print "dataset created!"

    
def listTargets():
    file = open('./data/config/targets.txt')
    targets = [ line.strip() for line in file.readlines()]
    return targets

'''we are using the output from the CMU Tokenizer to load the POS for a 
    particular utterance. The POS embedding is generated using gensim (similar to tweet
    embedding)'''

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


def main(argv):

    ''' input path for the data'''
    path = argv[0]

    ''' we created a POS embedding space using gensim (POS tags for tweets are from 
        CMU's twitter tokenizer'''
    posPath = argv[1]
    posFile = argv[2]
    
    '''  the targets are used for the sense-disambiguation (EMNLP2015) research'''
    targets = listTargets()
    
    '''creating the vocabulary '''
    vocab = create_all_vocab(targets,path)
    '''mapping the vocabulary to POS. Note, since we are using static information 
        we are using only a specific POS embedding of a word '''
    vocab_pos = vocab_pos_mapping(vocab,posPath+posFile)


    '''prepare the train/test pkl files for CNN'''
    '''the pkl also contains the embedding mapping (word to vector id)'''
    for target in targets:
        train_test(target,vocab,vocab_pos,path,posPath,posFile)



if __name__=="__main__":

    main(sys.argv[1:])
    
