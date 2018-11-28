from collections import defaultdict
from collections import Counter


def create_all_vocab_file(path):
    cutoff = 5
    vocab = defaultdict(float)
    allLines = []
    
    train_file = path + 'incongruence_cotext_training.txt'
    lines = open(train_file).readlines()
    allLines.extend(lines)
    
    raw = [process_line(l) for l in allLines ]
    cntx = Counter( [ w for e in raw for w in e ] )
    lst = [ x for x, y in cntx.iteritems() if y > cutoff ] + ["## UNK ##"]
    vocab = dict([ (y,x) for x,y in enumerate(lst) ])
    
    return vocab

def create_all_vocab(targets,path):
    cutoff = 5
    vocab = defaultdict(float)

    allLines = []
    for target in targets:
        train_file = path + 'tweet.' + target + '.target.TRAIN'
        lines = open(train_file).readlines()
        allLines.extend(lines)
    
    raw = [process_line(l) for l in allLines ]
    cntx = Counter( [ w for e in raw for w in e ] )
    lst = [ x for x, y in cntx.iteritems() if y > cutoff ] + ["## UNK ##"]
    vocab = dict([ (y,x) for x,y in enumerate(lst) ])
    
    return vocab # (this is a dictionary of [word] = [position] which is fine since we are only bothered about the key of the dict.
   

    
def create_vocab(target,path):
    
    cutoff = 0
    vocab = defaultdict(float)
    train_file = path + 'tweet.' + target + '.target.TRAIN'
    
    lines = open(train_file).readlines()
    raw = [process_line(l) for l in lines ]
    cntx = Counter( [ w for e in raw for w in e ] )
    lst = [ x for x, y in cntx.iteritems() if y > cutoff ] + ["## UNK ##"]
    vocab = dict([ (y,x) for x,y in enumerate(lst) ])
    
    return vocab # (this is a dictionary of [word] = [position] which is fine since we are only bothered about the key of the dict.
   

def process_line(line):
    
     [label, kw, _, text] = line.split('\t')
     return text.split()
