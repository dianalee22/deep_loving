import csv
import string
from gensim.parsing.preprocessing import STOPWORDS
import numpy as np

#TODO: add comments to each function

def isEnglish(s):
    #data coming in with unicode?
    #TODO: take in string WITH SYMBOLS REMOVED and return true if string is English
    token = s.split()
    for t in token:
        if not(t.isascii()):
            return False
    return True

def removeSymbols(s):
    #TODO: remove all symbols from a string
    char_set = string.punctuation
    for x in s:
        if x in char_set:
            s = s.replace(x,"")

    #TODO: split on white space
    t = s.split()

    #TODO: remove usernames & any words w/ digits included & links
    char_set = string.digits + '@'
    tokens = []
    for token in t:
        if not(any(elem in char_set for elem in token)):
            if 'http://' not in token:
                tokens.append(token)
    return " ".join(tokens)

def tokenize(s):
    #TODO: tokenize the data - how are we doing this? by words, by characters?
    words = s.split()

    #TODO: remove stop words - filler words from gensim.parsing.preprocessing
        #if you're running into an error, make sure that  gensim is installed (pip install -U gensim)
    words = [word for word in words if word not in STOPWORDS]
    return words

def convert_to_id(vocab,sentences):
    return np.stack([[vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in sentence] for sentence in sentences])

def get_data(train_file,test_file):
    #TODO: Figure out what the data needs to look like when it leaves preprocessing
    #COMMENT THESE OUT WHEN OFFICIALLY RUNNING
    train_file = 'data/train.csv'
    test_file = 'data/test.csv'

    #TODO: read in files
    #TODO: Check if comment is in English
    #TODO: Remove symbols from the data, usernames, numbers, and links
    train_data = []
    train_labels = []
    with open(train_file) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            if isEnglish(row[2]): #Check if comment is in English
                train_labels.append(row[0])
                s = removeSymbols(row[2]) #Remove symbols from the data, usernames, numbers, and links
                words = tokenize(s.lower()) #make sure everything is lowercase and tokenize
                train_data.append(words)

    test_data = []
    test_labels = []
    with open(test_file) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            if isEnglish(row[2]): #Check if comment is in English
                test_labels.append(row[0])
                s = removeSymbols(row[2]) #Remove symbols from the data, usernames, numbers, and links
                words = tokenize(s.lower()) #make sure everything is lowercase and tokenize
                test_data.append(words)

    #CHECKING THAT DATA HAS BEEN APPENDED CORRECTLY
    print(train_data[0:3])
    print(test_data[0:3])

    #TODO: build vocab from train data
    vocab = {}
    reverse_vocab = {} #why is this necessary?
    frequency = defaultdict(int) #why do we need frequency?
    vocab_ind = 1
    for data in train_data:
        for word in data:
            if word not in vocab:
                vocab[word] = vocab_ind
                reverse_vocab[vocab_ind] = word
                vocab_ind +=1
            freq[word] += 1
    vocab['UNK'] = len(vocab) + 1
    reverse_vocab[len(vocab)] = 'UNK' #why isn't it len(vocab) + 1 like the vocab?

    #TODO: convert data to their indices
    #TODO: check that this is converting to IDs correctly
    #What to do for test data, if word isn't in vocab, UNK it?
    #What words should we UNK?
    #Does this convert_to_id handle UNKING things?
    train_data = convert_to_id(vocab,train_data)
    test_data = convert_to_id(vocab,test_data)

    #TODO: see what filter_vocab does?
    #What to do for test data, if word isn't in vocab, UNK it?
    #What words should we UNK?

    #TODO: go through paper again and see how they are doing preprocessing/preparing the ngrams?
