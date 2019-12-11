from collections import defaultdict
import csv
from gensim.parsing.preprocessing import STOPWORDS
import glob
import numpy as np
import string

def removeSymbols(s):
    """
    Takes in a comment and removes all symbols
    param s: string representing the social media comment from the read-in data
    return: string representing the social media comment with all of the symbols removed
    """
    #TODO: remove all symbols from a string
    char_set = string.punctuation + string.digits
    for x in s:
        if x in char_set:
            s = s.replace(x,"")
    return s

def tokenize(s):
    """
    Takes in a comment and splits the string on the white space
    param s: string representing the social media comment from the read-in data
    return: tokenized comment
    """
    #TODO: tokenize the data - how are we doing this? by words, by characters?
    words = s.split()

    #TODO: remove stop words - filler words from gensim.parsing.preprocessing
        #if you're running into an error, make sure that  gensim is installed (pip install -U gensim)
    words = [word for word in words if word not in STOPWORDS]
    return words

def convert_to_id(vocab,data):
    """
    Converts words to their unique IDs
    param vocab: dictionary from words to their unique ID
    param data: list of tokenized comments
    return: list of tokenized comments converted into their IDs
    """
    d = []
    for comment in data:
        c = []
        for word in comment:
            id = 0
            if word in vocab:
                id = vocab[word]
            else:
                id = vocab['UNK']
            c.append(id)
        d.append(c)

    return d

def get_data():
    """
    Reads in data from the data files, remove symbols from the comments, tokenizes the comments and appends
    them to train/test data, builds a vocabulary from the training data, and
    converts the comments to their id forms
    return: tuple of (training data of comments in id form, testing data of comments in id form, vocab dictionary)
    """
    #TODO: Figure out what the data needs to look like when it leaves preprocessing
    annotations_path = 'hate-speech-dataset-master/annotations_metadata.csv'
    # train_files = glob.glob('hate-speech-dataset-master/sampled_train/*.txt')
    # test_files = glob.glob('hate-speech-dataset-master/sampled_test/*.txt')
    all_files = glob.glob('hate-speech-dataset-master/all_files/*.txt')

    # train_file_id = {x.replace('hate-speech-dataset-master/sampled_train/','').replace('.txt','') for x in train_files}
    # test_file_id = {x.replace('hate-speech-dataset-master/sampled_test/','').replace('.txt','') for x in test_files}
    all_file_id = {x.replace('hate-speech-dataset-master/all_files/','').replace('.txt','') for x in all_files}

    #TODO: read in all labels into a dict mapping the file id to the corresponding label
    labels = {}
    with open(annotations_path) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            label = 0
            if row[4]  == 'hate':
                label = 1
            labels[row[0]] = label

    #TODO: gather the labels for the training and testing data
    # train_labels = []
    # for f in train_file_id:
    #     train_labels.append(labels[f])
    #
    # test_labels = []
    # for f in test_file_id:
    #     test_labels.append(labels[f])

    all_labels = []
    for f in all_file_id:
        all_labels.append(labels[f])

    #TODO: read in the comments, remove symbols, and tokenize
    # train_data = []
    # for file in train_files:
    #     f = open(file,"r")
    #     s = removeSymbols(f.read())
    #     train_data.append(tokenize(s.lower()))
    #
    # test_data = []
    # for file in test_files:
    #     f = open(file,"r")
    #     s = removeSymbols(f.read())
    #     test_data.append(tokenize(s.lower()))

    all_data = []
    for file in all_files:
        f = open(file,"r")
        s = removeSymbols(f.read())
        all_data.append(tokenize(s.lower()))


    count = len(all_data)
    #print(count)
    split = int(count // (10/9))
    #print(split)

    train_data = all_data[0:split]
    train_labels = all_labels[0:split]
    test_data = all_data [split:]
    test_labels = all_labels[split:]

    #TODO: build vocab from train data
    vocab = {}
    reverse_vocab = {}
    frequency = defaultdict(int)
    vocab_ind = 1
    for data in train_data:
        for word in data:
            if word not in vocab:
                vocab[word] = vocab_ind
                reverse_vocab[vocab_ind] = word
                vocab_ind +=1
            frequency[word] += 1
    vocab['UNK'] = len(vocab) + 1
    reverse_vocab[len(vocab)] = 'UNK' #why isn't it len(vocab) + 1 like the vocab?

    #TODO: convert data to their indices

    train_data = convert_to_id(vocab,train_data)
    test_data = convert_to_id(vocab,test_data)

    #TODO: Pad the data
    #TODO: Pad data
    max_train = max(map(lambda x:len(x), train_data))
    max_test =  max(map(lambda x:len(x), test_data))
    max_all=  max(max_train,max_test)

    new_train_data  =[]
    for data in train_data:
        t = data
        if max_all - len(data) > 0:
            t = np.pad(data,(0,max_all-len(data)),'constant',constant_values = len(vocab))
        new_train_data.append(t)

    new_test_data = []
    for data in test_data:
        t = data
        if max_all - len(data) > 0:
            t = np.pad(data,(0,max_all-len(data)),'constant',constant_values = len(vocab))
        new_test_data.append(t)

    print("train size")
    print(len(new_train_data),len(new_train_data[0]))
    print(len(new_test_data),len(new_test_data[0]))

    print("labels size")
    print(len(train_labels))
    print(len(test_labels))
    return new_train_data, new_test_data, train_labels, test_labels, vocab, reverse_vocab, frequency
