import numpy as np
import tensorflow as tf
import csv
import string

def isEnglish(s):
    #data coming in with unicode?
    #TODO: take in string WITH SYMBOLS REMOVED and return true if string is English
    pass

def removeSymbols(s):
    #TODO: remove all symbols from a string
    char_set = string.punctuation
    for x in s:
        if x in char_set:
            s = s.replace(x,"")

    #TODO: tokenize string
    t = s.split()

    #TODO: remove usernames & any words w/ digits included
    char_set = string.digits
    tokens = []
    for token in t:
        if not(any(elem in char_set for elem in token)):
            tokens.append(token)
    return tokens

def get_data(train_file,test_file):
    train_file = 'data/train.csv'
    test_file = 'data/test.csv'
    #separate into comments and labels
    #for comments - figure out how to remove the \\xja;lskdjf; @symbols etc. anything that isn't a regular letter and isn't a word
    #first split on spaces? unk any word that contains a non-english letter??
    #create a vocab dict that maps each word in the corpus to a unique index (its id)
    # TODO: read in and tokenize training data
    # convert to their indices (making a 1-d list/array)

    # TODO: read in and tokenize testing data
    # convert to their indices (making a 1-d list/array)

    # TODO: return tuple of training tokens, testing tokens, and the vocab dictionary.

    #Do we need to pad the data? How are we tokenizing? How do we maintain the labels while tokenizing


    #READ IN FILES
    train_data = []
    train_labels = []
    with open(train_file) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            if isEnglish(row[2]):
                train_labels.append(row[0])
                train_data.append(row[2])

    test_data = []
    test_labels = []
    with open(test_file) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            if isEnglish(row[2]):
                test_labels.append(row[0])
                test_data.append(row[2])

    #Tokenize and remove symbols
    new_train_data = []
    for data in train_data:
        tokens = removeSymbols(data)
        #TODO: Check that this actually adds the tokens properly and doesn't just concatenate
        new_train_data.append(tokens)
