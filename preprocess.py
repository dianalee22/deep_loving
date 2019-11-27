import numpy as np
import tensorflow as tf
import csv
def get_data(train_file,test_file):
    train_file = 'data/train.csv'
    test_file = 'data/test.csv'
    
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
            train_labels.append(row[0])
            train_data.append(row[2])

    test_data = []
    test_labels = []
    with open(test_file) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            test_labels.append(row[0])
            test_data.append(row[2])
