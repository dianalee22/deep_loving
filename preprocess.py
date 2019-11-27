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

def get_data(train_file,test_file):
    #TODO: Figure out what the data needs to look like when it leaves preprocessing
    #COMMENT THESE OUT WHEN OFFICIALLY RUNNING
    train_file = 'data/train.csv'
    test_file = 'data/test.csv'

    #TODO: read in files
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

    #TODO: Remove symbols from the data, usernames, numbers, and links
    new_train_data = []
    for data in train_data:
        new_data = removeSymbols(data)
        new_train_data.append(new_data)

    #TODO: tokenize the data - how are we doing this? by words, by characters?

    #TODO: pad data? is this necessary

    #TODO: build vocab

    #TODO: convert data to their indices

