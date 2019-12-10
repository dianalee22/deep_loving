import sys
import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import preprocess
from model import *

parser = argparse.ArgumentParser(description='deep_loving')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Sizes of image batches fed through the network')
parser.add_argument('--num-epochs', type=int, default=15,
                    help='Number of passes through the training data to make before stopping')
args = parser.parse_args()

def train(model, train_input, train_labels):
    """
    """
    num_examples = train_input.size()[0]
    indices = torch.randperm(num_examples)

    train_input = torch.index_select(train_input, 0, indices) 
    train_labels = torch.index_select(train_labels, 0, indices)

    for i in range(0, len(train_input), model.batch_size):
        input_batch = train_input[i:i+model.batch_size]
        label_batch = train_labels[i:i+model.batch_size]

        probabilities = model.call(input_batch)
        loss = model.loss_function(probabilities, label_batch)
        # > TOASK-TA: why do I zero the gradient?
        # descend gradient
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        
        
def test(model, test_input, test_labels):
    """
    returns average accuracy across batches
    """
    accuracy = 0
    count = 0
    for i in range(0, len(test_input), model.batch_size):
        input_batch = torch.LongTensor(test_input[i:i+model.batch_size])
        label_batch = test_labels[i:i+model.batch_size]
        probabilities = model.call(input_batch)
        accuracy += model.accuracy_function(probabilities, test_labels)
        count += 1
    return accuracy / count

def main():
    train_data, test_data, train_labels, test_labels, vocab, reverse_vocab, frequency = preprocess.get_data()
    vocab_size = len(vocab) + 1
    model = Model(vocab_size)
    # train and test for num_epochs 
    for i in range(args.num_epochs):
        train_input = torch.LongTensor(train_data)
        train_labels = torch.LongTensor(train_labels)
        test_input = torch.LongTensor(test_data)
        test_labels = torch.LongTensor(test_labels)
        train(model, train_input, train_labels)
        print(test(model, test_data, test_labels))

if __name__ == '__main__':
    main()