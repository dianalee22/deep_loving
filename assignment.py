import sys
import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from preprocess import *
from model import *

parser = argparse.ArgumentParser(description='deep_loving')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Sizes of image batches fed through the network')
parser.add_argument('--num-epochs', type=int, default=10,
                    help='Number of passes through the training data to make before stopping')

def train(model, train_input, train_labels):
	"""
	"""
	# > TOASK: what does preprocess return and what do the models take in?
	# shuffle train data
    indices = range(len(train_input))
    shuffled = tf.random.shuffle(indices)
    shuffled_inputs = tf.gather(train_input, shuffled)
    shuffled_labels = tf.gather(train_labels, shuffled)

	for i in range(0, len(train_input), model.batch_size):
		input_batch = shuffled_inputs[i:i+model.batch_size]
        label_batch = shuffled_labels[i:i+model.batch_size]

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
		input_batch = test_inputs[i:i+model.batch_size]
        label_batch = test_labels[i:i+model.batch_size]
		probabilities = model.call(input_batch)
		accuracy += accuracy_function(probabilities, test_labels)
		count += 1
	return accuracy / count

def main():
	train_data, test_data, train_labels, test_labels, vocab, reverse_vocab, frequency = preprocess.get_data('some-file-here')
	vocab_size = len(vocab)
	model = Model(vocab_size)
	# train and test for num_epochs 
	for i in range(args.num_epochs):
		train(model, train_data, train_labels)
		print(test(model, test_data))

if __name__ == '__main__':
	main()