import sys
import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F

def train(model, train_data):
	"""
	"""
	pass

def test(model, test_data):
	"""
	"""
	pass

def main():
	num_epochs = 10
	train_data, test_data = preprocess.get_data('some-file-here')
	vocab_size = 100
	model = Model(vocab_size)
	
	#for i in range(num_epochs):
		#train(model, train_data)
		#accuracy = test(model, test_data)

if __name__ == '__main__':
	main()