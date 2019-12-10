import sys
import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
#import preprocess

class Model(nn.Module):
	"""
	"""

	def __init__(self, vocab_size):
		"""
		"""
		super(Model, self).__init__()
		self.vocab_size = vocab_size #5565

		# Initialize hyperparameters
		self.batch_size = 500 
		self.embedding_size = 40 
		self.lstm1_size = 100
		self.lastm2_size = 100
		self.dense1_size = 100

		# Initialize trainable parameters

		# The paper adds 1 to the embedding size, why?? Also look at other params in paper
		self.embedding = torch.nn.Embedding(self.vocab_size, self.embedding_size)
		self.drop1 = nn.Dropout(0.25)
		self.lstm = nn.LSTM(self.embedding_size, self.lstm1_size)
		self.drop2 = nn.Dropout(0.5) # maybe make this 25%
		# find out what this input size is from the prev layer - the 100 is a placeholder
		self.dense = nn.Linear(100, 2) 
		self.softmax = nn.Softmax(dim=2) # should we do softmax??
		# the paper doesn't really use an optimizer, we can figure that out
		self.optimizer = torch.optim.Adam(self.parameters(), lr=(0.001))

	def call(self, input):
		"""
		"""
		embedding = self.drop1(self.embedding(input)) # [500, 135, 40]
		lstm_output = self.lstm(embedding)[0] # [500, 135, 100]
		dense_output = self.drop2(self.dense(lstm_output)) #[500, 135, 2]
		# flatten to [500 x 270], then dense layer to get it to [500, 2], then softmax 
		word_prbs = self.softmax(dense_output) #[500, 135, 2]
		prbs = torch.mean(word_prbs, dim=1) #[500, 2]
	
		return prbs

	def loss_function(self, prbs, labels):
		# Labels: [500] for training
		loss = nn.CrossEntropyLoss()
		model_loss = torch.mean(loss(prbs, labels))
		return model_loss

	def accuracy_function(self, prbs, labels):
		"""
		:return: mean accuracy over batch.
		"""
		num_examples_test_input = prbs.size()[0]
		# Remove this once labels and prbs are of the same size again!!
		labels = labels[:num_examples_test_input]
		indices = torch.max(prbs, 1)[1]
		eq_output = torch.eq(indices, labels) 
		int_array = torch.FloatTensor(eq_output.numpy().astype(int)) # converts it from an array of bools to an array of floats
		accuracy = torch.mean(int_array) # may have to cast
		return accuracy

	def f1_function(prbs, labels):
		pass
