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
		self.dense1_size = 100

		# Initialize trainable parameters
		self.embedding = torch.nn.Embedding(self.vocab_size, self.embedding_size)
		self.drop1 = nn.Dropout(0.25)
		self.lstm = nn.LSTM(self.embedding_size, self.lstm1_size)
		self.drop2 = nn.Dropout(0.25) # Paper had this as 50%
		self.dense1 = nn.Linear(self.lstm1_size, 2)
		self.dense2 = nn.Linear(270, 2) 
		self.softmax = nn.Softmax(dim=1) 
		self.optimizer = torch.optim.Adam(self.parameters(), lr=(0.001))

	def call(self, input):
		"""
		"""
		embedding = self.drop1(self.embedding(input)) # [500, 135, 40]
		lstm_output = self.lstm(embedding)[0] # [500, 135, 100]
		dense_output1 = self.drop2(self.dense1(lstm_output)) #[500, 135, 2]

		hidden_layer_size = dense_output1.size()[1] * dense_output1.size()[2]

		reshaped = dense_output1.view(-1, hidden_layer_size)
		dense_output2 = self.dense2(reshaped)
		word_prbs = self.softmax(dense_output2)

		return word_prbs

	def loss_function(self, prbs, labels):

		loss = nn.CrossEntropyLoss()
		model_loss = torch.mean(loss(prbs, labels))
		return model_loss

	def accuracy_function(self, prbs, labels):
		"""
		:return: mean accuracy over batch.
		"""
		indices = torch.max(prbs, 1)[1]
		eq_output = torch.eq(indices, labels) 

		# converts it from an array of bools to an array of floats
		int_array = torch.FloatTensor(eq_output.numpy().astype(int)) 

		accuracy = torch.mean(int_array) 
		return accuracy

	def f1_function(prbs, labels):
		pass
