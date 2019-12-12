import sys
import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
#import preprocess

class Model(nn.Module):

	def __init__(self, vocab_size, phrase_size):

		super(Model, self).__init__()
		self.vocab_size = vocab_size
		self.phrase_size = phrase_size

		# Initializing the hyperparameters
		self.batch_size = 500 
		self.embedding_size = 40 
		self.lstm_size = 100
		self.dense1_size = 100

		# Initializing trainable parameters
		self.embedding = torch.nn.Embedding(self.vocab_size, self.embedding_size)
		self.drop1 = nn.Dropout(0.25)
		self.lstm = nn.LSTM(self.embedding_size, self.lstm_size)
		self.drop2 = nn.Dropout(0.25)
		self.dense1 = nn.Linear(self.lstm_size, 2)
		self.dense2 = nn.Linear(self.phrase_size * 2, 2) 
		self.softmax = nn.Softmax(dim=1) 
		self.optimizer = torch.optim.Adam(self.parameters(), lr=(0.001))

	def call(self, input):
		"""
		Returns the probabilities of each comment / phrase being hate speech or not
		"""
		embedding = self.drop1(self.embedding(input))
		lstm_output = self.lstm(embedding)[0]
		dense_output1 = self.drop2(self.dense1(lstm_output))

		reshaped = dense_output1.view(-1, self.phrase_size * 2)
		dense_output2 = self.dense2(reshaped)
		prbs = self.softmax(dense_output2)

		return prbs

	def loss_function(self, prbs, labels):
		"""
		Calculates the Cross Entropy loss
		"""
		loss = nn.CrossEntropyLoss()
		model_loss = torch.mean(loss(prbs, labels))
		return model_loss

	def accuracy_function(self, prbs, labels):
		"""
		:return: mean accuracy over batch.
		"""
		indices = torch.max(prbs, 1)[1]
		eq_output = torch.eq(indices, labels) 

		# converts eq_output from an array of bools to an array of floats
		float_array = torch.FloatTensor(eq_output.numpy().astype(int)) 

		accuracy = torch.mean(float_array) 
		return accuracy
