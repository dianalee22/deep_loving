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
		self.vocab_size = vocab_size

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
		self.lstm = nn.LSTM(self.vocab_size*self.embedding_size, self.lstm1_size)
		self.drop2 = nn.Dropout(0.5)
		# find out what this input size is from the prev layer - the 100 is a placeholder
		self.dense = nn.Linear(100, self.vocab_size) 
		self.softmax = nn.Softmax()
		# the paper doesn't really use an optimizer, we can figure that out
		self.optimizer = torch.optim.Adam(self.parameters(), lr=(0.0001))

	def forward(self, input):
		"""
		"""
		embedding = self.drop1(self.embedding(input))
		lstm_output = self.lstm(embedding)
		dense_output = self.drop2(self.dense(lstm_output))
		prbs = self.softmax(dense_output)
	
		return prbs

	def accuracy_function(self, logits, labels):
		"""
		Computes the accuracy across a batch of logits and labels.

		:return: mean accuracy over batch.
		"""
		pass

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