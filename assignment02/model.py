import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import shuffle as skshuffle
import numpy as np
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import pdb

class MLP(torch.nn.Module):
	def __init__(self, n_in, n_hidden, drop_p, n_out, act):
		super(MLP, self).__init__()
		'''
		n_in: Number of Inputs
		n_hidden: List with units in hidden layers
		n_out: Number of Output Units
		'''
		self.n_in = n_in
		self.n_out = n_out
		self.n_hidden = n_hidden
		self.p = drop_p
		self.input_layer = nn.Linear(self.n_in, self.n_hidden[0])
		if act =='relu':
			self.nonlin = nn.ReLU()
		elif act =='tanh':
			self.nonlin = nn.Tanh()
		elif act=='sigmoid':
			self.nonlin = nn.Sigmoid()
		self.hidden = nn.ModuleList()

		for i in range(len(self.n_hidden)-1):
			self.hidden.append(nn.Linear(self.n_hidden[i], self.n_hidden[i+1]))
			self.hidden.append(self.nonlin) 
			self.hidden.append(nn.Dropout(p = self.p))
		self.final_fc = nn.Linear(self.n_hidden[-1], self.n_out)

		self.loss = nn.CrossEntropyLoss()

	def forward(self, X):
		'''
		forward pass
		'''
		X = self.nonlin(self.input_layer(X))
		for layer in self.hidden:
			X = layer(X)
		return X


class ModelEvaluator:
	def __init__(self, model, epochs, lr, use_gpu=False, optim='adam'):
		'''
		model: instance of pytorch model class
		epochs: number of training epochs
		lr: learning rate
		use_gpu: to use gpu
		optim: optimizer used for training, SGD or adam
		'''
		self.epochs = epochs
		self.lr = lr
		self.model = model
		self.use_gpu = use_gpu
		self.train_loss = []
		self.test_loss = []

		if self.use_gpu:
			self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
			
			if self.device == 'cuda:0':
				if torch.cuda.device_count()>1:
					self.model = nn.DataParallel(model)
				self.model.to(device)

		if optim=='adam':
			self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
		elif optim=='sgd':
			self.optimizer = torch.optim.SGD(self.model.parameters(), lr = lr, momentum=0.9)
		elif optim=='adadelta':
			self.optimizer = torch.optim.Adadelta(self.model.parameters(), lr = lr, eps=1e-6, weight_decay=0)
		elif optim=='adagrad':
			self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr = lr, lr_decay=1e-6, weight_decay=0)
		elif optim=='rmsprop':
			self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr = lr, alpha=0.995, eps=1e-7, weight_decay=0)
		else:
			ValueError('Optimizer Not Supported')


	def train(self, epoch, trainloader, print_every=100):
		'''
		method for training
		'''
		loss_batch = 0
		for b_idx, (train_data, train_labels) in enumerate(trainloader):
			if self.use_gpu and self.device == 'cuda:0':
				train_data, train_labels = train_data.to(self.device), train_labels.to(self.device)
			# Forward Pass 
			train_preds = self.model.forward(train_data)
			loss = self.model.loss(train_preds, train_labels)
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
			
			if b_idx%print_every == 0:
				print('Train Epoch: {0} [{1}/{2} ({3:.0f}%)]\t Loss {4:.6f}'.
					format(epoch, b_idx*len(train_data), len(trainloader.dataset), 
						100.*b_idx/len(trainloader), loss))
			
			loss_batch += loss
		loss_batch /= len(trainloader)
		self.train_loss.append(loss_batch)    

	def validation(self, valloader):
		'''
		method for testing
		'''
		correct_, total_ = 0, 0
		with torch.no_grad():
			loss = 0
			for val_data, val_labels in valloader:
				if self.use_gpu and self.device == 'cuda:0':
					val_data, val_labels = test_data.to(self.device), val_labels.to(self.device)
				val_preds = self.model.forward(val_data)
				
				loss += self.model.loss(val_data, val_labels)			
				
				_, val_pred_labels = torch.max(val_preds.data, 1)
				total_ += val_labels.size(0)
				correct_ += (val_pred_labels.cpu() == val_labels.cpu()).sum()
			
			loss /= len(valloader)
			self.val_loss.append(loss)
			accuracy_val = (100.0*correct_/total_)
			print('Validation Loss {1:.2f} Accuracy on validation set {2:.2f}'.format(loss, accuracy_val))
			return accuracy_val

	def test(self, testloader):
		'''
		method for testing
		'''
		correct_, total_ = 0, 0
		with torch.no_grad():
			loss = 0
			for test_data, test_labels in testloader:
				if self.use_gpu and self.device == 'cuda:0':
					test_data, test_labels = test_data.to(self.device), test_labels.to(self.device)
				test_preds = self.model.forward(test_data)

				loss += self.model.loss(test_preds, test_labels)

				_, test_pred_labels = torch.max(test_preds.data, 1)
				total_ += test_labels.size(0)
				correct_ += (test_pred_labels.cpu() == test_labels.cpu()).sum()
			
			loss /= len(testloader)
			self.test_loss.append(loss)
			accuracy_test = (100*correct_/total_)
			print('Accuracy of model on test set {0:.2f}'.format(accuracy_test))
			return accuracy_test
	
	def evaluator(self, trainloader, testloader, print_every=1000, validation=False):
		for epoch in range(self.epochs):
			self.train(epoch, trainloader, print_every=print_every)
			if validation:
				acc_ = self.validation(testloader)
			else:
				acc_ = self.test(testloader)
		return acc_

	def plot_loss(self, validation=False):
		'''
		to visualize loss
		'''
		plt.plot(range(len(self.train_loss)), self.train_loss, label='Training Loss')
		if validation:
			plt.plot(range(len(self.val_loss)), self.val_loss, label='Testing Loss')
		else:
			plt.plot(range(len(self.test_loss)), self.test_loss, label='Testing Loss')
		plt.xlabel('Iteration')
		plt.ylabel('Loss')
		
		return plt