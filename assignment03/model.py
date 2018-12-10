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
	def __init__(self, model, epochs, lr, momentum, use_gpu=False, optim='adam', reg=None):
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
		self.test_acc = []
		self.train_iter_loss = []
		self.reg = reg
		if self.use_gpu:
			self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
			
			if self.device == 'cuda:0':
				if torch.cuda.device_count()>1:
					self.model = nn.DataParallel(model)
				self.model.to(device)

		if optim=='sgd':
			self.optimizer = torch.optim.SGD(self.model.parameters(), lr = lr, momentum=momentum)
		elif optim=='nesterov':
			self.optimizer = torch.optim.SGD(self.model.parameters(), lr = lr, momentum=momentum, nesterov=True)
		else:
			ValueError('Optimizer Not Supported')


	def train(self, epoch, trainloader, print_every=100):
		'''
		method for training
		'''
		self.model.train()
		loss_batch = 0
		for b_idx, (train_data, train_labels) in enumerate(trainloader):
			train_data = train_data.view(-1, self.model.n_in)
			if self.use_gpu and self.device == 'cuda:0':
				train_data, train_labels = train_data.to(self.device), train_labels.to(self.device)
			# Forward Pass 
			train_preds = self.model.forward(train_data)
			loss = self.model.loss(train_preds, train_labels)
			if self.reg=='l2':
				loss = self.regularization(loss, lam=0.0001, reg='l2')
			elif self.reg=='l1':
				loss = self.regularization(loss, lam=0.000001, reg='l1')
			else:
				pass				
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
			if b_idx%print_every == 0:
				print('Train Epoch: {0} [{1}/{2} ({3:.0f}%)]\t Loss {4:.6f}'.
					format(epoch, b_idx*len(train_data), len(trainloader.dataset), 
						100.*b_idx/len(trainloader), loss))
			
			train_loss = loss.item()
			self.train_iter_loss.append(train_loss)
			loss_batch += train_loss
		loss_batch /= len(trainloader)
		self.train_loss.append(loss_batch)

	def test(self, testloader):
		'''
		method for testing
		'''
		self.model.eval()
		correct_, total_ = 0, 0
		with torch.no_grad():
			loss = 0
			for test_data, test_labels in testloader:
				test_data = test_data.view(-1, self.model.n_in)
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
			self.test_acc.append(accuracy_test)
			print('Accuracy of model on test set {0:.2f}'.format(accuracy_test))
			return accuracy_test
	
	def evaluator(self, trainloader, testloader, print_every=1000):
		for epoch in range(self.epochs):
			self.train(epoch, trainloader, print_every=print_every)
			acc_ = self.test(testloader)
		return acc_

	def regularization(self, loss, lam, reg='l2'):
		ll = torch.tensor(0.0)
		for W in self.model.parameters():
			if reg=='l2':
				ll += W.norm(2)
			else:
				ll += W.norm(1)
		loss = loss + 0.5*lam*ll**2
		return loss	

	def plot_loss(self):
		'''
		to visualize loss
		'''
		plt.plot(range(len(self.train_loss)), self.train_loss, label='Training Loss')
		plt.plot(range(len(self.test_loss)), self.test_loss, label='Testing Loss')
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.legend()
		return plt

	def plot_acc(self):

		plt.plot(range(len(self.test_acc)), self.test_acc, label='Testing Accuracy')
		plt.xlabel('Epoch')
		plt.ylabel('Accuracy')
		plt.legend()
		return plt

	def plot_iter_loss(self):
		'''
		to visualize loss
		'''
		plt.plot(range(len(self.train_iter_loss)), self.train_iter_loss, label='Training Loss')
		plt.xlabel('Iteration')
		plt.ylabel('Loss')
		plt.legend()
		return plt