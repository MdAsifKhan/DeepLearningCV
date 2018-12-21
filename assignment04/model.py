import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import shuffle as skshuffle
import numpy as np
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import pdb

class CNN(nn.Module):
	def __init__(self, pool='max', drop_p=0):
		super(CNN, self).__init__()
		self.drop_p = drop_p
		self.cnn1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
		self.bn1 = nn.BatchNorm2d(32)
		self.relu1 = nn.ReLU()
		self.cnn2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
		self.bn2 = nn.BatchNorm2d(64)
		self.relu2 = nn.ReLU()

		if pool == 'max':
		    self.pool = nn.MaxPool2d(kernel_size=2)
		else:
		    self.pool = nn.AvgPool2d(kernel_size=2)

		if self.drop_p:
		    self.drop = nn.Dropout(p=self.drop_p)
		self.fc1 = nn.Linear(64 * 28 * 23, 512)
		self.relu6 = nn.ReLU()
		self.fc2 = nn.Linear(512, 20)

		self.loss = nn.CrossEntropyLoss()

	def forward(self, x):
		if self.drop_p:
			out = self.pool(self.drop(self.relu1(self.bn1(self.cnn1(x)))))
			out = self.pool(self.drop(self.relu2(self.bn2(self.cnn2(out)))))
			out = out.view(out.size(0), -1)
			out = self.fc2(self.drop(self.relu6(self.fc1(out))))
		else:
			out = self.pool(self.relu1(self.bn1(self.cnn1(x))))
			out = self.pool(self.relu2(self.bn2(self.cnn2(out))))
			out = out.view(out.size(0), -1)
			out = self.fc2(self.relu6(self.fc1(out)))
		return out

	def cnn1_out(self, x):
		return self.cnn1(x)

	def cnn2_out(self, x):
		if self.drop_p:
			out = self.pool(self.drop(self.relu1(self.bn1(self.cnn1(x)))))
		else:
			out = self.pool(self.relu1(self.bn1(self.cnn1(x))))
		return self.cnn2(out)

class ModelEvaluator:
	def __init__(self, model, epochs, lr, batch_size, use_gpu=False, optim='adam', reg=None):
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
		self.batch_size = batch_size
		self.optim = optim
		if self.use_gpu:
			self.model = self.model.cuda()

		if optim == 'adam':
			self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
		elif optim == 'sgd':
			self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
		elif optim == 'adadelta':
		    self.optimizer = torch.optim.Adadelta(self.model.parameters(),
													lr=lr, eps=1e-6,
													weight_decay=0)
		elif optim == 'adagrad':
		    self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=lr,
													lr_decay=1e-6, weight_decay=0)
		elif optim == 'rmsprop':
		    self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr,
													alpha=0.995, eps=1e-7,
													weight_decay=0)
		else:
		    ValueError('Optimizer Not Supported')


	def train(self, epoch, trainloader, print_every=100):
		'''
		method for training
		'''
		self.model.train()
		loss_batch = 0
		for b_idx, (train_data, train_labels) in enumerate(trainloader):
			shape = train_data.shape
			train_data = train_data.view(shape[0], 1, shape[1], shape[2])
			if self.use_gpu:
				train_data, train_labels = train_data.cuda(), train_labels.cuda()
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
				shape = test_data.shape
				test_data = test_data.view(shape[0], 1, shape[1], shape[2])
				if self.use_gpu:
					test_data, test_labels = test_data.cuda(), test_labels.cuda()
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
			if epoch%5==0:
					save_model = {'epoch': epoch, 
						'state_dict': self.model.state_dict(),
						'optimizer': self.optimizer.state_dict()}
			model_name = 'Model_lr_{}_opt_{}_epoch_{}'.format(self.lr, self.optim, epoch)
			model_dir = '../assignment04/model/' + model_name
			torch.save(save_model, model_dir)			
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