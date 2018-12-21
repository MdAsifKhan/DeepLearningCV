import numpy as np  
import torch
from model import ModelEvaluator, CNN
from utils import plot_img
import matplotlib.pyplot as plt


# load data
data = np.load('ORL_faces.npz')
trainX = data['trainX']
trainY = data['trainY']
testX = data['testX']
testY = data['testY']

import pdb

train_images, test_images = trainX.shape[0], testX.shape[0]
trainX = trainX.astype(np.float32).reshape(train_images, 112, 92)
testX = testX.astype(np.float32).reshape(test_images, 112, 92)
trainY = trainY.astype(np.long)
testY = testY.astype(np.long)

# Image Sanity check
sample_set = np.random.randint(0, train_images, 50)
plt = plot_img(trainX[sample_set])
plt.savefig('sanity_check_img_plot')

trainX = torch.from_numpy(trainX)
trainY = torch.from_numpy(trainY)
testX = torch.from_numpy(testX)
testY = torch.from_numpy(testY) 

train_data = torch.utils.data.TensorDataset(trainX, trainY)
test_data = torch.utils.data.TensorDataset(testX, testY)

batch_size = 10

train_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

# Hyperparameters
lr = 0.001
epochs = 15

# Model
drop_p = 0.1
n_in = trainX.shape[1]
n_out = len(np.unique(trainY))

pool = 'max'
optim = 'adam'
use_gpu = True

model = CNN(pool)
modeleval = ModelEvaluator(model, epochs, lr, batch_size, use_gpu=use_gpu, optim=optim)

acc_ = modeleval.evaluator(train_loader, test_loader, print_every=100)

modelname = 'model_optimizer_{}_lr_{}'.format(optim, lr)
print('Accuracy of {0} is {1:.2f}'.format(modelname, acc_))

torch.save(modeleval.model.state_dict(), modelname)
plt = modeleval.plot_loss()
plt.savefig('train_test_loss, optim_{}_lr_{}'.format(optim, lr))