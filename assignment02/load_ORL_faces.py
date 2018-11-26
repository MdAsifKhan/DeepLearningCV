import numpy as np  
import torch
from model import ModelEvaluator,MLP
import pdb 

# load data
data = np.load('ORL_faces.npz')
trainX = data['trainX']
trainY = data['trainY']
testX = data['testX']
testY = data['testY']

trainX = trainX/255
testX = testX/255
trainX = trainX.astype(np.float32)
testX = testX.astype(np.float32)
trainY = trainY.astype(np.long)
testY = testY.astype(np.long)

trainX = torch.from_numpy(trainX)
trainY = torch.from_numpy(trainY)
testX = torch.from_numpy(testX)
testY = torch.from_numpy(testY) 

train_data = torch.utils.data.TensorDataset(trainX, trainY)
test_data = torch.utils.data.TensorDataset(testX, testY)

batch_size = 100

train_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

# Hyperparameters
lr = 0.01
epochs = 50
# Model
n_hidden = [1024, 512]
l2 = 0
drop_p = 0.1
optim = 'sgd'
nlin = 'relu'
n_in = trainX.shape[1]
n_out = len(np.unique(trainY))

model = MLP(n_in, n_hidden, drop_p, n_out, nlin)
modeleval = ModelEvaluator(model, epochs, lr, use_gpu=True, optim=optim)
acc_ = modeleval.evaluator(train_loader, test_loader, print_every=100, validation=False)

modelname = 'model_optimizer{}_nonlin_{}'.format(optim, nlin)
print('Accuracy of {0} is {1:.2f}'.format(modelname, acc_))

torch.save(modeleval.model.state_dict(), modelname)
plt = modeleval.plot_loss()
plt.savefig('train_test_loss, optim_{}, non_lin_{}'.format(optim, nlin))
