import torch
import torchvision.datasets as dsets
from model import MLP, ModelEvaluator
import torchvision.transforms as transforms
import numpy as np

trainset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
testset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())


batch_size = 200
nm_iters = 10000
nm_epochs = int(nm_iters / (len(trainset) / batch_size))
 
train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)

n_in = np.prod(trainset[0][0].numpy().shape)
print(n_in)
n_out = 10

# (A) MLP I OPT SGD
# Hyperparameters
lr = 0.01
n_hidden = [800, 800]
drop_p = 0.2
optim = 'sgd'
nlin = 'relu'
momentum = 0.9

model = MLP(n_in, n_hidden, drop_p, n_out, nlin)
modeleval = ModelEvaluator(model, nm_epochs, lr, momentum, use_gpu=True, optim=optim)
acc_ = modeleval.evaluator(train_loader, test_loader, print_every=100)

modelname = 'model_optimizer{}_nonlin_{}'.format(optim, nlin)
print('Accuracy of {0} is {1:.2f}'.format(modelname, acc_))

torch.save(modeleval.model.state_dict(), modelname)

plt = modeleval.plot_loss()
plt.savefig('train_test_loss, optim_{}, non_lin_{}'.format(optim, nlin))
plt.cla()
plt.clf()
plt1 = modeleval.plot_acc()
plt1.savefig('test_acc, optim_{}, non_lin_{}'.format(optim, nlin))
plt.cla()
plt.clf()
plt1 = modeleval.plot_iter_loss()
plt1.savefig('train_iterloss, optim_{}, non_lin_{}'.format(optim, nlin))