import torch.nn as nn
import torch
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import random
from GAN import GAN
from evaluator import ModelEvaluator
import os

random.seed(4086)
torch.manual_seed(4086)

def assert_path(path):
	if os.path.isdir(path):
		pass
	else:
		os.makedirs(path)

dataset = 'mnist'
dataroot = '/home/asif/course-work/DeepLearningCV/project/lsun/data'

assert_path(dataroot)

imageSize = 64

if dataset in ['imagenet', 'food']:
	dataset = dset.ImageFolder(root=dataroot,
							transform=transforms.Compose([
							transforms.Resize(imageSize),
							transforms.CenterCrop(imageSize),
							transforms.ToTensor(),
							transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
							]))
	nc=3

elif dataset == 'lsun':
	dataset = dset.LSUN(root=dataroot, classes=['conference_room_train'],
							transform=transforms.Compose([
							transforms.Resize(imageSize),
							transforms.CenterCrop(imageSize),
							transforms.ToTensor(),
							transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
						]))
	nc=3

elif dataset == 'mnist':
	dataset = dset.MNIST(root=dataroot, download=True,
							transform=transforms.Compose([
							transforms.Resize(imageSize),
							transforms.ToTensor(),
							transforms.Normalize((0.5,), (0.5,)),
						]))
	nc=1

elif dataset == 'fake':
	dataset = dset.FakeData(image_size=(3, imageSize, imageSize),
								transform=transforms.ToTensor())
	nc=3

assert dataset

batch_size = 10
num_workers = 3

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
					shuffle=True, num_workers=num_workers)


nm_epochs = 20
use_gpu = True

filters = [4, 4, 4, 4, 4]
strides = [1, 2, 2, 2, 2]
padding = [0, 1, 1, 1, 1]

nz = 256
ngf = 128
ndf = 128
lrd = 0.001
lrg = 0.001
beta1 = 0.5

model_path = '/home/asif/course-work/DeepLearningCV/project/model/'
result_path = '/home/asif/course-work/DeepLearningCV/project/results/'

assert_path(model_path)
assert_path(result_path)

gan_  = GAN(nz, ngf, ndf, nc, 
					filters, strides, padding, 
					lrg, lrd, batch_size, beta1,
					out_path=model_path, use_gpu=True, resume=None)


netG, netD = gan_.model()
evaluator = ModelEvaluator(nz, lrg, lrd, beta1, batch_size, nm_epochs, 
						model_path=model_path, out_path=result_path, use_gpu=True)

evaluator.train(dataloader, netG, netD)