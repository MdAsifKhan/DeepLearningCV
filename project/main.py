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
from arguments import opt

random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

def assert_path(path):
	if os.path.isdir(path):
		pass
	else:
		os.makedirs(path)


assert_path(opt.dataroot)

if opt.dataset in ['imagenet', 'food']:
	dataset = dset.ImageFolder(root=opt.dataroot,
							transform=transforms.Compose([
							transforms.Resize(opt.imageSize),
							transforms.CenterCrop(opt.imageSize),
							transforms.ToTensor(),
							transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
							]))
	nc=3

elif opt.dataset == 'lsun':
	dataset = dset.LSUN(root=opt.dataroot, classes=['conference_room_train'],
							transform=transforms.Compose([
							transforms.Resize(opt.imageSize),
							transforms.CenterCrop(opt.imageSize),
							transforms.ToTensor(),
							transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
						]))
	nc=3

elif opt.dataset == 'mnist':
	dataset = dset.MNIST(root=opt.dataroot, download=True,
							transform=transforms.Compose([
							transforms.Resize(opt.imageSize),
							transforms.ToTensor(),
							transforms.Normalize((0.5,), (0.5,)),
						]))
	nc=1

elif opt.dataset == 'fake':
	dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
								transform=transforms.ToTensor())
	nc=3

assert dataset


dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
					shuffle=True, num_workers=opt.num_workers)


assert_path(opt.model_path)
assert_path(opt.result_path)

gan_  = GAN(opt.nz, opt.ngf, opt.ndf, opt.nc, 
					opt.filters, opt.strides, opt.padding, 
					opt.lrg, opt.lrd, opt.batch_size, opt.beta1,
					out_path=opt.model_path, use_gpu=opt.use_gpu, resume=opt.resume)


netG, netD = gan_.model()
evaluator = ModelEvaluator(opt.nz, opt.lrg, opt.lrd, opt.beta1, opt.batch_size, opt.nm_epochs, 
						model_path=opt.model_path, out_path=opt.result_path, use_gpu=opt.use_gpu)

evaluator.train(dataloader, netG, netD)