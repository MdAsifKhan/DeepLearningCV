import torch
import torch.nn as nn
import pdb

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)

class Generator(nn.Module):
	def __init__(self, nz, ngf, nc, filters, strides, padding):
		super(Generator, self).__init__()

		self.nz = nz
		self.ngf = ngf
		self.nc = nc
		self.filters = filters
		self.strides = strides
		self.padding = padding
		self.gen = nn.Sequential(
						nn.ConvTranspose2d(self.nz, self.ngf * 8, self.filters[0], self.strides[0], self.padding[0], bias=False),
						nn.BatchNorm2d(self.ngf * 8),
						nn.ReLU(True),

						nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, self.filters[1], self.strides[1], self.padding[1], bias=False),
						nn.BatchNorm2d(self.ngf * 4),
						nn.ReLU(True),

						nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, self.filters[2], self.strides[2], self.padding[2], bias=False),
						nn.BatchNorm2d(self.ngf * 2),
						nn.ReLU(True),

						nn.ConvTranspose2d(self.ngf * 2, self.ngf, self.filters[3], self.strides[3], self.padding[3], bias=False),
						nn.BatchNorm2d(self.ngf),
						nn.ReLU(True),

						nn.ConvTranspose2d(self.ngf, self.nc, self.filters[4], self.strides[4], self.padding[4], bias=False),
						nn.Tanh()
						)

	def forward(self, x):
		return self.gen(x)


class Discriminator(nn.Module):
	def __init__(self, ndf, nc, filters, strides, padding):
		super(Discriminator, self).__init__()
		self.ndf = ndf
		self.nc = nc
		self.filters = filters
		self.strides = strides
		self.padding = padding
		self.conv1 = nn.Sequential( 
						nn.Conv2d(self.nc, self.ndf, self.filters[4], self.strides[4], self.padding[4], bias=False),
						nn.LeakyReLU(0.2, inplace=True)
						)

		self.conv2 = nn.Sequential(
						nn.Conv2d(self.ndf, self.ndf * 2, self.filters[3], self.strides[3], self.padding[3], bias=False),
						nn.BatchNorm2d(self.ndf * 2),
						nn.LeakyReLU(0.2, inplace=True)
						)

		self.conv3 = nn.Sequential(
						nn.Conv2d(self.ndf * 2, self.ndf * 4, self.filters[2], self.strides[2], self.padding[2], bias=False),
						nn.BatchNorm2d(self.ndf * 4),
						nn.LeakyReLU(0.2, inplace=True)
						)

		self.conv4 = nn.Sequential(
						nn.Conv2d(self.ndf * 4, self.ndf * 8, self.filters[1], self.strides[1], self.padding[1], bias=False),
						nn.BatchNorm2d(self.ndf * 8),
						nn.LeakyReLU(0.2, inplace=True)
						)

		self.conv5 = nn.Sequential(
			nn.Conv2d(self.ndf * 8, 1, self.filters[0], self.strides[0], self.padding[0], bias=False),
			nn.Sigmoid()
			)


	def forward(self, x):
		out = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x)))))
		return out.view(-1, 1).squeeze(1)

	def get_conv_features(self, x):
		features1 = self.conv1(x)
		features2 = self.conv2(features1)
		features3 = self.conv3(features2)
		features4 = self.conv4(features3)

		pool_ = nn.MaxPool2d(int(features4.size(2) / 2))
		features4 = pool_(features4)
		feature_vec = features4.view(x.size(0), -1).squeeze(1)
		return feature_vec


class GAN(nn.Module):
	def __init__(self, nz, ngf, ndf, nc, 
					filters, strides, padding, 
					lrg, lrd, batch_size, beta1, out_path='',
					use_gpu=True, resume=False):
		
		self.nz = nz
		self.ngf = ngf
		self.ndf = ndf
		self.nc = nc
		self.filters = filters
		self.strides = strides
		self.padding = padding

		self.use_gpu = use_gpu
		self.out_path = out_path
		self.resume = resume


	def model(self):
		netG = Generator(self.nz, self.ngf, self.nc, self.filters, 
							self.strides, self.padding)
		netG.apply(weights_init)

		netD = Discriminator(self.ndf, self.nc, self.filters, 
							self.strides, self.padding)
		netD.apply(weights_init)

		if self.use_gpu:
			netG, netD = netG.cuda(), netD.cuda()

		if self.resume:
			netG = load_state_dict(torch.load(self.out_path + 'gen/netG_epoch_{}.pth'.format(resume)))
			print('Generator')
			print(netG)
			netD = load_state_dict(torch.load(self.out_path + 'discr/netD_epoch_{}.pth'.format(resume)))
			print('Discriminator')
			print(netD)

		return netG, netD