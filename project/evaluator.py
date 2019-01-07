import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

class ModelEvaluator:
	def __init__(self, nz, lrg, lrd, beta1, batch_size, nm_epochs, 
						model_path, out_path, use_gpu=True):
		self.nz = nz
		self.lrg = lrg
		self.lrd = lrd
		self.beta1 = beta1
		self.batch_size = batch_size
		self.use_gpu = use_gpu
		self.nm_epochs = nm_epochs
		
		self.model_path = model_path
		self.out_path = out_path
		
		self.device = torch.device('cuda:0' if self.use_gpu else 'cpu')
		self.criterion = nn.BCELoss()
		self.real_label = 1
		self.fake_label = 0

	def train(self, dataloader, netG, netD):
		fixed_noise = torch.randn(self.batch_size, self.nz, 1, 1, device=self.device)


		optimizerD = optim.Adam(netD.parameters(), lr=self.lrd, betas=(self.beta1, 0.999))
		optimizerG = optim.Adam(netG.parameters(), lr=self.lrg, betas=(self.beta1, 0.999))

		for epoch in range(self.nm_epochs):
			for i, data in enumerate(dataloader, 0):
				############################
				# (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
				###########################
				# train with real
				
				netD.zero_grad()
				real_cpu = data[0].to(self.device)
				label = torch.full((self.batch_size,), self.real_label, 
										device=self.device)

				output = netD(real_cpu)
				
				errD_real = self.criterion(output, label)
				errD_real.backward()
				D_x = output.mean().item()

				# train with fake
				noise = torch.randn(self.batch_size, self.nz, 1, 1, device=self.device)
				fake = netG(noise)
				label.fill_(self.fake_label)
				
				output = netD(fake.detach())
				
				errD_fake = self.criterion(output, label)
				errD_fake.backward()
				D_G_z1 = output.mean().item()
				
				errD = errD_real + errD_fake
				optimizerD.step()

				############################
				# (2) Update G network: maximize log(D(G(z)))
				###########################
				netG.zero_grad()
				label.fill_(self.real_label)  # fake labels are real for generator cost
				output = netD(fake)
				
				errG = self.criterion(output, label)
				errG.backward()
				D_G_z2 = output.mean().item()
				optimizerG.step()

				print('[{}/{}][{}/{}] Loss_D: {:.4f} Loss_G: {:.4f} D(x): {:.4f} D(G(z)): {:.4f} / {:.4f}'
							.format(epoch, self.nm_epochs, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
				
				if i % 100 == 0:
					vutils.save_image(real_cpu, self.out_path + 'real_samples.png', normalize=True)
					fake = netG(fixed_noise)
					vutils.save_image(fake.detach(), self.out_path + 'fake_samples_epoch_{}.png'.format(epoch), normalize=True)


				torch.save(netG.state_dict(), self.model_path + 'netG_epoch_{}.pth'.format(epoch))
				torch.save(netD.state_dict(), self.model_path + 'netD_epoch_{}.pth'.format(epoch))