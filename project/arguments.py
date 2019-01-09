import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', required=True, help='imagenet | food | mnist | fake | lsun', default='imagenet')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--filters', type=list, default=[4, 4, 4, 4, 4], help='size of filters in convolution layers')
parser.add_argument('--strides', type=list, default=[1, 2, 2, 2, 2], help='size of strides')
parser.add_argument('--padding', type=list, default=[0, 1, 1, 1, 1], help='size of padding')

parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nm_epochs', type=int, default=25, help='number of epochs to train for')

parser.add_argument('--resume', type=int, default=None, help='epoch at which training resumes')
parser.add_argument('--lrd', type=float, default=0.0002, help='learning rate discriminator, default=0.0002')
parser.add_argument('--lrg', type=float, default=0.0002, help='learning rate generator, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--netG', default='', help='path to netG (to continue training)')
parser.add_argument('--netD', default='', help='path to netD (to continue training)')
parser.add_argument('--model_path', default='.', help='folder to output model checkpoints')
parser.add_argument('--result_path', default='.', help='folder to output image checkpoints')
parser.add_argument('--use_gpu', type=bool, default=True, help='Enable GPU Training')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
if opt.dataroot == '':
	opt.dataroot = '/home/asif/course-work/DeepLearningCV/project/lsun/data'
if opt.model_path == '':
	opt.model_path = '/home/asif/course-work/DeepLearningCV/project/model/'
if opt.result_path == '':
	opt.result_path = '/home/asif/course-work/DeepLearningCV/project/results/'

opt.imageSize = 64
opt.batch_size = 200
opt.num_workers = 3
opt.nm_epochs = 100
opt.use_gpu = True

#opt.resume = assign_last_epoch_number

opt.filters = [4, 4, 4, 4, 4]
opt.strides = [1, 2, 2, 2, 2]
opt.padding = [0, 1, 1, 1, 1]

opt.nz = 256
opt.ngf = 128
opt.ndf = 128
opt.lrd = 0.0002
opt.lrg = 0.0002
opt.beta1 = 0.5
