from sklearn import svm
import os
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from GAN import Generator
import numpy as np
import torch.backends.cudnn as cudnn

cudnn.benchmark = True


def save_features(dataloader, batch_size, n_features, filename):
    features_ = np.zeros([len(dataloader), batch_size, n_features])
    labels_ = np.zeros([len(dataloader), batch_size])
    
    for b_idx, (imgs, labels) in enumerate(dataloader):
        imgs = imgs.cuda()
        features = netD.get_conv_features(imgs)
    
        features_[b_idx,:,:] = features.detach().cpu().numpy()
        labels_[b_idx,: ] = labels.detach().cpu().numpy()

    shape = features_.shape
    features_ = features_.reshape(shape[0]*shape[1], shape[2])
    labels_ = labels_.reshape(shape[0]*shape[1])
    feature_mat = np.concatenate((features_, labels_[:, np.newaxis]), axis=1)
    np.savetxt(filename, feature_mat)


dataset = 'mnist'
dataroot = '/home/AsifKhan/data/project/lsun/data'
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

batch_size = 100
num_workers = 3

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                    shuffle=True, num_workers=num_workers)

filters = [4, 4, 4, 4, 4]
strides = [1, 2, 2, 2, 2]
padding = [0, 1, 1, 1, 1]

nz = 256
ngf = 128
ndf = 128
lrd = 0.001
lrg = 0.001
beta1 = 0.5

model_path = '/home/AsifKhan/data/project/model/'
result_path = '/home/AsifKhan/data/project/results/'



netD = Discriminator(ndf, nc, filters, strides, padding)
epoch = 10
netD.load_state_dict(torch.load(model_path + 'netD_epoch_{}.pth'.format(epoch)))
print(netD)
netD.eval()
n_features = 4096 # 1024x2x2

feature_file = dataroot + '/train_features_{}.txt'.format(dataset)

if not os.path.exists(feature_file):
    save_features(dataloader, batch_size, n_features, feature_file)

print('load features')

data = np.loadtxt(feature_file, dtype=np.float16)
features, labels = data[:, : -1], data[:, -1: ]
shape = features.shape
print('Data has {} samples and {} features '.format(shape[0], shape[1]))

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)

print('Train SVM')
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X_train, y_train)
predict_labels = clf.predict(X_test)
print(classification_report(y_test, predict_labels))