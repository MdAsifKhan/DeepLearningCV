import os
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from GAN import Discriminator
import numpy as np
import torch.backends.cudnn as cudnn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import svm
from arguments import opt
cudnn.benchmark = True

def assert_path(path):
    if os.path.isdir(path):
        pass
    else:
        os.makedirs(path)

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
    feature_mat = np.concatenate((features_, labels_[:, np.newaxis]), axis=1).astype(np.float16)
    np.savetxt(filename, feature_mat)


feature_file = features_path + 'features_{}.txt'.format(opt.dataset)

if opt.dataset in ['imagenet', 'food']:
    dataset = dset.ImageFolder(root=opt.dataroot,
                            transform=transforms.Compose([
                            transforms.Resize(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    nc=3

elif dataset == 'lsun':
    dataset = dset.LSUN(root=opt.dataroot, classes=['conference_room_train'],
                            transform=transforms.Compose([
                            transforms.Resize(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
    nc=3

elif dataset == 'mnist':
    dataset = dset.MNIST(root=opt.dataroot, download=True,
                            transform=transforms.Compose([
                            transforms.Resize(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,)),
                        ]))
    nc=1

elif dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                                transform=transforms.ToTensor())
    nc=3

assert dataset


print('Saving Features')
if not os.path.exists(feature_file):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                    shuffle=True, num_workers=opt.num_workers)

    
    netD = Discriminator(opt.ndf, opt.nc, opt.filters, opt.strides, opt.padding)
    netD.cuda()
    
    epoch = 10
    netD.load_state_dict(torch.load(opt.model_path + 'netD_epoch_{}.pth'.format(epoch)))
    
    print(netD)
    netD.eval()
    n_features = 4096 # 1024x2x2
    save_features(dataloader, opt.batch_size, n_features, feature_file)

print('Load Features')
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