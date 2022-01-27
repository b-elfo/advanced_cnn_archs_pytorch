import os

from torch import FloatTensor

import torchvision
import torchvision.transforms as transforms 
from torchvision.datasets import ImageNet, CIFAR10, CIFAR100, MNIST, FashionMNIST
from torch.utils.data import DataLoader, Dataset

from PIL import Image

class Data(Dataset):
    def __init__(self,
                 path_to_data: str = './data',
                 img_fn: str = 'image.png', 
                 trg_fn: str = 'target.png',
                 targets: int = 10,
                 normalize: bool = False,
                 normal_means: tuple = (0.,0.,0.),
                 normal_stds: tuple = (1.,1.,1.)
                 ):
        # compile list of folders containing data
        self.folders = []
        for folder in os.listdir(path_to_data):
            self.folders.append(folder)
        # set image and target names
        self.img_fn = img_fn
        self.trg_fn = trg_fn
        # transforms
        transform = [ transforms.ToTensor() ]
        if normalize: # if normalizing
            transform.append(transforms.Normalize(normal_means,normal_stds))
        self.transforms = transforms.Compose(transform)
        # define number of classes
        self.targets = targets

    def __len__(self):
        return len(self.folders)

    def __getitem__(self,
                    idx: int,
                    ):
        path = self.folders[idx]
        image  = Image.open(os.path.join(path,self.img_fn))
        target = Image.open(os.path.join(path,self.trg_fn))
        return FloatTensor(image), FloatTensor(target)

###

def create_dataloader(data_name: str = None,
                      path_to_train_data: str = './data/train',
                      path_to_valid_data: str = './data/valid',
                      batch_size: int = 16,
                      ):
    # initialize dataset
    if data_name=='MNIST':
        train_dataset = MNIST('data/',
                              train=True,
                              transform=transforms.ToTensor(),
                              download=True,
                              )
        valid_dataset = MNIST('data/',
                              train=False,
                              transform=transforms.ToTensor(),
                              download=True,
                              )
    elif data_name=='FashionMNIST':
        train_dataset = FashionMNIST('data/',
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=True,
                                     )
        valid_dataset = FashionMNIST('data/',
                                     train=False,
                                     transform=transforms.ToTensor(),
                                     download=True,
                                     )
    elif data_name=='CIFAR10':
        train_dataset = CIFAR10('data/',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True,
                                )
        valid_dataset = CIFAR10('data/',
                                train=False,
                                transform=transforms.ToTensor(),
                                download=True,
                                )
    elif data_name=='CIFAR100':
        train_dataset = CIFAR100('data/',
                                 train=True,
                                 transform=transforms.ToTensor(),
                                 download=True,
                                 )
        valid_dataset = CIFAR100('data/',
                                 train=False,
                                 transform=transforms.ToTensor(),
                                 download=True,
                                 )
    elif data_name=='ImageNet':
        train_dataset = ImageNet('data/',
                                 train=True,
                                 transform=transforms.ToTensor(),
                                 download=True,
                                 )
        valid_dataset = ImageNet('data/',
                                 train=False,
                                 transform=transforms.ToTensor(),
                                 download=True,
                                 )
    else:
        train_dataset = Data(path_to_train_data)
        valid_dataset = Data(path_to_valid_data)

    # create dataloader
    train_dataloader = DataLoader(dataset=train_dataset, 
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  )
    valid_dataloader = DataLoader(dataset=valid_dataset, 
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  )
    return train_dataloader, valid_dataloader