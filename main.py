import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,datasets
import numpy as np
from utilFunctions import save_images

#
set_seed = 100
torch.manual_seed(set_seed)
torch.use_deterministic_algorithms(True) #to allow for deterministic results

#creating datasets and loader
train_data = datasets.FashionMNIST(root = './data',train=True,download=True,transform=transforms.ToTensor())
test_data = datasets.FashionMNIST(root = './data',train=False,download=True,transform=transforms.ToTensor())


train_loader = DataLoader(train_data,batch_size=16,shuffle=True)
test_loader = DataLoader(test_data,batch_size=16,shuffle=True)

#label map
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

#hyperParams and device
img_size = (28,28) #not needed since all images are the same size but added for future refrence if applied to other datasets
num_epochs = 5
lr = 0.01
nc = 1 #number of channels, since grayscale only 1 channel
nz = 100  #size of latent vector used to create the image
ngf = 28
ndf = 28
lr = 0.0005
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

save_images(train_loader) #used to ensure dataloader working properly

def initialize_weights(mo): #we are following the dcgan architecture so the weights should be normalized with mean = 0 and stdev = 0.02
    cname = mo.__class__.__name__
    if cname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif cname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module): #the generating image class
    def __init__(self):
        self.main = nn.Sequential(

        )