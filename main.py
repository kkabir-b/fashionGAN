import torch
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,datasets
import numpy as np
from utilFunctions import save_images

#creating datasets and loader
train_data = datasets.FashionMNIST(root = '/home/kkabir/Desktop/projects/fashionGAN/data',train=True,download=True,transform=transforms.ToTensor())
test_data = datasets.FashionMNIST(root = '/home/kkabir/Desktop/projects/fashionGAN/data',train=False,download=True,transform=transforms.ToTensor())


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

