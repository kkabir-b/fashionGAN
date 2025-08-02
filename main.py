import torch
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,datasets

#creating datasets and loader
train_data = datasets.FashionMNIST(root = '/data',train=True,download=True,transform=transforms.ToTensor())
test_data = datasets.FashionMNIST(root = '/data',train=False,download=True,transform=transforms.ToTensor())


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
