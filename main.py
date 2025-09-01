import torch
import torch.optim as optim
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
num_epochs = 7
batch_size = 16
lr = 0.01
nc = 1 #number of channels, since grayscale only 1 channel
nz = 100  #size of latent vector used to create the image
lr = 0.0002
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

save_images(train_loader) #used to ensure dataloader working properly

def initialize_weights(mo): #we are following the dcgan architecture so the weights should be normalized with mean = 0 and stdev = 0.02
    cname = mo.__class__.__name__
    if cname.find('Conv') != -1:
        nn.init.normal_(mo.weight.data, 0.0, 0.02)
    elif cname.find('BatchNorm') != -1:
        nn.init.normal_(mo.weight.data, 1.0, 0.02)
        nn.init.constant_(mo.bias.data, 0)

class Generator(nn.Module): #the generating image class
    def __init__(self): #makes use of sigmoid function as all mnist data is between 0 and 1
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz,4,kernel_size=7,stride=1,dilation=1,bias=False) #leads to a 4 x 7 x 7 output
            ,nn.BatchNorm2d(4)
            ,nn.ReLU(True)

            ,nn.ConvTranspose2d(4,2,kernel_size=2,stride=2,padding=0,bias=False) #leads to a 2 x 14 x 14 output
            ,nn.BatchNorm2d(2)
            ,nn.ReLU(True)

            ,nn.ConvTranspose2d(2,1,kernel_size=2,stride=2,padding=0,bias=False) #leads toa 1 x 28 x 28 output
            ,nn.Sigmoid()
        )
    
    def forward(self,input):
        return self.main(input)

netG = Generator().to(device)
netG.apply(initialize_weights) #iniializing netG with default weights

class Discriminator(nn.Module): #module used to classify between real and fake image
    def __init__(self):
        super(Discriminator,self).__init__() #each layer of the nn added on a different layer for simplicity
        self.main = nn.Sequential(
            nn.Conv2d(1,1,kernel_size=(3,3)), #leads to a 26 x 26 output
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(1,1,kernel_size=2,stride=2), #leads to a 13 x 13 ooutput
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(1,1,kernel_size=3,stride=2,padding=0), #leads to a 6x6 output
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Flatten(),
            nn.Linear(36,10),
            nn.ReLU(),
            nn.Linear(10,1),
            nn.Sigmoid()
        )
    
    def forward(self,inputs):
        return self.main(inputs)
    
netD = Discriminator().to(device)
netD.apply(initialize_weights)


#initializing the optimizers and loss functions
criterion = nn.BCELoss()
fixed_noise = torch.randn(batch_size,nz,1,1,device=device)
real_label = 1
fake_label = 0
optimD = optim.Adam(netD.parameters(),lr=lr)
optimG = optim.Adam(netG.parameters(),lr=lr)

#training loop
img_list = []
g_loss = []
d_loss = []
iters = 0

for epoch in range(num_epochs):
    for i,data in enumerate(train_loader,0):
    # ----- updating discriminator
        #real batch
        netD.zero_grad()
        real_cpu = data[0].to(device) #all the images moved to gpu or cpu
        b_size = real_cpu.size(0) #batch size
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device) #labels made
        output = netD(real_cpu).view(-1) #gets output of net D ie 0 or 1 if real or fake
        errD_real = criterion(output, label) #error calced
        errD_real.backward() #backwards propagated to make changes :]
        D_x = output.mean().item()

        #fake batch
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimD.step()
    # ------- updating generator
        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimG.step()

        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(train_loader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

      
        g_loss.append(errG.item())
        d_loss.append(errD.item())

#plot the loss and save the image
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(g_loss,label="G")
plt.plot(d_loss,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('loss.png')