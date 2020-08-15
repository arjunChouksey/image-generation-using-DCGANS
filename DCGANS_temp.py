from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

batch_size = 64
image_size = 64

transform = transforms.Compose([transforms.Resize(image_size),transforms.ToTensor(),transforms.Normalize(mean = [0.5,0.5,0.5],std = [0.5,0.5,0.5])])

train_data =  datasets.CIFAR10(root = 'CIFAR10',train = True,transform = transform,download = True)

train_loader = DataLoader(train_data,batch_size = batch_size,shuffle =True)



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv')!=-1:
        m.weight.data.normal_(0.0,0.02)
    elif classname.find('BatchNorm')!=-1:
        m.weight.data.normal_(1,0.02)
        m.bias.data.fill_(0)
        

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.main = nn.Sequential(nn.ConvTranspose2d(100,512,4,1,0,bias = False),
                                  nn.BatchNorm2d(512),
                                  nn.ReLU(inplace = True),
                                  nn.ConvTranspose2d(512,256,4,2,1,bias =False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(inplace = True),
                                  nn.ConvTranspose2d(256,128,4,2,1,bias =False),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(inplace =True),
                                  nn.ConvTranspose2d(128,64,4,2,1,bias =False),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(inplace =True),
                                  nn.ConvTranspose2d(64,3,4,2,1,bias = False),
                                  nn.Tanh()
                                  )
    def forward(self,x):
        x = self.main(x)
        return x

netG = Generator()
netG.apply(weights_init)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.main = nn.Sequential(nn.Conv2d(3,64,4,2,1,bias = False),
                                  nn.LeakyReLU(0.2,inplace=True),
                                  nn.Conv2d(64,128,4,2,1,bias = False),
                                  nn.BatchNorm2d(128),
                                  nn.LeakyReLU(0.2,inplace = True),
                                  nn.Conv2d(128,256,4,2,1,bias = False),
                                  nn.BatchNorm2d(256),
                                  nn.LeakyReLU(0.2,inplace = True),
                                  nn.Conv2d(256,512,4,2,1,bias = False),
                                  nn.BatchNorm2d(512),
                                  nn.LeakyReLU(0.2, inplace = True),
                                  nn.Conv2d(512,1,4,1,0,bias= False),
                                  nn.Sigmoid())
    
    def forward(self,x):
        x = self.main(x)
        return x.view(-1)
        
netD = Discriminator()
netD.apply(weights_init)

criterion = nn.BCELoss()
learning_rate = 0.0002 
des_optimizer = torch.optim.Adam(netD.parameters(),lr = learning_rate,betas = (0.5,0.999))
gen_optimizer = torch.optim.Adam(netG.parameters(),lr = learning_rate,betas = (0.5,0.999))

Epochs = 10
k = 0
for epoch in range(Epochs):
    for i,data in tqdm(enumerate(train_loader,0)):
        
        print("training discrimiator")
        #zerograd for discriminator
        netD.zero_grad()
        #train discriminator with real image
        real,_ = data
        input = torch.autograd.Variable(real)
        target = torch.ones(input.size()[0])
        target = torch.autograd.Variable(target)
        output = netD(input)
        lossD_real = criterion(output,target)
        
        #train dicriminator with fake images 
        noise = torch.randn(input.size()[0],100,1,1)
        noise = torch.autograd.Variable(noise)
        target = torch.zeros(input.size()[0])
        fake = netG(noise)
        output = netD(fake.detach())
        lossD_fake = criterion(output,target)
        
        #backpropogating
        lossD = lossD_real + lossD_fake
        lossD.backward()
        des_optimizer.step()
        
        #zer_grad for generator
        print("training generator")
        netG.zero_grad()
        target = torch.ones(input.size()[0])
        target = torch.autograd.Variable(target)
        output =  netD(fake)
        lossG = criterion(output,target)
        lossG.backward()
        gen_optimizer.step()
        
        if i%1000 == 0:
            save_image(real,str(k)+'real.png',normalize = True)
            fake = netG(noise)
            save_image(fake,str(k)+'fake.png',normalize = True)
            k+=1
    print("epochs: {},loss_discriminator:{},loss_generator:{}".format(epoch+1,lossD.item(),lossG.item()))
    history["lossG"].append(lossG.item())
    history["lossD"].append(lossD.item())
    