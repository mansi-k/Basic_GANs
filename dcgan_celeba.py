import argparse
import os
import numpy as np
import math
import zipfile
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from torchvision.datasets import ImageFolder

import torch.nn as nn
import torch.nn.functional as F
import torch



parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--latent_dim", type=int, default=100)
parser.add_argument("--img_size", type=int, default=32)
parser.add_argument("--channels", type=int, default=1)
parser.add_argument("--load_ckpt", type=str, default=None)
parser.add_argument("--save_ckpt", type=str, default=None)
opt = parser.parse_args()
print(opt)
lr=0.0002
b1=0.5
b2=0.999



img_shape = (opt.channels, opt.img_size, opt.img_size)


def init_w(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif class_name.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        

        self.model = nn.Sequential(
            nn.Conv2d(opt.channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        
       

    def forward(self, img):
        
        out = self.model(img)
        
        

        return out






class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        

        self.model_ = nn.Sequential(
            nn.ConvTranspose2d( opt.latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d( 256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d( 128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d( 64, opt.channels, 4, 2, 1, bias=False),
            nn.Tanh()
            
        )

    def forward(self, z):
        
       
        img = self.model_(z)
        return img










g = Generator()
d = Discriminator()
g.apply(init_w)
d.apply(init_w)


loss = torch.nn.BCELoss()


cuda = True if torch.cuda.is_available() else False
if cuda:
    g.cuda()
    d.cuda()
    loss.cuda()
    



opt_G = torch.optim.Adam(g.parameters(), lr=lr, betas=(b1, b2))
opt_D = torch.optim.Adam(d.parameters(), lr=lr, betas=(b1,b2))




    
ckpt = 'checkpoints'

def save_checkpoint(state, model_type):
    torch.save(state, ckpt+"/"+model_type+"_"+opt.save_ckpt)



data_folder = 'data/celeba/'
# zip_path = f'{data_root}/img_align_celeba.zip'
# with zipfile.ZipFile(zip_path, 'r') as ziphandler:
#   ziphandler.extractall(data_folder)

transform = transforms.Compose([
    
    transforms.Resize((opt.img_size,opt.img_size)),
    transforms.CenterCrop(opt.img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataset = ImageFolder(root=data_folder, transform=transform)
dataloader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=True, num_workers=2, drop_last=True)



Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


start_epoch = 0
if opt.load_ckpt:
    g_ckpt = torch.load(ckpt+'/G_'+opt.load_ckpt, map_location='cpu')
    d_ckpt = torch.load(ckpt+'/D_'+opt.load_ckpt, map_location='cpu')
    assert g_ckpt['epoch']==d_ckpt['epoch'] , "different epochs"
    g.load_state_dict(g_ckpt['model_state_dict'])
    opt_G.load_state_dict(g_ckpt['optimizer_state_dict'])
    g_loss = g_ckpt['loss']
    d.load_state_dict(d_ckpt['model_state_dict'])
    opt_D.load_state_dict(d_ckpt['optimizer_state_dict'])
    d_loss = d_ckpt['loss']
    start_epoch = g_ckpt['epoch']+1
    
    

ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
for epoch in range(start_epoch,opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        opt_G.zero_grad()
        
        valid = torch.full((imgs.shape[0],), 1, dtype=torch.float, device=device)
        
        z = torch.randn(imgs.shape[0], opt.latent_dim, 1, 1,device=device)
        gen_imgs = g(z)

        
        g_loss = loss(g(gen_imgs).view(-1), valid)
        g_loss.backward()
        
        
        opt_G.step()

      
      
      
      
        opt_D.zero_grad()
        
        fake = torch.full((imgs.shape[0],), 0, dtype=torch.float, device=device)
        real_imgs = Variable(imgs.type(Tensor))
         
        fake_loss = loss(d(gen_imgs.detach()).view(-1), fake)
        real_loss = loss(d(real_imgs).view(-1), valid)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        opt_D.step()
        

    print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        
    
    save_image(gen_imgs.data[:25], "images_128/%d.png" % epoch, nrow=5, normalize=True)
    save_checkpoint({
        'epoch': epoch,
        'model_state_dict': g.state_dict(),
        'optimizer_state_dict': opt_G.state_dict(),
        'loss': g_loss
    },'G')
    
    save_checkpoint({
        'epoch': epoch,
        'model_state_dict': d.state_dict(),
        'optimizer_state_dict': opt_D.state_dict(),
        'loss': d_loss
    },'D')