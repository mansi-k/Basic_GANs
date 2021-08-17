
import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

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
    if class_name.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif class_name.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        

        self.model = nn.Sequential(
    
            nn.Conv2d(opt.channels, 16, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Dropout2d(0.25),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(32, 0.8),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(64, 0.8),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(128, 0.8),
        )

        
        n = int(opt.img_size/16)
        
        self.model_ = nn.Sequential(nn.Linear(128 * n * n, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.model_(out)

        return validity

















class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.n = int(opt.img_size / 4)
        self.model = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.n * self.n))

        self.model_ = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.ReLU(True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.model(z)
        out = out.view(out.shape[0], 128, self.n, self.n)
        img = self.model_(out)
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
opt_D = torch.optim.Adam(d.parameters(), lr=lr, betas=(b1, b2))
    
    
    
    
ckpt = 'checkpoints'

def save_checkpoint(state, model_type):
    torch.save(state, ckpt+"/"+model_type+"_"+opt.save_ckpt)





Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor




dataloader = torch.utils.data.DataLoader(
    datasets.EMNIST(
        "data/emnist",
        train=True,
        split='letters',
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)





start_epoch = 0

if opt.load_ckpt:
    g_ckpt = torch.load(ckpt+'/G_'+opt.load_ckpt)
    d_ckpt = torch.load(ckpt+'/D_'+opt.load_ckpt)
    assert g_ckpt['epoch']==d_ckpt['epoch'] , "different epochs"
    g.load_state_dict(g_ckpt['model_state_dict'])
    opt_G.load_state_dict(g_ckpt['optimizer_state_dict'])
    g_loss = g_ckpt['loss']
    
    d.load_state_dict(d_ckpt['model_state_dict'])
    opt_D.load_state_dict(d_ckpt['optimizer_state_dict'])
    d_loss = d_ckpt['loss']
    
    start_epoch = g_ckpt['epoch']+1









for epoch in range(start_epoch,opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        
        opt_G.zero_grad()

        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)

        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        gen_imgs = g(z)
        g_loss = loss(d(gen_imgs), valid)
        g_loss.backward()
        opt_G.step()
        
        
        
        real_imgs = Variable(imgs.type(Tensor))
        
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        opt_D.zero_grad()

        
        real_loss = loss(d(real_imgs), valid)
        fake_loss = loss(d(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        opt_D.step()
        
        

    print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        
    
    save_image(gen_imgs.data[:25], "images/%d.png" % epoch, nrow=5, normalize=True)
    save_checkpoint({
        'epoch': epoch,
        'model_state_dict': g.state_dict(),
        'optimizer_state_dict': opt_G.state_dict(),
        'loss': g_loss,
        
    },'G')
    
    save_checkpoint({
        'epoch': epoch,
        'model_state_dict': d.state_dict(),
        'optimizer_state_dict': opt_D.state_dict(),
        'loss': d_loss,
        
    },'D')



