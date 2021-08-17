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


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=5000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--load_ckpt", type=str, default=None, help="checkpoint name to continue training")
parser.add_argument("--save_ckpt", type=str, default=None, help="checkpoint name to save states")
opt = parser.parse_args()
print(opt)
lr=0.0002
b1=0.5
b2=0.999
img_shape = (opt.channels, opt.img_size, opt.img_size)





class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity
        
        
        
        
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        
        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img



g = Generator()
d = Discriminator()


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





dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "/data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)




Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor




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



for epoch in range(start_epoch, opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        
        
        
        opt_G.zero_grad()
        
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)

        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        gen_imgs = g(z)
        
        g_loss = loss(d(gen_imgs), valid)

        g_loss.backward()
        opt_G.step()



        

        opt_D.zero_grad()
        
        
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)
        real_imgs = Variable(imgs.type(Tensor))

        
        real_loss = loss(d(real_imgs), valid)
        fake_loss = loss(d(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        opt_D.step()
        
        

    print(
        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
        % (epoch, opt.n_epochs, i, len(dataloader), d_loss, g_loss)
    )

    
    save_image(gen_imgs.data[:25], "images/%d.png" % epoch, nrow=5, normalize=True)
    
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
    
    
    
