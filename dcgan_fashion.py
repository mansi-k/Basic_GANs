
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
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--load_ckpt", type=str, default=None, help="checkpoint name to continue training")
parser.add_argument("--save_ckpt", type=str, default=None, help="checkpoint name to save states")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False





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


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_feat, out_feat, normalize=True):
            block = [nn.Conv2d(in_feat, out_feat, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if normalize:
                block.append(nn.BatchNorm2d(out_feat, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, normalize=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        
        n = int(opt.img_size/16)
        self.model_ = nn.Sequential(nn.Linear(128 * n * n, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.model_(out)

        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    
    
ckpt_dir = 'checkpoints'

def save_checkpoint(state, model_type):
    torch.save(state, ckpt_dir+"/"+model_type+"_"+opt.save_ckpt)




# Configure data loader
os.makedirs("data/fashionmnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.FashionMNIST(
        "data/fashionmnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))



Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


start_epoch = 0
if opt.load_ckpt:
    g_ckpt = torch.load(ckpt_dir+'/G_'+opt.load_ckpt)
    d_ckpt = torch.load(ckpt_dir+'/D_'+opt.load_ckpt)
    assert g_ckpt['epoch']==d_ckpt['epoch'] , "different epochs"
    generator.load_state_dict(g_ckpt['model_state_dict'])
    optimizer_G.load_state_dict(g_ckpt['optimizer_state_dict'])
    g_loss = g_ckpt['loss']
    discriminator.load_state_dict(d_ckpt['model_state_dict'])
    optimizer_D.load_state_dict(d_ckpt['optimizer_state_dict'])
    d_loss = d_ckpt['loss']
    start_epoch = g_ckpt['epoch']+1
    
    
# ----------
#  Training
# ----------

for epoch in range(start_epoch,opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

    print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        
    
    save_image(gen_imgs.data[:25], "images/%d.png" % epoch, nrow=5, normalize=True)
    save_checkpoint({
        'epoch': epoch,
        'model_state_dict': generator.state_dict(),
        'optimizer_state_dict': optimizer_G.state_dict(),
        'loss': g_loss
    },'G')
    
    save_checkpoint({
        'epoch': epoch,
        'model_state_dict': discriminator.state_dict(),
        'optimizer_state_dict': optimizer_D.state_dict(),
        'loss': d_loss
    },'D')