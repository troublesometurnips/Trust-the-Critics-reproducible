#Code for training baseline wgan_gp on infogan
#Initially built from www.github.com/caogang/wgan-gp

import os, sys
sys.path.append(os.getcwd())

import argparse
import time
import log
import json
import random

import numpy as np
import pandas as pd

import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim

import dataloader
from infogan import Generator
from infogan import Discriminator
from generate_samples import generate_image
from calc_gradient_penalty import calc_gradient_penalty
from get_data import get_data

#get command line args~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

parser = argparse.ArgumentParser('Testbed for WGAN Training')
parser.add_argument('--data', type=str, required=True, help = 'directory where data is located')
parser.add_argument('--target', type=str, default='cifar10', choices=['cifar10','mnist','fashion'])
parser.add_argument('--temp_dir', type=str, required=True, help = 'temporary directory for saving')
parser.add_argument('--dim', type=int, default=64, help = 'int determining network dimensions')
parser.add_argument('--seed', type=int, default=-1, help = 'Set random seed for reproducability')
parser.add_argument('--lamb', type=float, default=1000., help = 'parameter multiplying gradient penalty')
parser.add_argument('--critters', type=int, default=5, help = 'number of critic iters per gen update')
parser.add_argument('--bs', type=int, default=128, help = 'batch size')
parser.add_argument('--iters', type=int, default=100000, help = 'number of generator updates')
parser.add_argument('--plus', action= 'store_true', help = 'take one sided penalty')
parser.add_argument('--num_workers', type=int, default = 0, help = 'number of data loader processes')
parser.add_argument('--beta_1', type=float, default=0.5, help = 'beta_1 for Adam')
parser.add_argument('--beta_2', type=float, default=0.999, help = 'beta_2 for Adam')
parser.add_argument('--glr', type=float, default=1e-4, help = 'generator learning rate')

args = parser.parse_args()

#code to get deterministic behaviour
if args.seed != -1: #if non-default seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark=False #If true, optimizes convolution for hardware, but gives non-deterministic behaviour
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

#save args in config file
config_file_name = os.path.join(args.temp_dir, 'train_config.txt')
with open(config_file_name, 'w') as f:
    json.dump(args.__dict__, f, indent=2)

########################
# Dataset iterator
########################

loader = getattr(dataloader, args.target)(args, train=True)
num_chan = loader.in_channels
hpix = loader.hpix
wpix = loader.wpix

iterator = iter(loader)

########################
#Initialize Generator, Discriminator, and optimizers 
########################

netG = Generator(args.dim, num_chan, hpix, wpix)
netD = Discriminator(args.dim, num_chan, hpix, wpix)
print(netG)
print(netD)

os.makedirs(os.path.join(args.temp_dir,'model_dicts'), exist_ok = True)
torch.save(netG.state_dict(), os.path.join(args.temp_dir,'model_dicts','generator0.pth'))#save untrained generator for getting baseline FID/MMD

print('Arguments:')
for p in vars(args).items():
    print('  ',p[0]+': ',p[1])
print('\n')

use_cuda = torch.cuda.is_available()

if use_cuda:
    netD = netD.cuda()
    netG = netG.cuda()


optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(args.beta_1, args.beta_2))
optimizerG = optim.Adam(netG.parameters(), lr=args.glr, betas=(args.beta_1, args.beta_2))

#######################
#main training loop
#######################

start_time = time.time()
gen_idx = 0

for iteration in range(args.iters):
    ############################
    # (1) Update D network
    ###########################

    for p in netD.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update
    for i in range(args.critters):
        
        real, iterator = get_data(iterator, loader)

        for param in netD.parameters():#more efficient than netD.zero_grad()
            param.grad = None#zeros gradients

        D_real = netD(real)
        D_real = D_real.mean()
        D_real.backward()

        #generate fake data
        noise = torch.randn(args.bs, 64)#size of latent dim for infogan
        if use_cuda:
            noise = noise.cuda()
        fake = netG(noise).detach() #must detach from computational graph, otherwise grads w.r.t Gen parameters would be computed
        
        D_fake = netD(fake)
        D_fake = -D_fake.mean()
        D_fake.backward()
        
        #compute gradient penalty
        gradient_penalty = calc_gradient_penalty(netD, real, fake, args.lamb, plus = args.plus)
        gradient_penalty.backward()
 
        D_cost = D_real + D_fake + gradient_penalty#D_fake has -1 baked in.
        D_cost_nopen = D_real + D_fake
        optimizerD.step()

    ############################
    # (2) Update G network
    ###########################
    for p in netD.parameters():
        p.requires_grad = False  # to avoid computation
        
    for param in netG.parameters():
        param.grad = None #more efficient than netG.zero_grad()

    noise = torch.randn(args.bs, 64)
    if use_cuda:
        noise = noise.cuda()
    
    fake = netG(noise)
    G = netD(fake)

    G = G.mean()
    G.backward()
    G_cost = G
    optimizerG.step()

    # Record data
    log.plot('dcost', D_cost.cpu().data.numpy())#
    log.plot('time', time.time() - start_time)
    log.plot('gcost', G_cost.cpu().data.numpy())
    log.plot('no_gpen', D_cost_nopen.cpu().data.numpy())

    # Save logs every 100 iters
    if (iteration < 5) or (iteration % 100 ==99): 
        log.flush(args.temp_dir)

    log.tick()

    if iteration % (args.iters//10) == (args.iters//10 -1):
        gen_idx += 1
        print('At iteration {}. Saving the {}th generator'.format(iteration + 1, gen_idx))
        torch.save(netG.state_dict(), os.path.join(args.temp_dir,'model_dicts','generator{}.pth'.format(gen_idx)))
        
