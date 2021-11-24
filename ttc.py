"""Cleaned up code for training Trust The Critics (TTC) scheme"""

import os, sys
sys.path.append(os.getcwd())

import argparse
import time
import log
import copy
import json
import random

import numpy as np
import pandas as pd
import pickle

import torch
from torch import nn
from torch import autograd
from torch import optim

import dataloader
import networks
from get_training_time import write_training_time
from generate_samples import generate_image
from critic_trainer import critic_trainer

#################
#Get command line args
#################
parser = argparse.ArgumentParser('Training code for TTC')
parser.add_argument('--target', type=str, default='mnist', choices=['cifar10','mnist','fashion', 'celeba', 'bsds500', 'monet', 'celebaHQ', 'all_zero'])
parser.add_argument('--source', type=str, default='noise', choices=['noise', 'untrained_gen', 'noisybsds500', 'photo', 'unit_sphere'])
parser.add_argument('--model', type=str, default='dcgan', choices=['dcgan', 'infogan', 'arConvNet', 'sndcgan','bsndcgan', 'norm_taker'])
parser.add_argument('--temp_dir', type=str, required=True, help = 'temporary directory for saving')
parser.add_argument('--data', type=str, required=True, help = 'directory where data is located')
parser.add_argument('--dim', type=int, default=64, help = 'int determining network dimensions')
parser.add_argument('--seed', type=int, default=-1, help = 'Set random seed for reproducibility')
parser.add_argument('--lamb', type=float, default=1000., help = 'parameter multiplying gradient penalty')
parser.add_argument('--theta', type=float, default=0.5, help = 'parameter determining step size as fraction of W1 distance')
parser.add_argument('--sigma', type=float, default=0.02, help = 'std of noise. Only relevant if doing denoising. Effective value 1/2 of this')
parser.add_argument('--critters', type=int, default=5, help = 'number of critic iters')
parser.add_argument('--bs', type=int, default=128, help = 'batch size')
parser.add_argument('--plus', action= 'store_true', help = 'take one sided penalty')
parser.add_argument('--num_workers', type=int, default = 0, help = 'number of data loader processes')
parser.add_argument('--num_crit', type=int, default = 5, help = 'number of critics to train')
parser.add_argument('--num_step', type=int, default=1, help = 'how many steps to use in gradient descent')
parser.add_argument('--beta_1', type=float, default=0.5, help = 'beta_1 for Adam')
parser.add_argument('--beta_2', type=float, default=0.999, help = 'beta_2 for Adam')

args = parser.parse_args()

use_cuda = torch.cuda.is_available()

print('Arguments:')
for p in vars(args).items():
    print('  ',p[0]+': ',p[1])
print('\n')

#code to get deterministic behaviour
if args.seed != -1: #if non-default seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark=False #If true, optimizes convolution for hardware, but gives non-deterministic behaviour
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
else:
    torch.backends.cudnn.benchmark=True
    print('using benchmark')

#begin definitions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

###################
# Initialize Dataset iterators
###################

target_loader = getattr(dataloader, args.target)(args, train=True)

args.num_chan = target_loader.in_channels
args.hpix = target_loader.hpix
args.wpix = target_loader.wpix

source_loader = getattr(dataloader, args.source)(args, train=True)
###################
#save args in config file
###################

config_file_name = os.path.join(args.temp_dir, 'train_config.txt')
with open(config_file_name, 'w') as f:
    json.dump(args.__dict__, f, indent=2)
    
####################
#Initialize networks and optimizers
####################

critic_list = [None]*args.num_crit
steps = [1]*args.num_crit

for i in range(args.num_crit):
    critic_list[i] = getattr(networks, args.model)(args.dim, args.num_chan, args.hpix, args.wpix)

if use_cuda:
    for i in range(args.num_crit):
        critic_list[i] = critic_list[i].cuda()



optimizer_list = [None]*args.num_crit
print('Adam parameters are {} and {}'.format(args.beta_1, args.beta_2))
for i in range(args.num_crit):
    optimizer_list[i] = optim.Adam(critic_list[i].parameters(), lr=1e-4, betas=(args.beta_1, args.beta_2))


abs_start = time.time()
#main training loop
for iteration in range(args.num_crit):
    ############################
    # (1) Train D network
    ###########################
    #trains critic at critic_list[iteration], and reports current W1 distance
    critic_list, Wasserstein_D = critic_trainer(critic_list, optimizer_list, iteration, steps, target_loader, source_loader, args)

    ###########################
    # (2) Pick step size
    ###########################

    steps[iteration] = args.theta*Wasserstein_D.detach()

    ###########################
    # (3) freeze critic and save
    ###########################
    
    for p in critic_list[iteration].parameters():
        p.requires_grad = False  # this critic is now fixed

    if iteration< args.num_crit -1:
        critic_list[iteration+1].load_state_dict(critic_list[iteration].state_dict())#initialize next critic at current critic
            
    log.plot('steps', steps[iteration].cpu().data.numpy())
    log.flush(args.temp_dir)
    log.tick()


    #Save critic model dicts
    if (iteration % 10 == 9) or (iteration ==args.num_crit -1):
        os.makedirs(os.path.join(args.temp_dir,'model_dicts'), exist_ok = True)

        for j in range(iteration+1):
            torch.save(critic_list[j].state_dict(), os.path.join(args.temp_dir,'model_dicts','critic{}.pth'.format(j)))

print(steps)
write_training_time(args)
