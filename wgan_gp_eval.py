#Computes MMD and FID for a list of generators trained with wgan-gp

import os, sys
sys.path.append(os.getcwd())

import argparse
import time
import log
import shutil
import random

import numpy as np

import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim
from torchvision import utils
import pandas as pd
import pickle
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import numpy as np

import dataloader
from generate_samples import generate_image
from generate_samples import save_individuals
from mmd import mmd
from pytorch_fid import fid_score

#get command line args~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

parser = argparse.ArgumentParser('Evaluation code for a list of generators')
parser.add_argument('--temp_dir', type=str, required=True, help = 'directory where model state dicts are located')
parser.add_argument('--data', type=str, required=True, help = 'directory where data is located')
parser.add_argument('--dim', type=int, default=64, help = 'int determining network dimensions')
parser.add_argument('--model', type=str, default='dcgan', choices=['dcgan', 'infogan'])
parser.add_argument('--seed', type=int, default=-1, help = 'Set random seed for reproducibility')
parser.add_argument('--bs', type=int, default=128, help = 'batch size')
parser.add_argument('--num_workers', type=int, default = 0, help = 'number of data loader processes')
parser.add_argument('--target', type=str, required=True, help = 'target distribution. options = {mnist, fashion, cifar10, celebaHQ}')
parser.add_argument('--MMD', action= 'store_true', help = 'compute the MMD between args.bs samples of updated source and target')
parser.add_argument('--FID', action= 'store_true', help = 'compute the FID between 10000 generated examples and test set for each generator.')
parser.add_argument('--bigsample', action= 'store_true', help = 'Generate 10,000 images for use in computing FID')
parser.add_argument('--eval_freq', type=int, default = 5, help = 'frequency of MMD/FID evaluation')
args = parser.parse_args()

temp_dir = args.temp_dir#directory for temp saving

num_gen = len(os.listdir(os.path.join(temp_dir,'model_dicts')))#number of generators to evaluate. Includes initial untrained generator

#code to get deterministic behaviour
if args.seed != -1: #if non-default seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark=False #If true, optimizes convolution for hardware, but gives non-deterministic behaviour
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


#begin definitions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
target_loader = getattr(dataloader, args.target)(args, train=False)#test data is used
tgen = iter(target_loader)#only relevant to MMD computation

num_chan = target_loader.in_channels
hpix = target_loader.hpix
wpix = target_loader.wpix


gen_list = [None]*num_gen

if args.model == 'dcgan':
    from dcgan import Generator
elif args.model == 'infogan':
    from infogan import Generator

for i in range(num_gen):
    gen_list[i] = Generator(args.dim, num_chan, hpix, wpix)
    gen_list[i].load_state_dict(torch.load(os.path.join(temp_dir,'model_dicts','generator{}.pth'.format(i))))#load pre-trained generators
        

print('Arguments:')
for p in vars(args).items():
    print('  ',p[0]+': ',p[1])
print('\n')

use_cuda = torch.cuda.is_available()

if use_cuda:
    for i in range(num_gen):
        gen_list[i] = gen_list[i].cuda()



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Sample a noise vector, and apply the sequence of generators to it.

if args.bigsample:
    num_batch = 3000//args.bs if args.target == 'celebaHQ' else 10000//args.bs 
else:
    num_batch = 1

latent_dim = 64 if args.model == 'infogan' else 128

starttime = time.time()

if args.MMD:
    mmdvals = torch.zeros(1+(num_gen-1)//args.eval_freq, num_batch).cuda()

#reset seed to get same noise sequence
#code to get deterministic behaviour
if args.seed != -1: #if non-default seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark=False #If true, optimizes convolution for hardware, but gives non-deterministic behaviour
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

for b_idx in tqdm(range(num_batch)):
    
    noise = torch.randn(args.bs, latent_dim).cuda()

    if args.MMD:#if computing MMD, gather a minibatch of target data
        tbatch = next(tgen)[0]
        tbatch = tbatch.cuda()
   

    for i in range(0, num_gen, args.eval_freq):#generate the data for the different trained generators.
        fake = gen_list[i](noise)
        if b_idx == 0:
            generate_image(i, fake[0:64,:,:,:].detach().cpu(), 'pdf', temp_dir)#visualize samples
        if args.MMD:
            mmdvals[i//args.eval_freq, b_idx] = mmd(fake.view(args.bs, -1), tbatch.view(args.bs,-1), alph = [10**n for n in range(3,-11,-1)])
        if args.bigsample:
            save_individuals(b_idx, i, fake, 'jpg', temp_dir, to_rgb = True if num_chan ==1 else False)

if args.bigsample:#save zip file of pics
    shutil.make_archive(os.path.join(temp_dir,'pics'), 'zip', os.path.join(temp_dir,'pics'))

print('time required: {}'.format(time.time() - starttime))
#save mmd vals
if args.MMD or args.FID:
    metrics = {}

if args.MMD:
    metrics['mmd'] = np.array(mmdvals.detach().cpu())
    print(mmdvals)

#compute FID and save
if args.FID:
    fidvals = torch.zeros(1+(num_gen-1)//args.eval_freq).cuda()
    for g_idx in range(1+(num_gen-1)//args.eval_freq):
        gen_num = g_idx*args.eval_freq

        paths = [os.path.join(temp_dir, 'pics/timestamp{}'.format(gen_num)), os.path.join(temp_dir, args.target + 'test')]
        fidvals[g_idx] = fid_score.calculate_fid_given_paths(paths, 100, torch.device('cuda'), 2048)
    metrics['fid'] = np.array(fidvals.detach().cpu())
    print(fidvals)

if args.MMD or args.FID:
    with open(temp_dir + '/metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f, pickle.HIGHEST_PROTOCOL)
