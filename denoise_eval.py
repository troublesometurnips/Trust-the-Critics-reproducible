#Code for evaluating the effectiveness of denoising using ttc
#Loads 128x128 crops of entire bsds500 test set, adds noise, restores, and computes PSNR

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
import networks
from generate_samples import generate_image
from generate_samples import save_individuals
from steptaker import steptaker


#get command line args~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

parser = argparse.ArgumentParser('Code for evaluating denoising performance')
parser.add_argument('--temp_dir', type=str, required=True, help = 'directory where model state dicts and log are located. Everything saved here.')
parser.add_argument('--data', type=str, required=True, help = 'directory where data is located')
parser.add_argument('--model', type=str, default='arConvNet', choices=['dcgan', 'arConvNet', 'sndcgan'])
parser.add_argument('--dim', type=int, default=64, help = 'int determining network dimensions')
parser.add_argument('--seed', type=int, default=-1, help = 'Set random seed for reproducibility')
parser.add_argument('--bs', type=int, default=200, help = 'batch size')
parser.add_argument('--sigma', type=float, default=0.2, help = 'noise std. Effective value is one half input')
parser.add_argument('--num_workers', type=int, default = 0, help = 'number of data loader processes')
parser.add_argument('--num_step', type=int, default = 200, help = 'number of steps for benchmark (adversarial regularization)')
parser.add_argument('--stepsize', type=float, default = 0.05, help = 'gradient descent step size for benchmark (adversarial regularization)')

args = parser.parse_args()

temp_dir = args.temp_dir#directory for temp saving

num_crit = len(os.listdir(os.path.join(temp_dir,'model_dicts')))#number of critics

#code to get deterministic behaviour
if args.seed != -1: #if non-default seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark=False #If true, optimizes convolution for hardware, but gives non-deterministic behaviour
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

#get dataloader
target_loader = getattr(dataloader, 'bsds500')(args, train=False)

args.num_chan = target_loader.in_channels
args.hpix = target_loader.hpix
args.wpix = target_loader.wpix
#begin definitions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

critic_list = [None]*num_crit

for i in range(num_crit):#initialize pre-trained critics
    critic_list[i] = getattr(networks, args.model)(args.dim, args.num_chan, args.hpix, args.wpix)
    critic_list[i].load_state_dict(torch.load(os.path.join(temp_dir,'model_dicts','critic{}.pth'.format(i))))


#Extract list of steps from log file
log = pd.read_pickle(os.path.join(temp_dir,'log.pkl'))

steps_d = log['steps']
steps = []
for key in steps_d.keys():
    steps.append(steps_d[key])


print('Arguments:')
for p in vars(args).items():
    print('  ',p[0]+': ',p[1])
print('\n')

use_cuda = torch.cuda.is_available()

if use_cuda:
    for i in range(num_crit):
        critic_list[i] = critic_list[i].cuda()

def psnr_calc(noisy, real):
    """Calculates PSNR between noisy and real data
    Inputs
    - noisy; batch of noisy data
    - real; batch of clean data, in correspondence with noisy
    Outputs
    - psnrs; full list of psnrs
    - mean and std of list of psnrs
    """
    numpix = noisy.size(1)*noisy.size(2)*noisy.size(3)
    bs = noisy.size(0)
    avg_sq_norm = (1/numpix)*torch.norm(0.5*(noisy.view(bs, -1)- real.view(bs,-1)), dim = 1)**2#multiplication by 0.5 because vals between [-1,1]
    psnrs = -10*torch.log10(avg_sq_norm)
    return psnrs, torch.tensor([torch.mean(psnrs), torch.std(psnrs)])

def adv_reg(noisy_img, critic, num_step, mu, stepsize):
    """Implementation of adversarial regularization (benchmark denoiser), which solves a backward euler problem
    via gradient descent
    Inputs
    - noisy_img; batch of noisy images to be restored
    - critic; trained critic to be used as a learned regularizer
    - num_step; how many steps to do in gradient descent for restoration
    - mu; coefficient for learned regularizer. Computed from noise statistics
    - stepsize; stepsize for gradient descent algorithm
    Outputs
    - Restored batch of images
    """
    observation = noisy_img.detach().clone()#initial noisy observation
    for i in range(num_step):
        noisy_img = (steptaker(noisy_img, critic, mu*stepsize) - 2*stepsize*(noisy_img-observation)).detach().clone()

    return noisy_img

def find_good_mu(noise_mag):
    """computes the regularization parameter in adversarial regularizer according to their heuristic"""
    return 2*torch.mean(noise_mag)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Generate a noise image, and flow it according to the critics
gen = iter(target_loader)#make target_loader into iterable

datasize = len(os.listdir(os.path.join(temp_dir,'bsds500','test','images')))#number of pics in test data
print('total datasize is {}. Will be missing batches unless this equals args.bs'.format(datasize))

starttime = time.time()

psnrvals = torch.zeros(1 + num_crit, 2).cuda()#in the form mean, std

num_samp = min(args.bs, 200)
norm_diffs = torch.zeros([datasize]).cuda()#for computing coefficient for adv reg. technique

real = next(gen)[0]
real = real.cuda()
noisy = real + args.sigma*torch.randn_like(real)
noisy_bench = noisy.detach().clone()#the version of the noisy data to be restored with adv. reg.
    
generate_image('00',noisy[0:num_samp,:,:,:].detach().cpu(), 'pdf', temp_dir)#make pictures of noisy images
generate_image('clean', real[0:num_samp,:,:,:].detach().cpu(), 'pdf', temp_dir)#and clean images

full_psnrs, psnrvals[0, :] = psnr_calc(noisy, real)#initial psnrs
index = 2#index for making a full res picture of a particular image
print('original psnrs {}'.format(full_psnrs[index]))
for i in range(num_crit):#apply the steps of TTC
    eps = torch.tensor(steps[i]).cuda()#current step
    noisy = steptaker(noisy, critic_list[i], eps)

    generate_image(i, noisy[0:num_samp,:,:,:].detach().cpu(), 'pdf', temp_dir)

    full_psnrs, psnrvals[i+1, :] = psnr_calc(noisy, real)

norm_diffs = torch.norm((real-noisy_bench).view(args.bs,-1), dim = 1)    

#apply adversarial regularization to restore noisy_bench and compute psnr
mu = find_good_mu(norm_diffs)
print('mu is {}'.format(mu))
restored_bench = adv_reg(noisy_bench, critic_list[0], args.num_step, mu, args.stepsize)
generate_image('adv_reg', restored_bench[0:num_samp,:,:,:].detach().cpu(), 'pdf', temp_dir)
full_psnrs_bench, psnr_bench = psnr_calc(restored_bench, real)

    

print('Done going through test data')

print('time required: {}'.format(time.time() - starttime))
#save psnr vals
metrics = {}

metrics['psnr'] = np.array(psnrvals[:,0].detach().cpu())
metrics['psnr_std'] = np.array(psnrvals[:,1].detach().cpu())
metrics['psnr_bench'] = np.array(psnr_bench[0].detach().cpu())
metrics['psnr_bench_std'] = np.array(psnr_bench[1].detach().cpu())

for i in range(num_crit+1):
    print('PSNR for {}th critic is {:.2f}, std is {:.2f}'.format(i, psnrvals[i,0], psnrvals[i,1]))

print('Final PSNR for benchmark {:.2f}, std is {:.2f}'.format(psnr_bench[0], psnr_bench[1]))

psnr_delta = full_psnrs - full_psnrs_bench
print('psnrs for TTC {}'.format(full_psnrs[index]))
print('psnrs for adv. reg. {}'.format(full_psnrs_bench[index]))

generate_image('deer', torch.stack([real[index, :,:,:], noisy_bench[index, :, :, :], restored_bench[index, :, :, :], noisy[index, :, :,:]]).detach().cpu(), 'pdf', temp_dir)


num_fails = torch.count_nonzero(psnr_delta - torch.abs(psnr_delta))
print('Benchmark was better than TTC on {} examples'.format(num_fails))

with open(temp_dir + '/metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f, pickle.HIGHEST_PROTOCOL)
