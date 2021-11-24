"""takes a list of critics and step sizes, and produces a sequence of pictures using TTC
Optional evaluation of FID and MMD (the latter not used in paper)"""

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
from mmd import mmd
from steptaker import steptaker
from pytorch_fid import fid_score


#get command line args~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

parser = argparse.ArgumentParser('TTC Evaluation Code')
parser.add_argument('--target', type=str, default='cifar10', choices=['cifar10','mnist','fashion', 'celeba', 'monet', 'celebaHQ'])
parser.add_argument('--source', type=str, default='cifar10', choices=['noise', 'untrained_gen', 'photo'])
parser.add_argument('--temp_dir', type=str, required=True, help = 'directory where model state dicts are located')
parser.add_argument('--data', type=str, required=True, help = 'directory where data is located')
parser.add_argument('--model', type=str, default='dcgan', choices=['dcgan', 'infogan', 'arConvNet', 'sndcgan','bsndcgan'])
parser.add_argument('--dim', type=int, default=64, help = 'int determining network dimensions')
parser.add_argument('--seed', type=int, default=-1, help = 'Set random seed for reproducibility')
parser.add_argument('--bs', type=int, default=128, help = 'batch size')
parser.add_argument('--num_workers', type=int, default = 0, help = 'number of data loader processes')
parser.add_argument('--MMD', action= 'store_true', help = 'compute the MMD between args.bs samples of updated source and target')
parser.add_argument('--FID', action= 'store_true', help = 'compute the FID between generated examples and test set for each generator.')
parser.add_argument('--numsample', type=int, default = 0, help = 'how many pics to generate for FID')
parser.add_argument('--eval_freq', type=int, default = 5, help = 'frequency of MMD/FID evaluation')
parser.add_argument('--num_step', type=int, default=1, help = 'how many steps to use in gradient descent')
parser.add_argument('--commonfake', action= 'store_true', help = 'Use if you want a common source element to compare two models')
parser.add_argument('--translate_eval', type=int, default = -1, help = 'index for particular image you want translated')
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
target_loader = getattr(dataloader, args.target)(args, train=False)

args.num_chan = target_loader.in_channels
args.hpix = target_loader.hpix
args.wpix = target_loader.wpix

source_loader = getattr(dataloader, args.source)(args, train=False)

if args.commonfake:
    gen = iter(source_loader)
    commonfake = next(gen)[0]

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



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Sample source images and update them according to the critics and step sizes
gen = iter(source_loader)#make dataloaders into iterables
tgen = iter(target_loader)

if args.numsample>0:
    num_batch = args.numsample//args.bs
else:
    num_batch = 1


starttime = time.time()

if args.MMD:
    mmdvals = torch.zeros(1+(num_crit//args.eval_freq), num_batch).cuda()#where MMD measurements will be stored

num_samp = min(args.bs, 128)
print('Using max num_samp = 128')

#repeating seed selection again here to get same noise sequence
#code to get deterministic behaviour
if args.seed != -1: #if non-default seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark=False #If true, optimizes convolution for hardware, but gives non-deterministic behaviour
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

for b_idx in tqdm(range(num_batch)):
    
    fake = next(gen)[0]

    if args.commonfake:
        fake = commonfake


    fake = fake.cuda()
    if args.translate_eval >= 0:
        indices = [0, args.translate_eval]#code requires minibatch size of > 1
        fake = fake[indices,:,:,:]
        orig = fake.detach().clone()
    
    if args.MMD:
        tbatch = next(tgen)[0]#target data only necessary here if computing MMD
        tbatch = tbatch.cuda()

    
    if b_idx ==0:
        generate_image('00',fake[0:num_samp,:,:,:].detach().cpu(), 'jpg', temp_dir)#visualize initial images

    if args.MMD:#record MMD for training phase 0 and minibatch b_idx
        mmdvals[0, b_idx] = mmd(fake.view(args.bs, -1), tbatch.view(args.bs,-1), alph = [10**n for n in range(3,-11,-1)])

    #save minibatch if doing FID computation
    if args.numsample>0:
        save_individuals(b_idx, 0, fake, 'jpg', temp_dir, to_rgb = True if args.num_chan == 1 else False)


    for i in range(num_crit):#apply the steps of TTC
        eps = torch.tensor(steps[i]).cuda()
        fake = steptaker(fake, critic_list[i], eps, num_step = args.num_step)


        if ((i+1) % args.eval_freq == 0):

            if b_idx == 0:#only visualize if on the first batch
                if args.translate_eval >= 0:
                    img_bank  = torch.stack((orig[1,:,:,:], fake[1,:,:,:]))#puts orig and fake next to each other
                    generate_image(i, img_bank.detach().cpu(), 'pdf', temp_dir)
                else:
                    generate_image(i, fake[0:num_samp, :,:,:].detach().cpu(), 'pdf', temp_dir)
            
            if args.MMD:#record MMD for training phase i and minibatch b_idx
                mmdvals[(i+1)//args.eval_freq, b_idx] = mmd(fake.view(args.bs, -1), tbatch.view(args.bs,-1), alph = [10**n for n in range(3,-11,-1)])

            #save minibatch if doing big sample
            if args.numsample>0:
                save_individuals(b_idx, i+1, fake, 'jpg', temp_dir, to_rgb = True if args.num_chan ==1 else False)

if args.numsample>0:#save zip file of pics
    shutil.make_archive(os.path.join(temp_dir,'pics'), 'zip', os.path.join(temp_dir,'pics'))

print('time required: {}'.format(time.time() - starttime))
if args.MMD or args.FID:
    metrics = {}

#save mmd vals
if args.MMD:
    metrics['mmd'] = np.array(mmdvals.detach().cpu())
    print(mmdvals)

#compute FID and save
if args.FID:
    fidvals = torch.zeros(1+(num_crit//args.eval_freq)).cuda()
    for c_idx in range(1+num_crit//args.eval_freq):
        crit_num = c_idx*args.eval_freq
        paths = [os.path.join(temp_dir, 'pics/timestamp{}'.format(crit_num)), os.path.join(temp_dir, args.target + 'test')]
        fidvals[c_idx] = fid_score.calculate_fid_given_paths(paths, 100, torch.device('cuda'), 2048)
    metrics['fid'] = np.array(fidvals.detach().cpu())
    print(fidvals)

if args.MMD or args.FID:
    with open(temp_dir + '/metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f, pickle.HIGHEST_PROTOCOL)#save metrics
