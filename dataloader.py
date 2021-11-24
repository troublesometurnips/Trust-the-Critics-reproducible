import os
import numpy as np
import torch as th
import torchvision 
import torchvision.transforms as transforms
import scipy
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, MNIST, SVHN

from infogan import Generator
"""A collection of data loaders used for training TTC and WGAN-GP"""
def fashion(args,train):

    preprocess = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(), transforms.Normalize((0.5),(0.5))])


    ds = FashionMNIST(args.data, download=True, train=train, transform=preprocess)


    dataloader = th.utils.data.DataLoader(ds,
                      batch_size = args.bs,
                      drop_last = True,
                      shuffle = True,
                      num_workers = args.num_workers,
                      pin_memory = th.cuda.is_available())

    dataloader.in_channels = 1
    dataloader.hpix = 32
    dataloader.wpix = 32

    return dataloader

def mnist(args,train):

    preprocess = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(),transforms.Normalize((0.5),(0.5))])


    ds = MNIST(args.data, download=True, train=train, transform=preprocess)


    dataloader = th.utils.data.DataLoader(ds,
                      batch_size = args.bs,
                      drop_last = True,
                      shuffle = True,
                      num_workers = args.num_workers,
                      pin_memory = th.cuda.is_available())

    dataloader.in_channels = 1
    dataloader.hpix = 32
    dataloader.wpix = 32

    return dataloader


def cifar10(args, train):
    preprocess = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    ds = CIFAR10(args.data, download = True, train = train, transform = preprocess)

    dataloader = th.utils.data.DataLoader(ds,
                    batch_size = args.bs,
                    drop_last = True,
                    shuffle = True,
                    pin_memory = th.cuda.is_available(),
                    num_workers = args.num_workers)

    dataloader.in_channels = 3
    dataloader.hpix = 32
    dataloader.wpix = 32


    return dataloader


def cifar100(args, train):
    preprocess = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    ds = CIFAR100(args.data, download = True, train = train, transform = preprocess)

    dataloader = th.utils.data.DataLoader(ds,
                    batch_size = args.bs,
                    drop_last = True,
                    shuffle = True,
                    pin_memory = th.cuda.is_available(),
                    num_workers = args.num_workers)

    dataloader.in_channels = 3
    dataloader.hpix = 32
    dataloader.wpix = 32


    return dataloader

def svhn(args, train):
    preprocess = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    ds = SVHN(args.data, download = True, split = 'train', transform = preprocess)

    dataloader = th.utils.data.DataLoader(ds,
                    batch_size = args.bs,
                    drop_last = True,
                    shuffle = True,
                    pin_memory = th.cuda.is_available(),
                    num_workers = args.num_workers)

    dataloader.in_channels = 3
    dataloader.hpix = 32
    dataloader.wpix = 32


    return dataloader

def celeba(args, train):
    h = 128
    w = 128
    preprocess = transforms.Compose([transforms.Resize((h,w)), transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    ds = torchvision.datasets.ImageFolder(args.data, transform = preprocess)

    dataloader = th.utils.data.DataLoader(ds,
                    batch_size = args.bs,
                    drop_last = True,
                    shuffle = True,
                    pin_memory = th.cuda.is_available(),
                    num_workers = args.num_workers)

    dataloader.in_channels = 3
    dataloader.hpix = h
    dataloader.wpix = w

    return dataloader

def celebaHQ(args, train):
    h = 128
    w = 128
    preprocess = transforms.Compose([transforms.Resize((h,w)), transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    split = 'celebaHQ/train' if train else 'celebaHQ/test'
    path = os.path.join(args.data, split)
    ds = torchvision.datasets.ImageFolder(path, transform = preprocess)

    dataloader = th.utils.data.DataLoader(ds,
                    batch_size = args.bs,
                    drop_last = True,
                    shuffle = True,
                    pin_memory = th.cuda.is_available(),
                    num_workers = args.num_workers)

    dataloader.in_channels = 3
    dataloader.hpix = h
    dataloader.wpix = w

    return dataloader

def noise(args, train):
    nl = NoiseLoader()
    
    nl.bs = args.bs
    nl.c = args.num_chan
    nl.h = args.hpix
    nl.w = args.wpix
    
    return nl

class NoiseLoader:
    def __iter__(self):
        return self

    def __next__(self):
        return th.randn([self.bs, self.c, self.h, self.w]), th.zeros([1]) #second is dummy variable

def unit_sphere(args, train):#generating unit norm noise. For validation
    nl = UnitSphereLoader()
    
    nl.bs = args.bs
    nl.c = args.num_chan
    nl.h = args.hpix
    nl.w = args.wpix
    
    return nl

class UnitSphereLoader:
    def __iter__(self):
        return self

    def __next__(self):
        noise = th.randn([self.bs, self.c*self.h*self.w])
        reciprocal_norms = th.norm(noise, dim = 1)**(-1)
        out = th.matmul(th.diag(reciprocal_norms), noise)
        return out.view(-1, self.c, self.h, self.w), th.zeros([1]) #second is dummy variable

def all_zero(args, train):#generating all zero data. For validation
    nl = ZeroLoader()
    
    nl.bs = args.bs
    nl.in_channels = 1
    nl.hpix = 32
    nl.wpix = 32
    
    return nl

class ZeroLoader:
    def __iter__(self):
        return self

    def __next__(self):
        return th.zeros([self.bs, self.in_channels, self.hpix, self.wpix]), th.zeros([1]) #second is dummy variable

def untrained_gen(args, train):#for training TTC from an untrained generator
    print('Note: using infogan generator')
    generator = UGENLoader()

    generator.bs = args.bs
    generator.in_c = 64 #latent dimension is 64
    generator.out_chan = args.num_chan
    generator.dim = args.dim
    generator.temp_dir = args.temp_dir

    return generator

class UGENLoader:
    def __iter__(self):
        self.ugen = Generator(self.dim, self.out_chan, 32, 32)
        self.ugen = self.ugen.cuda()
         
        self.ugen.load_state_dict(th.load(os.path.join(self.temp_dir, 'ugen.pth')))
        
        for p in self.ugen.parameters():
            p.requires_grad = False
        #self.ugen.eval()
        print('note:generator not in eval mode')
        return self

    def __next__(self):
        return self.ugen(th.randn([self.bs, self.in_c]).cuda()), th.zeros([1]) #second is dummy variable

def bsds500(args, train):
    h = 128
    w = 128
    preprocess = transforms.Compose([transforms.RandomCrop((h,w)),
                                transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    split = 'bsds500/train' if train else 'bsds500/test'
    path = os.path.join(args.data, split)
    ds = torchvision.datasets.ImageFolder(path, transform = preprocess)

    dataloader = th.utils.data.DataLoader(ds,
                    batch_size = args.bs,
                    drop_last = True,
                    shuffle = True,
                    pin_memory = th.cuda.is_available(),
                    num_workers = args.num_workers)

    dataloader.in_channels = 3
    dataloader.hpix = h
    dataloader.wpix = w

    return dataloader

def noisybsds500(args, train):
    """For adding noise to bsds500. Note that since the noise is added after
    the centering of the data to [-1,1], the effective std of the noise, for pixel values
    in [0,1], is one half sigma"""
    h = 128
    w = 128
    sigma = args.sigma

    preprocess = transforms.Compose([transforms.RandomCrop((h,w)),
                                transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), AddGaussianNoise(sigma)])

    split = 'bsds500/train' if train else 'bsds500/test'
    path = os.path.join(args.data, split)
    ds = torchvision.datasets.ImageFolder(path, transform = preprocess)

    dataloader = th.utils.data.DataLoader(ds,
                    batch_size = args.bs,
                    drop_last = True,
                    shuffle = True,
                    pin_memory = th.cuda.is_available(),
                    num_workers = args.num_workers)

    dataloader.in_channels = 3
    dataloader.hpix = h
    dataloader.wpix = w

    return dataloader

#transform adding mean zero noise to image, variance sigma^2
class AddGaussianNoise(object):
    def __init__(self,  std = 1.):
        self.std = std
    def __call__(self, tensor):
        return tensor + th.randn(tensor.size()) * self.std
    def __repr__(self):
        return self.__class__.__name__ + '(std = {})'.format(self.std)

def monet(args, train):
    h = 128
    w = 128
    print('Note: taking random crop')
    preprocess = transforms.Compose([transforms.RandomCrop((h,w)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    split = 'monet/train' if train else 'monet/test'
    path = os.path.join(args.data, split)
    ds = torchvision.datasets.ImageFolder(path, transform = preprocess)

    dataloader = th.utils.data.DataLoader(ds,
                    batch_size = args.bs,
                    drop_last = True,
                    shuffle = True,
                    pin_memory = th.cuda.is_available(),
                    num_workers = args.num_workers)

    dataloader.in_channels = 3
    dataloader.hpix = h
    dataloader.wpix = w

    return dataloader

def photo(args, train):
    h = 128
    w = 128
    preprocess = transforms.Compose([transforms.Resize((h,w)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    split = 'photo/train' if train else 'photo/test'
    path = os.path.join(args.data, split)
    ds = torchvision.datasets.ImageFolder(path, transform = preprocess)

    dataloader = th.utils.data.DataLoader(ds,
                    batch_size = args.bs,
                    drop_last = True,
                    shuffle = True,
                    pin_memory = th.cuda.is_available(),
                    num_workers = args.num_workers)

    dataloader.in_channels = 3
    dataloader.hpix = h
    dataloader.wpix = w

    return dataloader
    
