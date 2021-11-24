"""Zoo of discriminator models for use with TTC"""

import torch
from torch import nn

##################
#Simple convolutional block, used by several networks
##################

class ConvBlock(nn.Module):
    def __init__(self, chan_in = 3, chan_out = 32, ker_size = 3, stride = 1, pad = 1):
        super(ConvBlock, self).__init__()

        self.main = nn.Sequential(nn.Conv2d(chan_in, chan_out, ker_size, stride, padding = pad),
                            nn.LeakyReLU(negative_slope = 0.1))
    def forward(self, input):
        return self.main(input)
        
##################
#Pytorch version of "ConvNetClassifier" from Lunz et. al. Adversarial Regularizers in Inverse Problems
##################

class arConvNet(nn.Module):
    def __init__(self, DIM, num_chan, h, w):
        super(arConvNet, self).__init__()
        self.h = h
        self.w = w

        conv1 =  ConvBlock(chan_in = num_chan, chan_out =16, ker_size = 5, stride = 1, pad = 2)#produces 16xhxw
        conv2 =  ConvBlock(chan_in = 16, chan_out =32, ker_size = 5, stride = 1, pad = 2)#produces 32xhxw
        conv3 =  ConvBlock(chan_in = 32, chan_out =32, ker_size = 5, stride = 2, pad = 2)#produces 32xh/2xw/2
        conv4 =  ConvBlock(chan_in = 32, chan_out =64, ker_size = 5, stride = 2, pad = 2)#produces 64xh/4xw/4
        conv5 =  ConvBlock(chan_in = 64, chan_out =64, ker_size = 5, stride = 2, pad = 2)#produces 64xh/8xw/8
        conv6 =  ConvBlock(chan_in = 64, chan_out =128, ker_size = 5, stride = 2, pad = 2)#produces 128xh/16xw/16
        
        self.main = nn.Sequential(conv1,conv2,conv3,conv4,conv5,conv6)
                    
        self.linear1 = nn.Sequential(nn.Linear(128*(h//16)*(w//16), 256), nn.LeakyReLU(negative_slope = 0.1))
        self.linear2 = nn.Linear(256,1)

    def forward(self, input):

        output = self.main(input)
        output = output.view(-1, 128*(self.h//16)*(self.w//16))
        output = self.linear1(output)
        output = self.linear2(output)
        return output

################
#sndc discriminator, from "A Large Scale Study of Regularization and Normalization in GANs"
################
class sndcgan(nn.Module):
    def __init__(self, DIM, num_chan, h, w):
        super(sndcgan, self).__init__()
       
        self.DIM = DIM
        self.final_h = h//8
        self.final_w = w//8
        self.main = nn.Sequential(ConvBlock(chan_in = num_chan, chan_out = DIM, ker_size = 3, stride = 1, pad = 1),
                ConvBlock(chan_in = 1 * DIM, chan_out = 2 * DIM, ker_size = 4, stride = 2, pad = 1),
                ConvBlock(chan_in = 2 * DIM, chan_out = 2 * DIM, ker_size = 3, stride = 1, pad = 1),
                ConvBlock(chan_in = 2 * DIM, chan_out = 4 * DIM, ker_size = 4, stride = 2, pad = 1),
                ConvBlock(chan_in = 4 * DIM, chan_out = 4 * DIM, ker_size = 3, stride = 1, pad = 1),
                ConvBlock(chan_in = 4 * DIM, chan_out = 8 * DIM, ker_size = 4, stride = 2, pad = 1),
                ConvBlock(chan_in = 8 * DIM, chan_out = 8 * DIM, ker_size = 3, stride = 1, pad = 1))
        self.linear = nn.Linear(self.final_h*self.final_w*8*DIM, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, self.final_h*self.final_w*8*self.DIM)
        output = self.linear(output)
        return output
        
################
#bsndc discriminator, which is sndc discriminator with more convolutions
################
class bsndcgan(nn.Module):
    def __init__(self, DIM, num_chan, h, w):
        super(bsndcgan, self).__init__()
       
        self.DIM = DIM
        self.final_h = max(h//128,1)
        self.final_w = max(w//128,1)
        main = nn.Sequential(ConvBlock(chan_in = num_chan, chan_out = DIM, ker_size = 3, stride = 1, pad = 1),
                ConvBlock(chan_in = 1 * DIM, chan_out = 2 * DIM, ker_size = 4, stride = 2, pad = 1),
                ConvBlock(chan_in = 2 * DIM, chan_out = 2 * DIM, ker_size = 3, stride = 1, pad = 1),
                ConvBlock(chan_in = 2 * DIM, chan_out = 4 * DIM, ker_size = 4, stride = 2, pad = 1),
                ConvBlock(chan_in = 4 * DIM, chan_out = 4 * DIM, ker_size = 3, stride = 1, pad = 1),
                ConvBlock(chan_in = 4 * DIM, chan_out = 8 * DIM, ker_size = 4, stride = 2, pad = 1),
                ConvBlock(chan_in = 8 * DIM, chan_out = 8 * DIM, ker_size = 3, stride = 1, pad = 1),
                ConvBlock(chan_in = 8 * DIM, chan_out = 8 * DIM, ker_size = 3, stride = 2, pad = 1),
                ConvBlock(chan_in = 8 * DIM, chan_out = 8 * DIM, ker_size = 3, stride = 2, pad = 1),
                ConvBlock(chan_in = 8 * DIM, chan_out = 8 * DIM, ker_size = 4, stride = 1, pad = 0),
                )
        self.main = main
        self.linear = nn.Linear(self.final_h*self.final_w*8*DIM, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, self.final_h*self.final_w*8*self.DIM)
        output = self.linear(output)
        return output

###############
#dcgan discriminator, from Radford et. al. 
###############

class dcgan(nn.Module):
    def __init__(self, DIM, num_chan, h, w):
        super(dcgan, self).__init__()
       
        self.DIM = DIM
        self.final_h = h//8
        self.final_w = w//8
        main = nn.Sequential(nn.Conv2d(num_chan, DIM, 3, 2,padding = 1),
                nn.LeakyReLU(),
                nn.Conv2d(DIM, 2*DIM, 3, 2, padding = 1),
                nn.LeakyReLU(),
                nn.Conv2d(2 * DIM, 4 * DIM, 3, 2, padding = 1),
                nn.LeakyReLU())
        self.main = main
        self.linear = nn.Linear(self.final_h*self.final_w*4*DIM, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, self.final_h*self.final_w*4*self.DIM)
        output = self.linear(output)
        return output

################
#infogan discriminator, from "Are GANs created equal?"
################

class infogan(nn.Module):
    def __init__(self, DIM, num_chan, h, w):
        super(infogan, self).__init__()
       
        self.DIM = DIM
        self.final_h = h//4
        self.final_w = w//4
        main = nn.Sequential(ConvBlock(chan_in = num_chan, chan_out = DIM, ker_size = 4, stride = 2, pad = 1),
                ConvBlock(chan_in = DIM, chan_out = 2 * DIM, ker_size = 4, stride = 2, pad = 1),
                )
        self.main = main
        self.fc1 = nn.Sequential(nn.Linear(self.final_h*self.final_w*2*DIM, 1024),
                nn.LeakyReLU(negative_slope = 0.1))
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, self.final_h*self.final_w*2*self.DIM)
        output = self.fc1(output)
        output = self.fc2(output)
        return output

################
#a discriminator which computes the norm of the input. For validation.
################
"""If you use this discriminator in conjunction with the unit_sphere source and the
all_zero target, the critic should converge to the norm function, and the value of the loss
function should decay geometrically from 1 to 0 with rate (1-theta)"""
class norm_taker(nn.Module):
    def __init__(self, DIM, num_chan, h, w):
        super(norm_taker, self).__init__()
       
        self.fc1 = nn.Linear(1, 1)
        self.fc1.weight.data = 0.1*torch.randn([1,1])+torch.ones([1,1])
        self.total_dim = num_chan*h*w

    def forward(self, input):
        output = torch.norm(input.view(-1, self.total_dim), dim = 1, keepdim = True)
        output = self.fc1(output)
        return output
