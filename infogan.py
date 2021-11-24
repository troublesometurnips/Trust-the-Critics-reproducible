import torch
from torch import nn

"""An implementation of the discriminator and generator of INFOGAN from "Are all GANs Created Equal?"""

class Generator(nn.Module):
    def __init__(self, DIM, num_chan, h, w):
        super(Generator, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(64, 1024, bias = False),#latent dimension is 64
            nn.BatchNorm1d(1024),
            nn.ReLU(True)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024,128 * (h // 4) * (w // 4) , bias = False),
            nn.BatchNorm1d(128 * (h // 4) * (w // 4)),
            nn.ReLU(True)
        )

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, bias = False, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(64, num_chan, 4, stride=2, bias = False, padding = 1),
        )

        self.tanh = nn.Tanh()
        self.num_chan = num_chan
        self.h = h
        self.w = w

    def forward(self, input):
        output = self.fc1(input)
        output = self.fc2(output)
        output = output.view(-1, 128, self.h // 4, self.w // 4)
        output = self.conv1(output)
        output = self.conv2(output)
        output = self.tanh(output)
        return output.view(-1, self.num_chan, self.h, self.w)

class Discriminator(nn.Module):
    def __init__(self, DIM, num_chan, h, w):
        super(Discriminator, self).__init__()
       
        self.DIM = DIM
        self.final_h = h//4
        self.final_w = w//4
        main = nn.Sequential(nn.Conv2d(num_chan, DIM, 4, 2, padding = 1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2d(DIM, 2*DIM, 4, 2, padding = 1),
                nn.LeakyReLU(negative_slope=0.1),
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

