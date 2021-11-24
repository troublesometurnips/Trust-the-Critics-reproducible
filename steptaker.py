import torch
from torch import autograd
import sys


def steptaker(data, critic, step, num_step = 1):
    """Applies gradient descent (GD) to data using critic
    Inputs
    - data; data to apply GD to
    - critic; critic to compute gradients of
    - step; how large of a step to take
    - num_step; how finely to discretize flow. taken as 1 in TTC
    Outputs
    - data with gradient descent applied
    """

    for j in range(num_step):

        gradients = grad_calc(data, critic)

        data = (data - (step/num_step)*gradients).detach()

    return data.detach()

def rk4(data, critic, step, num_step = 1):
    """Assumes data is a batch of images, critic is a Kantorovich potential,
    and step is  desired step size. Applies fourth order Runge-Kutta to the data num_step times
    with stepsize step/num_step. Unused in TTC"""
    h = step/num_step
    for j in range(num_step):
        data_0 = data.detach().clone()
        k = grad_calc(data_0, critic)
        data += (h/6)*k

        k = grad_calc(data_0 + (h/2)*k, critic)
        data += (h/3)*k

        k = grad_calc(data_0 + (h/2)*k, critic)
        data += (h/3)*k

        k = grad_calc(data_0 + k, critic)
        data += (h/6)*k
        data = data.detach()

    return data




def grad_calc(data, critic):
    """Returns the gradients of critic at data"""
    data = data.detach().clone()
    data.requires_grad = True
    Dfake = critic(data)

    gradients = autograd.grad(outputs = Dfake, inputs = data,
                            grad_outputs = torch.ones(Dfake.size()).cuda(), only_inputs=True)[0]
    return gradients.detach()
    
