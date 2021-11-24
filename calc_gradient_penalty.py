import torch
from torch import autograd

#~~~~~~Gradient Penalty
def calc_gradient_penalty(model, real_data, fake_data, LAMBDA, plus = True):
    """Computes the gradient penalty from WGAN-GP.
    Inputs:
    - model; the critic network to be penalized
    - real_data; a minibatch of real data
    - fake_data; a minibatch of generated data
    - LAMBDA; coefficient multiplying penalty
    - plus; if true, take the one-sided penalty.
    Outputs:
    - gradient_penalty; the gradient penalty computed using interpolates of real
    and fake data"""

    use_cuda = torch.cuda.is_available()
    bs = real_data.shape[0]


    alpha = torch.rand(bs,1,1,1)#interpolation parameter
    alpha = alpha.expand_as(real_data)


    alpha = alpha.cuda() if use_cuda else alpha
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda()
    
    interpolates.requires_grad = True
    disc_interpolates = model(interpolates)#critic evaluated on interpolates

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones_like(disc_interpolates),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    if plus:
        gradient_penalty = (torch.clamp(gradients.norm(2, dim=1)-1,min =0)**2).mean() * LAMBDA
    else:
        gradient_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean() * LAMBDA

    return gradient_penalty

def AL_gradient_penalty(model, real_data, fake_data):
    """Returns a list of gradient norms at interpolates of real and fake"""
    use_cuda = torch.cuda.is_available()
    bs = real_data.shape[0]


    alpha = torch.rand(bs,1,1,1)
    alpha = alpha.expand_as(real_data)


    alpha = alpha.cuda() if use_cuda else alpha
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda()
    
    interpolates.requires_grad = True
    disc_interpolates = model(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones_like(disc_interpolates),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    norms = torch.norm(gradients, p =2, dim = 1)

    return norms

