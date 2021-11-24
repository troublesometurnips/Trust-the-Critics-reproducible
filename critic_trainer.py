from get_data import get_data
from calc_gradient_penalty import calc_gradient_penalty
import log
import time
import torch
from steptaker import steptaker

def critic_trainer(critic_list, optimizer_list, iteration, steps, target_loader, source_loader, args):
    """Trains critic
    Inputs
    - critic_list; list of all critics
    - optimizer_list; list of optimizers for critics
    - iteration; which critic are you currently training
    - steps; how large of a step to take for each critic
    - target/source_loader; dataloader for target or source
    - args; args to ttc.py

    Outputs:
    - critic_list; the same as input, but with critic at
    "iteration" trained.
    - step for critic at iteration"""

    start_time = time.time()
    target_iter = iter(target_loader)#make source and target loaders into iterables
    source_iter = iter(source_loader)

    steplist = []

    for i in range(args.critters):
       
        real, target_iter = get_data(target_iter, target_loader)#real, fake are now minibatches of data
        fake, source_iter = get_data(source_iter, source_loader) 

        for param in critic_list[iteration].parameters():
            param.grad = None#zero gradients of current critic

        D_real = critic_list[iteration](real)
        D_real = D_real.mean()
        D_real.backward()
        # train with fake

        #APPLY TRANSFORMATIONS - if iterations >=1, apply gradient descent maps from previous critics.
        for j in range(iteration):
            fake = steptaker(fake, critic_list[j], steps[j], num_step = args.num_step)


        D_fake = critic_list[iteration](fake)
        D_fake = -D_fake.mean()
        D_fake.backward()
        
        # train with gradient penalty
        gradient_penalty = calc_gradient_penalty(critic_list[iteration], real, fake, args.lamb, plus = args.plus)
        gradient_penalty.backward()

        D_cost = D_real + D_fake + gradient_penalty #D_fake has negative baked in
        D_cost_nopen = D_real + D_fake #to get a sense of magnitude of gradient penalty
        optimizer_list[iteration].step()

        # Record values in log
        log.plot('dcost', D_cost.cpu().data.numpy())
        log.plot('time', time.time() - start_time)
        log.plot('no_gpen', D_cost_nopen.cpu().data.numpy())


        # Save logs every 1000 iters
        if (i < 5) or (i % 100 == 99):
            log.flush(args.temp_dir)

        log.tick()#increment index of log

        if args.critters - i <= 100:
            steplist.append(-D_cost.detach())#gathering steps from last 100 minibatches
    steplist = torch.stack(steplist)
    print('Note: using last 100 batches to compute step')
    return critic_list, steplist.mean()
