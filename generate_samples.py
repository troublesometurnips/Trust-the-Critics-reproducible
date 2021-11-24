import torch
import torchvision
from torchvision import utils
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

# For generating samples
def generate_image(frame, data, ext, path):
    """After re-centering data, this code saves a grid of samples to the path
    path/samples/
    Inputs:
    - frame; title for the figure you want to save
    - data; minibatch of data to save
    - ext; file extension
    - path; path for saving images
    Outputs:
    - saved image in path/samples/
    """

    data = 0.5*data + 0.5*torch.ones_like(data)#by default, data is generated in [-1,1]

    grid = utils.make_grid(data, nrow = 16, padding = 1)
    plt.imshow(grid.cpu().numpy().transpose((1, 2, 0)))
    plt.axis('off')
    os.makedirs(os.path.join(path,'samples'), exist_ok = True)
    plt.savefig(os.path.join(path,'samples/{}.{}'.format(frame,ext)), dpi = 300)
    plt.close()

def save_individuals(b_idx, t_idx, data, ext, path, to_rgb = False):
    """After re-centering data, this code individually saves numbered images.
    Inputs
    - b_idx; batch identifier
    - t_idx; timestamp identifier
    - data; minibatch of data to be saved
    - ext; extension for saved images
    - path; path for saving images. Will be saved in path/pics/timestamp{t_idx}
    - to_rgb; if true, repeats greyscale images three times over channel
    Outputs
    - 
    """
    data = 0.5*data + 0.5*torch.ones_like(data)#by default, data is generated in [-1,1]
    bs = data.shape[0]
    os.makedirs(os.path.join(path, 'pics'), exist_ok = True)
    path = os.path.join(path, 'pics/timestamp{}'.format(t_idx))
    os.makedirs(path, exist_ok = True)

        
    for i in range(bs):
        single = data[i,:,:,:]
        if single.shape[0] == 1 and to_rgb:
            single = single.repeat(3,1,1)

        utils.save_image(single, os.path.join(path, '{:05d}.{}'.format(b_idx*bs + i, ext)))
    
