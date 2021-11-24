import torch

def get_data(generic_iterator, generic_loader):
    """Code to get minibatch from data iterator
    Inputs:
    - generic_iterator; iterator for dataset
    - generic_loader; loader for dataset

    Outputs:
    - data; minibatch of data from iterator
    - generic_iterator; iterator for dataset, reset if
    you've reached the end of the dataset"""

    try:
        data = next(generic_iterator)[0]
    
    except StopIteration:
        generic_iterator = iter(generic_loader)
        data = next(generic_iterator)[0]

    if torch.cuda.is_available():
        data = data.cuda()

    return data, generic_iterator
