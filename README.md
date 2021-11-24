# Trust-the-Critics (Code for Reproducibility)

The main branch of this repository includes a cleaned up version of the code for TTC. This branch ("reproducible") includes the code that was run to produce the results in Section 5 of the paper. By running the code in this branch using Python 3.7.4 with the provided random seeds and packages as in the requirements file, you should get the results from Section 5 exactly.

A brief description of the code is included here. See the main branch for more details.
 
- **ttc.py**: for training critics using the TTC algorithm. 
- **ttc_eval.py**: for evaluating trained critics from TTC for either image generation or image translation. This produces samples from the final distribution \mu_N and  optionally computes the FID.
- **denoise_eval.py**: for evaluating trained critics from TTC for image denoising. Adds noise to the test set from BSDS500 and denoises using TTC and the benchmark technique.
- **wgan_gp.py**: for training a wgan to compare against TTC.
- **wgan_eval.py**: for producing samples from trained WGAN and optionally computing FID.
- The rest of the code defines functions which are used by the above files.


### Random Seeds
The experiments in Section 5.1 (image generation) of the paper use the random seeds 0, 1, 2, 3, 4 for the distinct training runs. The experiments in Section 5.2 (image translation) do not use a reproducible seed (i.e. seed = -1 in the code); we do not see this as a major obstacle to reproducibility given the subjective nature of the results. The experiments in Section 5.3 (image denoising) use a single random seed of 0 for all noise levels.  

### Computing Architecture
This code was run on the Graham cluster of Compute Canada (https://www.computecanada.ca/), using a single NVIDIA V100 Volta GPU. Approximate training times are as follows:

- **MNIST/Fashion MNIST generation training run**: 1.2 hours (TTC), 0.7 hours (WGAN-GP)
- **CIFAR10 generation training run**: 3.2 hours (TTC), 1.5 hours (WGAN-GP)
- **Image translation training run**: 18.8 hours
- **Image denoising training run**: 8.6 hours

### Hyperparameters
Hyperparameters for TTC are specified in the text of the paper, with the exception of our hyperparameters for the benchmark denoising technique of Lunz et al. (https://proceedings.neurips.cc/paper/2018/file/d903e9608cfbf08910611e4346a0ba44-Paper.pdf). For this technique we use the default parameters as given in denoise_eval.py, which were selected using a grid search to obtain best mean PSNR over the test dataset.

### Untrained generators
We were not able to upload the untrained generators used to produce the data for TTC 1 to this repository. To create these generators, run wgan_gp.py with the model as infogan and your chosen seed. This will save the initial untrained generator's state dict in temp_dir/model_dicts/. Before running ttc.py with untrained generator as your source, move the state dict for this untrained generator to temp_dir/ugen.pth.
