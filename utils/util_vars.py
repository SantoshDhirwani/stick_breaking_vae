import torch
import os
import torchvision
import numpy as np
from torch import nn
from torchvision import transforms

# arg defaults in https://github.com/enalisnick/stick-breaking_dgms/blob/master/run_gauss_VAE_experiments.py
seed = 1234
torch.set_rng_state(torch.manual_seed(seed).get_state())
batch_size = 100
latent_ndims = 50
hidden_ndims = 500
learning_rate = 3e-4
lookahead = 30
print_interval = 10
n_train_epochs = 2000
init_weight_mean_var = (0, .001)
prior_mu = torch.Tensor([0.])
prior_sigma = torch.Tensor([1.])
prior_shape_alpha = torch.Tensor([1.])
prior_shape_beta = concentration_alpha0 = torch.Tensor([5.])
uniform_low = torch.Tensor([.01])
uniform_high = torch.Tensor([.99])
activation = nn.ReLU()
train_valid_test_splits = (45000, 5000, 10000)
n_monte_carlo_samples = 1
dataloader_kwargs = {}
download_needed = not os.path.exists('./MNIST')

# use GPU, if available
CUDA = torch.cuda.is_available()
if CUDA:
    torch.cuda.manual_seed(seed)
    dataloader_kwargs.update({'num_workers': 1, 'pin_memory': True})

# get datasets
train_dataset = torchvision.datasets.MNIST('.', train=True, transform=transforms.ToTensor(), download=download_needed)
test_dataset = torchvision.datasets.MNIST('.', train=False, transform=transforms.ToTensor())

# get dimension info
input_shape = list(train_dataset.data[0].shape)
input_ndims = np.product(input_shape)

# define data loaders
train_dataset = train_dataset.data.reshape(-1, 1, *input_shape) / 255  # reshaping and scaling bytes to [0,1]
test_dataset = test_dataset.data.reshape(-1, 1, *input_shape) / 255
pruned_train_dataset = train_dataset.data[:train_valid_test_splits[0]]
train_loader = torch.utils.data.DataLoader(pruned_train_dataset,
                                           shuffle=True, batch_size=batch_size, **dataloader_kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          shuffle=False, batch_size=batch_size, **dataloader_kwargs)

parametrizations = dict(Kumar='Kumaraswamy', GLogit='Gauss_Logit', GEM='GEM')