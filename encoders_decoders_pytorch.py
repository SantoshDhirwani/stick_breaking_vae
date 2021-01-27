from torch import nn
from util_vars import input_ndims, hidden_ndims, activation, latent_ndims


class GaussianEncoder(object):
    def __init__(self):
        self.input_to_hidden = nn.Linear(input_ndims, hidden_ndims)
        self.hidden_to_mu = nn.Linear(hidden_ndims, latent_ndims)
        self.hidden_to_sigma = nn.Linear(hidden_ndims, latent_ndims)
        self.activation = activation
        self.encoder_layers = nn.ModuleList([self.input_to_hidden, self.hidden_to_mu, self.hidden_to_sigma])

    def encode(self, x):
        hidden = self.activation(self.input_to_hidden(x))
        parameters = self.hidden_to_mu(hidden), self.hidden_to_sigma(hidden)
        return parameters


class StickBreakingEncoder(object):
    def __init__(self):
        self.input_to_hidden = nn.Linear(input_ndims, hidden_ndims)
        self.hidden_to_alpha = nn.Linear(hidden_ndims, latent_ndims)
        self.hidden_to_beta = nn.Linear(hidden_ndims, latent_ndims)
        self.activation = activation
        self.encoder_layers = nn.ModuleList([self.input_to_hidden, self.hidden_to_alpha, self.hidden_to_beta])
        self.softplus = nn.Softplus()  # smooth approximation to ReLU, to constrain output to positive

    def encode(self, x):
        # Softplus per Nalisnick & Smythe github implementation
        hidden = self.activation(self.input_to_hidden(x))
        parameters = self.softplus(self.hidden_to_alpha(hidden)), self.softplus(self.hidden_to_beta(hidden))
        return parameters


class Decoder(object):
    def __init__(self):
        self.latent_to_hidden = nn.Linear(latent_ndims, hidden_ndims)
        self.hidden_to_reconstruction = nn.Linear(hidden_ndims, input_ndims)
        self.activation = activation
        self.decoder_layers = nn.ModuleList([self.latent_to_hidden, self.hidden_to_reconstruction])

    def decode(self, z):
        hidden = self.activation(self.latent_to_hidden(z))
        reconstruction = self.activation(self.hidden_to_reconstruction(hidden))
        return reconstruction
