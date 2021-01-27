import torch
import botorch
from model_classes.encoders_decoders_pytorch import GaussianEncoder, StickBreakingEncoder, Decoder
from utils.util_vars import init_weight_mean_var, input_ndims, latent_ndims, prior_mu, prior_sigma, prior_shape_alpha,\
    prior_shape_beta


class VAE(object):
    def __init__(self, target_distribution, latent_distribution, prior_param1, prior_param2):
        # GaussianVAE: target == latent == torch.distributions.MultivariateNormal
        # StickBreakingVAE: target == torch.distributions.beta.Beta, latent = botorch.distributions.Kumaraswamy
        self.target_distribution = target_distribution
        self.latent_distribution = latent_distribution
        self.prior_param1 = prior_param1
        self.prior_param2 = prior_param2

        self.init_weights(self.encoder_layers)
        self.init_weights(self.decoder_layers)

    def init_weights(self, layers):
        for layer in layers:
            with torch.no_grad():
                layer_size = (layer.out_features, layer.in_features)
                layer.weight = torch.nn.Parameter(torch.normal(*init_weight_mean_var, layer_size))
                layer.bias = torch.nn.Parameter(torch.zeros(layer.out_features))

    def ELBO_loss(self, recon_x, x, param1, param2, kl_divergence):
        # for Gaussian, param1 == mu, param2 == sigma.pow(2)
        # for Kumaraswamy, param1 == alpha, param2 == beta
        n_samples = len(recon_x)
        reconstruction_loss = (x.view(-1, input_ndims) - recon_x).pow(2).mul(.5).sum(axis=1)  # neg loglik Nalisnick
        regularization_loss = torch.stack([kl_divergence(param1[i], param2[i]) for i in range(n_samples)])
        return (reconstruction_loss + regularization_loss).mean()

    def monte_carlo_kl_divergence(self, param1, param2):
        q = self.latent_distribution(param1, param2)  # for Gaussian, param1==mu, param2==torch.diag(sigma)
        z = q.rsample()
        log_qzx = q.log_prob(z)
        log_pz = self.target_distribution(self.prior_param1, self.prior_param2).log_prob(z)
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def reparametrize(self, param1, param2, parametrization=None):
        if parametrization == 'Gaussian':
            # in Gaussian, param1 == mu, param2 == sigma
            epsilon = param2.data.new(param2.size()).normal_()  # noise term
            out = param1 + param2 * epsilon

        elif parametrization == 'Kumaraswamy':
            # for Kumaraswamy, param1 == alpha, param2 == beta
            v = self.get_kumaraswamy_samples(param1, param2)
            out = self.get_stick_segments(v)

        return out

    def get_kumaraswamy_samples(self, param1, param2):
        v = (1 - torch.rand(latent_ndims).pow(1 / param2)).pow(1 / param1)  # Kumaraswamy samples
        return v

    def get_stick_segments(self, v):
        pi = v.data.new_zeros(v.size())
        for k in range(latent_ndims):
            if k == 0:
                pi[k] = v[k]
            else:
                pi[k] = v[k] * torch.stack([(1 - v[j]) for j in range(latent_ndims) if j < k]).prod()
        return pi


class GaussianVAE(torch.nn.Module, GaussianEncoder, Decoder, VAE):
    def __init__(self):
        super(GaussianVAE, self).__init__()
        GaussianEncoder.__init__(self)
        Decoder.__init__(self)
        VAE.__init__(self, target_distribution=torch.distributions.MultivariateNormal,
                     latent_distribution=torch.distributions.MultivariateNormal,
                     prior_param1=torch.ones(latent_ndims) * prior_mu,
                     prior_param2=torch.diag(torch.ones(latent_ndims) * prior_sigma))

    def forward(self, x):
        mu, sigma = self.encode(x.view(-1, input_ndims))
        z = self.reparametrize(mu, sigma, parametrization='Gaussian') if self.training else mu
        reconstructed_x = self.decode(z)
        return reconstructed_x, mu, torch.stack([torch.diag(sigma[i].pow(2)) for i in range(len(sigma))])

    def Gaussian_ELBO_loss(self, recon_x, x, mu, sigma):
        n_samples = len(recon_x)
        reconstruction_loss = (x.view(-1, input_ndims) - recon_x).pow(2).mul(.5).sum(axis=1)
        regularization_loss = torch.stack([self.Gaussian_KL_divergence(mu[i], sigma[i].pow(2))
                                           for i in range(n_samples)])
        return (reconstruction_loss + regularization_loss).mean()

    def kl_divergence(self, mu, sigma):
        q = self.latent_distribution(mu, sigma)
        p = self.target_distribution(self.prior_param1, self.prior_param2)
        kl = torch.distributions.kl_divergence(q, p)
        return kl


class StickBreakingVAE(torch.nn.Module, StickBreakingEncoder, Decoder, VAE):
    def __init__(self):
        super(StickBreakingVAE, self).__init__()
        StickBreakingEncoder.__init__(self)
        Decoder.__init__(self)
        VAE.__init__(self, target_distribution=torch.distributions.beta.Beta,
                     latent_distribution=botorch.distributions.Kumaraswamy,
                     prior_param1=torch.ones(latent_ndims) * prior_shape_alpha,
                     prior_param2=torch.ones(latent_ndims) * prior_shape_beta)

    def forward(self, x):
        alpha, beta = self.encode(x.view(-1, input_ndims))
        pi = self.reparametrize(alpha, beta, parametrization='Kumaraswamy') if self.training \
            else self.get_kumaraswamy_samples(alpha, beta)
        reconstructed_x = self.decode(pi)
        return reconstructed_x, alpha, beta

    def kl_divergence(self, alpha, beta):
        psi_b_taylor_approx = beta.log() - 1. / beta.mul(2) - 1. / beta.pow(2).mul(12)  # Digamma function taylor approx
        kl = (alpha - prior_shape_alpha) / alpha * (-0.57721 - psi_b_taylor_approx - 1 / beta)
        kl += alpha.mul(beta).log()
        kl += self.beta_func(prior_shape_alpha, prior_shape_beta).log()
        kl += - (beta - 1) / beta
        kl += (torch.stack([1. / (i + alpha * beta) * self.beta_func(i / alpha, beta) for i in range(1, 11)])\
               * (prior_shape_beta - 1) * beta).sum(axis=0)  # 10th-order Taylor approximation
        return kl.sum()

    def beta_func(self, a, b):
        return (torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)).exp()
