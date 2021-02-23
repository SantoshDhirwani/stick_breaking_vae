import torch
import botorch
from model_classes.encoders_decoders_pytorch import GaussianEncoder, StickBreakingEncoder, Decoder
from utils.util_vars import init_weight_mean_var, input_ndims, latent_ndims, prior_mu, prior_sigma, prior_alpha, \
    prior_beta, uniform_low, uniform_high
from utils.util_funcs import beta_func
from utils.util_classes import Gauss_Logit, logistic_func, GammaRandomVariables
from numpy.testing import assert_almost_equal


class VAE(object):
    def __init__(self, target_distribution, latent_distribution, prior_param1, prior_param2):
        self.target_distribution = target_distribution
        self.latent_distribution = latent_distribution
        self.prior_param1 = prior_param1
        self.prior_param2 = prior_param2

        self.init_weights(self.encoder_layers)
        self.init_weights(self.decoder_layers)

        self.uniform_distribution = torch.distributions.uniform.Uniform(low=uniform_low, high=uniform_high)

    def init_weights(self, layers):
        for layer in layers:
            with torch.no_grad():
                layer_size = (layer.out_features, layer.in_features)
                layer.weight = torch.nn.Parameter(torch.normal(*init_weight_mean_var, layer_size))
                layer.bias = torch.nn.Parameter(torch.zeros(layer.out_features))

    def ELBO_loss(self, recon_x, x, param1, param2, kl_divergence):
        n_samples = len(recon_x)
        x = x.view(-1, input_ndims)
        if not torch.isfinite(recon_x.log()).all():
            raise AssertionError('Reconstructed x.log not finite!: ', recon_x.log())
        reconstruction_loss = - (x * recon_x.log() + (1 - x)
                                 * (1 - recon_x).log()).sum(axis=1)  # per Nalisnick & Smythe github
        regularization_loss = torch.stack([kl_divergence(param1[i], param2[i]) for i in range(n_samples)])
        return reconstruction_loss.mean(),  regularization_loss.mean()

    def monte_carlo_kl_divergence(self, param1, param2):
        # TODO: ensure positive KL divergence for GEM, GLogit
        q = self.latent_distribution(param1, param2)
        z = q.rsample()
        log_qzx = q.log_prob(z)  # TODO: ensure log_prob accurately calculated for GLogit (currently returns a scalar)
        log_pz = self.target_distribution(self.prior_param1, self.prior_param2).log_prob(z)
        kl = (log_qzx - log_pz).sum(axis=-1)
        return kl

    def reparametrize(self, param1, param2, parametrization=None):
        if parametrization == 'Gaussian':
            # for Gaussian, param1 == mu, param2 == sigma
            epsilon = param2.data.new(param2.size()).normal_()
            out = param1 + param2 * epsilon

        elif parametrization == 'Kumaraswamy':
            # for Kumaraswamy, param1 == alpha, param2 == beta
            v = self.set_v_K_to_one(self.get_kumaraswamy_samples(param1, param2))
            out = self.get_stick_segments(v)

        elif parametrization == 'GEM':
            # for GEM, param1 == alpha, param2 == beta
            v = self.set_v_K_to_one(self.get_GEM_samples(param1, param2))
            out = self.get_stick_segments(v)

        elif parametrization == 'Gauss_Logit':
            # for Gauss-Logit, param1 == mu, param2 == sigma
            epsilon = param2.data.new(param2.size()).normal_()
            v = self.set_v_K_to_one(logistic_func(param1 + param2 * epsilon))
            out = self.get_stick_segments(v)

        return out

    def get_kumaraswamy_samples(self, param1, param2):
        # u is analogous to epsilon noise term in the Gaussian VAE
        u = self.uniform_distribution.sample([1]).squeeze()
        v = (1 - u.pow(1 / param2)).pow(1 / param1)
        return v  # sampled fractions

    def get_GEM_samples(self, param1, param2):
        u_hat = self.uniform_distribution.sample([1]).squeeze()
        v = (u_hat.mul(param1 * torch.lgamma(param1)).pow(1 / param1)).div(param2)
        return v

    def set_v_K_to_one(self, v):
        # set Kth fraction v_i,K to one to ensure the stick segments sum to one
        v0 = v[:, -1].pow(0).reshape(v.shape[0], 1)  # TODO: fix non-differentiable here?
        v1 = torch.cat([v[:, :latent_ndims - 1], v0], dim=1)
        return v1

    def get_stick_segments(self, v):
        n_samples = v.size()[0]
        n_dims = v.size()[1]
        pi = torch.zeros((n_samples, n_dims))

        for k in range(n_dims):
            if k == 0:
                pi[:, k] = v[:, k]
            else:
                pi[:, k] = v[:, k] * torch.stack([(1 - v[:, j]) for j in range(n_dims) if j < k]).prod(axis=0)

        # ensure stick segments sum to 1
        assert_almost_equal(torch.ones(n_samples), pi.nansum(axis=1).detach().numpy(),
                            decimal=2, err_msg='stick segments do not sum to 1')
        return pi


class GaussianVAE(torch.nn.Module, GaussianEncoder, Decoder, VAE):
    def __init__(self):
        super(GaussianVAE, self).__init__()
        GaussianEncoder.__init__(self)
        Decoder.__init__(self)
        VAE.__init__(self, target_distribution=torch.distributions.MultivariateNormal,
                     latent_distribution=torch.distributions.MultivariateNormal,
                     prior_param1=torch.ones(latent_ndims) * prior_mu,
                     prior_param2=torch.diag(torch.ones(latent_ndims) * prior_sigma ** 2))

    def forward(self, x):
        mu, sigma = self.encode(x.view(-1, input_ndims))
        z = self.reparametrize(mu, sigma, parametrization='Gaussian') if self.training else mu
        reconstructed_x = self.decode(z)
        return reconstructed_x, mu, torch.stack([torch.diag(sigma[i]).pow(2) for i in range(len(sigma))])

    def kl_divergence(self, mu, sigma):
        q = self.latent_distribution(mu, sigma)
        p = self.target_distribution(self.prior_param1, self.prior_param2)
        kl = torch.distributions.kl_divergence(q, p)
        return kl

        # # Nalisnick & Smythe implementation
        # sigma = sigma.diag() if sigma.ndim == 2 else sigma
        # kl = -prior_sigma.repeat(sigma.shape).pow(2).log()
        # kl += -(sigma.mul(2).exp() + (mu - prior_mu).pow(2)) / prior_sigma.pow(2)
        # kl += sigma.mul(2).add(1.)
        # return -0.5 * kl.sum(axis=0)


class StickBreakingVAE(torch.nn.Module, StickBreakingEncoder, Decoder, VAE):
    def __init__(self, parametrization):
        super(StickBreakingVAE, self).__init__()
        StickBreakingEncoder.__init__(self)
        Decoder.__init__(self)
        self.parametrization = parametrization

        latent_switcher = dict(Kumaraswamy=botorch.distributions.Kumaraswamy,
                               Gauss_Logit=Gauss_Logit,
                               GEM=GammaRandomVariables)
        latent_distribution = latent_switcher.get(self.parametrization)

        VAE.__init__(self, target_distribution=torch.distributions.beta.Beta,
                     latent_distribution=latent_distribution,
                     prior_param1=torch.ones(latent_ndims) * prior_alpha,
                     prior_param2=torch.ones(latent_ndims) * prior_beta)

    def forward(self, x):
        param1, param2 = self.encode(x.view(-1, input_ndims))
        if self.training:
            pi = self.reparametrize(param1, param2, parametrization=self.parametrization)
        else:
            # for the pdf, reconstruct samples from the area of highest density
            v_switcher = dict(
                Kumaraswamy=(1 - self.latent_distribution(param1, param2).mean.pow(1 / param2)).pow(1 / param1),
                GEM=torch._standard_gamma(param1).mul(param1).mul(torch.distributions.Beta(param1, param2).mean).pow(
                    1 / param1).div(param2),
                Gauss_Logit=param1)

            v = self.set_v_K_to_one(v_switcher.get(self.parametrization))
            pi = self.get_stick_segments(v)

        reconstructed_x = self.decode(pi)

        if self.parametrization == 'Gauss_Logit':
            param2 = torch.stack([torch.diag(param2[i].pow(2)) for i in range(len(param2))])

        return reconstructed_x, param1, param2

    def kl_divergence(self, param1, param2):
        kl_switcher = dict(Kumaraswamy=self.kumaraswamy_kl_divergence,
                           GEM=self.monte_carlo_kl_divergence,
                           Gauss_Logit=self.monte_carlo_kl_divergence)
        kl_divergence_func = kl_switcher.get(self.parametrization)
        return kl_divergence_func(param1, param2)

    def kumaraswamy_kl_divergence(self, alpha, beta):
        assert((alpha != 0).all(), f'Zero at alpha indices: {torch.nonzero((alpha!=0) == False, as_tuple=False).squeeze()}')
        assert((beta != 0).all(), f'Zero at beta indices: {torch.nonzero((beta!=0) == False, as_tuple=False).squeeze()}')

        psi_b_taylor_approx = beta.log() - 1. / beta.mul(2) - 1. / beta.pow(2).mul(12)
        kl = ((alpha - prior_alpha) / alpha) * (-0.57721 - psi_b_taylor_approx - 1 / beta)
        kl += alpha.mul(beta).log() + beta_func(prior_alpha, prior_beta).log()  # normalization constants
        kl += - (beta - 1) / beta
        kl += torch.stack([1. / (i + alpha * beta) * beta_func(i / alpha, beta) for i in range(1, 11)]).sum(axis=0) \
              * (prior_beta - 1) * beta  # 10th-order Taylor approximation

        return kl.sum()
