import torch
import numpy as np
from torch.nn import functional as F




def reconstruction_loss(data, recon_data, distribution="bernoulli"):
    """
    Calculates the per image reconstruction loss for a batch of data. I.e. negative
    log likelihood.

    Parameters
    ----------
    data : torch.Tensor
        Input data (e.g. batch of images). Shape : (batch_size, n_chan,
        height, width).

    recon_data : torch.Tensor
        Reconstructed data. Shape : (batch_size, n_chan, height, width).

    distribution : {"bernoulli", "gaussian", "laplace"}
        Distribution of the likelihood on the each pixel. Implicitely defines the
        loss Bernoulli corresponds to a binary cross entropy (bse) loss and is the
        most commonly used. It has the issue that it doesn't penalize the same
        way (0.1,0.2) and (0.4,0.5), which might not be optimal. Gaussian
        distribution corresponds to MSE, and is sometimes used, but hard to train
        ecause it ends up focusing only a few pixels that are very wrong. Laplace
        distribution corresponds to L1 solves partially the issue of MSE.

    storer : dict
        Dictionary in which to store important variables for vizualisation.

    Returns
    -------
    loss : torch.Tensor
        Per image cross entropy (i.e. normalized per batch but not pixel and
        channel)
    """
    RECON_DIST = ["bernoulli", "gaussian", "laplace"]
    batch_size, n_chan, height, width = recon_data.size()
    is_colored = n_chan == 3

    if distribution == "bernoulli":
        loss = F.binary_cross_entropy(recon_data, data, reduction="sum")
    elif distribution == "gaussian":
        # loss in [0,255] space but normalized by 255 to not be too big
        loss = F.mse_loss(recon_data * 255, data * 255, reduction="sum") / 255
    elif distribution == "laplace":
        # loss in [0,255] space but normalized by 255 to not be too big but
        # multiply by 255 and divide 255, is the same as not doing anything for L1
        loss = F.l1_loss(recon_data, data, reduction="sum")
        loss = loss * 3  # emperical value to give similar values than bernoulli => use same hyperparam
        loss = loss * (loss != 0)  # masking to avoid nan
    else:
        assert distribution not in RECON_DIST
        raise ValueError("Unkown distribution: {}".format(distribution))

    loss = loss / batch_size

    return loss



def kl_loss_function(use_cuda, num_steps, latent_dist, is_continuous=True, is_discrete=True, cont_capacity=[0.0, 5.0, 25000, 30], disc_capacity=[0.0, 5.0, 25000, 30]):
    """
    Calculates loss for a batch of data.

    Parameters
    ----------
    data : torch.Tensor
        Input data (e.g. batch of images). Should have shape (N, C, H, W)

    recon_data : torch.Tensor
        Reconstructed data. Should have shape (N, C, H, W)

    latent_dist : dict
        Dict with keys 'cont' or 'disc' or both containing the parameters
        of the latent distributions as values.

    cont_capacity, disc_capacity :
        Starting at a capacity of 0.0, increase this to 5.0\n",
        over 25000 iterations with a gamma of 30.0\n",
    """

    # Calculate KL divergences
    kl_cont_loss = 0  # Used to compute capacity loss (but not a loss in itself)
    kl_disc_loss = 0  # Used to compute capacity loss (but not a loss in itself)
    cont_capacity_loss = 0
    disc_capacity_loss = 0

    if is_continuous:
        # Calculate KL divergence
        mean, logvar = latent_dist['cont']
        kl_cont_loss = _kl_normal_loss(mean, logvar)
        # Linearly increase capacity of continuous channels
        cont_min, cont_max, cont_num_iters, cont_gamma = cont_capacity
        # Increase continuous capacity without exceeding cont_max
        cont_cap_current = (cont_max - cont_min) * num_steps / float(cont_num_iters) + cont_min
        cont_cap_current = min(cont_cap_current, cont_max)
        # Calculate continuous capacity loss
        cont_capacity_loss = cont_gamma * torch.abs(cont_cap_current - kl_cont_loss)

    if is_discrete:
        # Calculate KL divergence
        kl_disc_loss = _kl_multiple_discrete_loss(use_cuda, latent_dist['disc'])
        # Linearly increase capacity of discrete channels
        disc_min, disc_max, disc_num_iters, disc_gamma = disc_capacity
        # Increase discrete capacity without exceeding disc_max or theoretical
        # maximum (i.e. sum of log of dimension of each discrete variable)
        disc_cap_current = (disc_max - disc_min) * num_steps / float(disc_num_iters) + disc_min
        disc_cap_current = min(disc_cap_current, disc_max)
        # Require float conversion here to not end up with numpy float
        ###########################################

        disc_theoretical_max = sum([float(np.log(one_disc_vec.shape[1])) for one_disc_vec in latent_dist['disc']])
        ###########################################
        disc_cap_current = min(disc_cap_current, disc_theoretical_max)
        # Calculate discrete capacity loss
        disc_capacity_loss = disc_gamma * torch.abs(disc_cap_current - kl_disc_loss)

    return (kl_cont_loss, kl_disc_loss, cont_capacity_loss, disc_capacity_loss)


def _kl_normal_loss(mean, logvar):
    """
    Calculates the KL divergence between a normal distribution with
    diagonal covariance and a unit normal distribution.

    Parameters
    ----------
    mean : torch.Tensor
        Mean of the normal distribution. Shape (N, D) where D is dimension
        of distribution.

    logvar : torch.Tensor
        Diagonal log variance of the normal distribution. Shape (N, D)
    """
    # Calculate KL divergence
    kl_values = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
    # Mean KL divergence across batch for each latent variable
    kl_means = torch.mean(kl_values, dim=0)
    # KL loss is sum of mean KL of each latent variable
    kl_loss = torch.sum(kl_means)


    return kl_loss

def _kl_multiple_discrete_loss(use_cuda, alphas):
    """
    Calculates the KL divergence between a set of categorical distributions
    and a set of uniform categorical distributions.

    Parameters
    ----------
    alphas : list
        List of the alpha parameters of a categorical (or gumbel-softmax)
        distribution. For example, if the categorical atent distribution of
        the model has dimensions [2, 5, 10] then alphas will contain 3
        torch.Tensor instances with the parameters for each of
        the distributions. Each of these will have shape (N, D).
    """
    # Calculate kl losses for each discrete latent
    kl_losses = [_kl_discrete_loss(use_cuda, alpha) for alpha in alphas]

    # Total loss is sum of kl loss for each discrete latent
    kl_loss = torch.sum(torch.cat(kl_losses))


    return kl_loss

def _kl_discrete_loss(use_cuda, alpha):
    """
    Calculates the KL divergence between a categorical distribution and a
    uniform categorical distribution.

    Parameters
    ----------
    alpha : torch.Tensor
        Parameters of the categorical or gumbel-softmax distribution.
        Shape (N, D)
    """
    EPS = 1e-12
    disc_dim = int(alpha.size()[-1])
    log_dim = torch.Tensor([np.log(disc_dim)])
    if use_cuda:
        log_dim = log_dim.cuda()
    # Calculate negative entropy of each row
    neg_entropy = torch.sum(alpha * torch.log(alpha + EPS), dim=1)
    # Take mean of negative entropy across batch
    mean_neg_entropy = torch.mean(neg_entropy, dim=0)
    # KL loss of alpha with uniform categorical variable
    kl_loss = log_dim + mean_neg_entropy
    return kl_loss
