import os
import imageio
import argparse
import subprocess
from torchvision import transforms
import torch
import numpy as np
from torch.nn import functional as F
from util_math import *
from torch.distributions.relaxed_categorical import ExpRelaxedCategorical

###############################################################################


def get_log_pz_qz_prodzi_qzCx(latent_sample, latent_dist, n_data, use_cude, is_mss=True):
    """
    Calculates log densities

    Parameters
    ----------
    latent_sample: torch.Tensor or np.ndarray or float
        Value at which to compute the density. (batch size, latent dim)

    latent_dist: torch.Tensor or np.ndarray or float
        statisitc for dist. Each of statistics has size of (batch size, latent dim).
        For guassian, latent_dist = (Mean, logVar)
        For gumbel_softmax, latent_dist = alpha(prob. of categorical variable)
    """
    batch_size, hidden_dim = latent_sample['cont'].shape

    # calculate log q(z|x)
    log_q_ziCx_cont = log_density_gaussian(latent_sample['cont'], *(latent_dist['cont'])) #64,10
    log_q_ziCx_disc = log_density_categorical(latent_sample['disc'], latent_dist['disc']) #64

    log_q_ziCx = torch.cat((log_q_ziCx_cont, log_q_ziCx_disc.unsqueeze(-1)), dim=1) # 64,11
    log_q_zCx = log_q_ziCx.sum(1) #64   sum across logP(z_i). i.e, \prod P(z_i | x_i)


    # calculate log p(z)
    zeros = torch.zeros_like(latent_sample['cont']) # mean and log var is 0
    log_pzi_cont = log_density_gaussian(latent_sample['cont'], zeros, zeros) # sum across logP(z_i). i.e, \prod P(z_i)

    unif_logits = torch.log(torch.ones_like(latent_sample['disc']) * 1 / latent_sample['disc'].shape[1])
    relaxedCate = ExpRelaxedCategorical(torch.tensor(.67), logits=unif_logits)

    log_pzi_dist = log_density_categorical(latent_sample['disc'], relaxedCate) # sum across logP(z_i). i.e, \prod P(z_i)

    log_pzi = torch.cat((log_pzi_cont, log_pzi_dist.unsqueeze(-1)), dim=1)
    log_pz = log_pzi.sum(1)

    # compute log q(z) ~= log 1/(NM) sum_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m))
    mat_log_qzi_cont = matrix_log_density(latent_sample['cont'], *(latent_dist['cont'])) #(256,256,10): only (n,n,10) is the result of correct pair of (latent sample, m, s).
    # (n,n,10) --> first n = num of given samples(batch). second n = for Monte Carolo. 10 = latent dim.
    batch_dim = 1
    latent_sample_disc = latent_sample['disc'].unsqueeze(0).unsqueeze(batch_dim + 1).transpose(batch_dim, 0)
    mat_log_qzi_disc = log_density_categorical(latent_sample_disc, latent_dist['disc']).transpose(1, batch_dim + 1) #(64,64,1)

    mat_log_qzi = torch.cat((mat_log_qzi_cont, mat_log_qzi_disc), dim=2)
    if is_mss:
        # use stratification
        log_iw_mat = log_importance_weight_matrix(batch_size, n_data).to(latent_sample.device)
        mat_log_qzi = mat_log_qzi + log_iw_mat.view(batch_size, batch_size, 1)


    log_qz = torch.logsumexp(mat_log_qzi.sum(2), dim=1, keepdim=False) - math.log(batch_size * n_data)
    # mat_log_qz.sum(2): sum across logP(z_i). i.e, \prod P(z_i|x) ==> (256,256) : joint dist of zi|x = z|x
    # logsumexp = sum across all possible pair of (m, s) for each of latent sample : from z|x -> z
    log_prod_qzi = (torch.logsumexp(mat_log_qzi, dim=1, keepdim=False) - math.log(batch_size * n_data)).sum(1)
    # logsumexp = sum across all possible pair of (m, s) for each of latent sample => (256,10): zi|x -> zi
    # and then logsum across z_i => 256: \prod zi

    return log_pz, log_qz, log_prod_qzi, log_q_zCx





def _traverse_discrete_line(dim, size):
    """
    Returns a (size, dim) latent sample, corresponding to a traversal of a
    discrete latent variable.

    Parameters
    ----------
    dim : int
        Number of categories of the selected discrete latent variable.

    traverse : bool
        If True, traverse the categorical variable otherwise keep it fixed
        or randomly sample.

    size : int
        Number of samples to generate.
    """
    samples = np.zeros((size, dim))

    for i in range(size):
        samples[i, i % dim] = 1.

    return torch.Tensor(samples)


def sample_gumbel_softmax(use_cuda, alpha, temperature=.67, train=True):
    """
    Samples from a gumbel-softmax distribution using the reparameterization
    trick.

    Parameters
    ----------
    alpha : torch.Tensor
        Parameters of the gumbel-softmax distribution. Shape (N, D)
    """

    # reparam'ed samples
    EPS = 1e-12
    if train:
        # Sample from gumbel distribution
        unif = torch.rand(alpha.size())
        if use_cuda:
            unif = unif.cuda()
        gumbel = -torch.log(-torch.log(unif + EPS) + EPS)
        # Reparameterize to create gumbel softmax sample
        log_alpha = torch.log(alpha + EPS)
        logit = (log_alpha + gumbel) / temperature
        return F.softmax(logit, dim=1)
    else:
        # In reconstruction mode, pick most likely sample
        _, max_alpha = torch.max(alpha, dim=1)
        one_hot_samples = torch.zeros(alpha.size())
        # On axis 1 of one_hot_samples, scatter the value 1 at indices
        # max_alpha. Note the view is because scatter_ only accepts 2D
        # tensors.
        one_hot_samples.scatter_(1, max_alpha.view(-1, 1).data.cpu(), 1)
        if use_cuda:
            one_hot_samples = one_hot_samples.cuda()
        return one_hot_samples


def kl_multiple_discrete_loss(use_cuda, disc_latents):
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
    kl_losses = [_kl_discrete_loss(use_cuda, alpha) for alpha in disc_latents]

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

def apply_poe(use_cuda, muS_infA, logvarS_infA, muS_infB, logvarS_infB, logalpha, logalphaA, logalphaB):
    '''
    induce zS = encAB(xA,xB) via POE, that is,
        q(zI,zT,zS|xI,xT) := qI(zI|xI) * qT(zT|xT) * q(zS|xI,xT)
            where q(zS|xI,xT) \propto p(zS) * qI(zS|xI) * qT(zS|xT)
    '''

    ZERO = torch.zeros(logvarS_infA.shape)
    if use_cuda:
        ZERO = ZERO.cuda()

    aa=((ZERO - logalpha), -(logvarS_infA - logalphaA), -(logvarS_infB - logalphaB))
    bb= torch.stack(((ZERO - logalpha), -(logvarS_infA - logalphaA), -(logvarS_infB - logalphaB)), dim=2)
    logvarS = -torch.logsumexp(
        torch.stack(((ZERO - logalpha), -(logvarS_infA - logalphaA), -(logvarS_infB - logalphaB)), dim=2),
        dim=2
    )
    stdS = torch.sqrt(torch.exp(logvarS))
    muS = (muS_infA / torch.exp(logvarS_infA) +
           muS_infB / torch.exp(logvarS_infB)) * (stdS ** 2)

    return muS, stdS, logvarS

#-----------------------------------------------------------------------------#

def sample_gaussian(use_cuda, mu, std):

    # reparam'ed samples
    if use_cuda:
        Eps = torch.cuda.FloatTensor(mu.shape).normal_()
    else:
        Eps = torch.randn(mu.shape)
            
    return (mu + Eps*std)


###############################################################################

class DataGather(object):
    '''
    create (array)lists, one for each category, eg, 
      self.data['recon'] = [2.3, 1.5, 0.8, ...],
      self.data['kl'] = [0.3, 1.8, 2.2, ...],
      self.data['acc'] = [0.3, 0.4, 0.5, ...], ...
    '''

    def __init__(self, *args):
        self.keys = args
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return {arg: [] for arg in self.keys}

    def insert(self, **kwargs):
        for key in kwargs.keys():
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()
###############################################################################

def str2bool(v):
    
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#-----------------------------------------------------------------------------#

def grid2gif(img_dir, key, out_gif, delay=100, duration=0.1):

    '''
    make (moving) GIF from images
    '''
    
    if True:  #os.name=='nt':
        
        fnames = [ \
          str(os.path.join(img_dir, f)) for f in os.listdir(img_dir) \
            if (key in f) and ('jpg' in f) ]
        
        fnames.sort()
        
        images = []
        for filename in fnames:
            images.append(imageio.imread(filename))
        
        imageio.mimsave(out_gif, images, duration=duration)
        
    else:  # os.name=='posix'
        
        img_str = str(os.path.join(img_dir, key+'*.jpg'))
        cmd = 'convert -delay %s -loop 0 %s %s' % (delay, img_str, out_gif)
        subprocess.call(cmd, shell=True)


def grid2gif2(img_dir, out_gif, delay=100, duration=0.1):
    '''
    make (moving) GIF from images
    '''

    if True:  # os.name=='nt':

        fnames = [ \
            str(os.path.join(img_dir, f)) for f in os.listdir(img_dir) \
            if ('jpg' in f)]

        fnames.sort()

        images = []
        for filename in fnames:
            images.append(imageio.imread(filename))

        imageio.mimsave(out_gif, images, duration=duration)

    else:  # os.name=='posix'

        img_str = str(os.path.join(img_dir, '*.jpg'))
        cmd = 'convert -delay %s -loop 0 %s %s' % (delay, img_str, out_gif)
        subprocess.call(cmd, shell=True)
#-----------------------------------------------------------------------------#

def mkdirs(path):

    if not os.path.exists(path):
        os.makedirs(path)

#
# def transform(image_size=None, image_size2=None, crop=None):
#     if image_size2:
#         transform = transforms.Compose([
#             transforms.Resize((image_size, image_size2)),
#             transforms.ToTensor()])
#     elif image_size:
#         transform = transforms.Compose([
#           transforms.Resize((image_size, image_size)),
#           transforms.RandomCrop(image_size),
#           transforms.ToTensor()])
#     elif crop:
#         transform = transforms.Compose([
#           transforms.RandomCrop(image_size),
#           transforms.ToTensor()])
#     else:
#         transform = transforms.Compose([
#           transforms.ToTensor()])
#     return transform



def transform(image, resize=None):
    from PIL import Image

    if len(image.shape) ==3:
        image = np.transpose(image, (1, 2, 0))
        image = Image.fromarray(image, mode='RGB')
    else:
        image = Image.fromarray(image, mode='L')
    if resize:
        image = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor()
        ])(image)
    else:
        image = transforms.Compose([
            transforms.ToTensor()
        ])(image)
    return image
