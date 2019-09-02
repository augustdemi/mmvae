import argparse
import numpy as np
import torch

#-----------------------------------------------------------------------------#

from solver import Solver
from utils import str2bool

###############################################################################

# set the random seed manually for reproducibility
SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

###############################################################################
    
def print_opts(opts):
    
    '''
    Print the values of all command-line arguments
    '''
    
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)

#-----------------------------------------------------------------------------#
    
def create_parser():
    
    '''
    Create a parser for command-line arguments
    '''
    
    parser = argparse.ArgumentParser()

    parser.add_argument( '--run_id', default=-1, type=int, 
      help='run id (default=-1 to create a new id)' )

    parser.add_argument( '--cuda', default=True, type=str2bool, 
      help='enable cuda' )
    
    # training hyperparameters
    parser.add_argument( '--max_iter', default=1e7, type=float,
      help='maximum number of batch iterations' )
    parser.add_argument( '--batch_size', default=64, type=int,
      help='batch size' )
    parser.add_argument( '--lamkl', default=1.0, type=float, 
      help='impact of prior kl term' )
    parser.add_argument( '--lr_VAE', default=1e-4, type=float, 
      help='learning rate of the VAE' )
    parser.add_argument( '--beta1_VAE', default=0.9, type=float, 
      help='beta1 parameter of the Adam optimizer for the VAE' )
    parser.add_argument( '--beta2_VAE', default=0.999, type=float, 
      help='beta2 parameter of the Adam optimizer for the VAE' )

    parser.add_argument( '--lr_D', default=1e-4, type=float,
      help='learning rate of the discriminator' )
    parser.add_argument( '--beta1_D', default=0.5, type=float,
      help='beta1 parameter of the Adam optimizer for the discriminator' )
    parser.add_argument( '--beta2_D', default=0.9, type=float,
      help='beta2 parameter of the Adam optimizer for the discriminator' )


    # for TC
    parser.add_argument( '--beta1', default=1.0, type=float,
      help='MI' )
    parser.add_argument( '--beta2', default=1.0, type=float,
      help='TC' )
    parser.add_argument( '--beta3', default=1.0, type=float,
      help='Dim-wise KL' )
    parser.add_argument( '--is_mss', default=False, type=str2bool,
      help='Minibatch Stratified Sampling' )

    # model hyperparameters
    parser.add_argument( '--image_size', default=64, type=int, 
      help='image size; now only (64 x 64) is supported' )
    parser.add_argument( '--zA_dim', default=1, type=int,
      help='dimension of the private-A latent representation' )
    parser.add_argument( '--zB_dim', default=1, type=int,
      help='dimension of the private-B latent representation' )
    parser.add_argument( '--zS_dim', default=1, type=int,
      help='dimension of the shared latent representation' )
    parser.add_argument( '--n_pts', default=100, type=int,
      help='dimension of the private-A latent representation' )
    parser.add_argument( '--n_data', default=10000, type=int,
      help='dimension of the private-A latent representation' )

    # dataset
    parser.add_argument( '--dset_dir', default='data', type=str, 
      help='dataset directory' )
    parser.add_argument( '--dataset', default='CelebA', type=str, 
      help='dataset name' )
    parser.add_argument( '--num_workers', default=2, type=int, 
      help='dataloader num_workers' )
    
    # iter# for previously saved model
    parser.add_argument( '--ckpt_load_iter', default=0, type=int, 
      help='iter# to load the previously saved model ' + 
        '(default=0 to start from the scratch)' )

    # saving directories and checkpoint/sample iterations
    parser.add_argument( '--print_iter', default=20, type=int, 
      help='print losses iter' )
#    parser.add_argument( '--ckpt_dir', default='checkpoints', type=str, 
#      help='checkpoint directory' )
#    parser.add_argument( '--ckpt_load', default=None, type=str, 
#      help='checkpoint name to load' )
    parser.add_argument( '--ckpt_save_iter', default=10000, type=int, 
      help='checkpoint saved every # iters' )
#    parser.add_argument( '--output_dir', default='outputs', type=str, 
#      help='output directory' )
    parser.add_argument( '--output_save_iter', default=50, type=int, 
      help='output saved every # iters' )
#    parser.add_argument( '--output_save', default=True, type=str2bool, 
#      help='whether to save traverse results' )

    parser.add_argument( '--eval_metrics', 
      action='store_true', default=False, 
      help='whether to evaluate disentanglement metrics' )
    parser.add_argument( '--eval_metrics_iter', default=50, type=int, 
      help='evaluate metrics every # iters' )

    # visdom 
    parser.add_argument( '--viz_on', 
      action='store_true', default=True, help='enable visdom visualization' )
    parser.add_argument( '--viz_port', 
      default=8097, type=int, help='visdom port number' )
    parser.add_argument( '--viz_ll_iter', 
      default=1000, type=int, help='visdom line data logging iter' )
    parser.add_argument( '--viz_la_iter', 
      default=5000, type=int, help='visdom line data applying iter' )
    #parser.add_argument( '--viz_ra_iter', 
    #  default=10000, type=int, help='visdom recon image applying iter' )
    #parser.add_argument( '--viz_ta_iter', 
    #  default=10000, type=int, help='visdom traverse applying iter' )

    return parser

#-----------------------------------------------------------------------------#

def main(args):
    
    solver = Solver(args)
    
    solver.train()


###############################################################################
    
if __name__ == "__main__":
    
    parser = create_parser()
    args = parser.parse_args()
    print_opts(args)
    
    main(args)
