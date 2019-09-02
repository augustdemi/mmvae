import os
import numpy as np

import torch.optim as optim
from datasets import DIGIT
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from torchvision import transforms
#-----------------------------------------------------------------------------#

from utils import DataGather, mkdirs, grid2gif, apply_poe, sample_gaussian
from model import *
import json
from dataset import create_dataloader

###############################################################################

class Solver(object):
    
    ####
    def __init__(self, args):
        
        self.args = args

        self.name = '%s_lamkl_%s_zA_%s_zB_%s_zS_%s_HYPER_beta1_%s_beta2_%s_beta3_%s' % \
                    (
                    args.dataset, args.lamkl, args.zA_dim, args.zB_dim, args.zS_dim, args.beta1, args.beta2, args.beta3)
        # to be appended by run_id

        self.use_cuda = args.cuda and torch.cuda.is_available()
         
        self.max_iter = int(args.max_iter)
        
        # do it every specified iters
        self.print_iter = args.print_iter
        self.ckpt_save_iter = args.ckpt_save_iter
        self.output_save_iter = args.output_save_iter
        
        # data info
        self.dset_dir = args.dset_dir
        self.dataset = args.dataset
        self.nc = 3


        # self.N = self.latent_values.shape[0]
        self.eval_metrics_iter = args.eval_metrics_iter
            

        # networks and optimizers
        self.batch_size = args.batch_size
        self.zA_dim = args.zA_dim
        self.zB_dim = args.zB_dim
        self.zS_dim = args.zS_dim
        self.lamkl = args.lamkl
        self.lr_VAE = args.lr_VAE
        self.beta1_VAE = args.beta1_VAE
        self.beta2_VAE = args.beta2_VAE

        self.lr_D = args.lr_D
        self.beta1_D = args.beta1_D
        self.beta2_D = args.beta2_D


        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.beta3 = args.beta3
        self.is_mss = args.is_mss

        # visdom setup
        self.viz_on = args.viz_on
        if self.viz_on:

            self.win_id = dict(
                recon='win_recon', kl='win_kl', alpha='win_alpha', tc='win_TC'
            )
            self.line_gather = DataGather(
                'iter', 'recon_both', 'recon_A', 'recon_B',
                'kl_both', 'kl_A', 'kl_B', 'kl_A_givenB', 'kl_B_givenA',
                'logalpha', 'logalphaA', 'logalphaB',
                'tcA', 'tcB', 'tcS'
            )

            # if self.eval_metrics:
            #     self.win_id['metrics'] = 'win_metrics'

            import visdom

            self.viz_port = args.viz_port  # port number, eg, 8097
            self.viz = visdom.Visdom(port=self.viz_port)
            self.viz_ll_iter = args.viz_ll_iter
            self.viz_la_iter = args.viz_la_iter

            self.viz_init()

        # create dirs: "records", "ckpts", "outputs" (if not exist)
        mkdirs("records");
        mkdirs("ckpts");
        mkdirs("outputs")

        # set run id
        if args.run_id < 0:  # create a new id
            k = 0;
            rfname = os.path.join("records", self.name + '_run_0.txt')
            while os.path.exists(rfname):
                k += 1
                rfname = os.path.join("records", self.name + '_run_%d.txt' % k)
            self.run_id = k
        else:  # user-provided id
            self.run_id = args.run_id

        # finalize name
        self.name = self.name + '_run_' + str(self.run_id)

        # records (text file to store console outputs)
        self.record_file = 'records/%s.txt' % self.name

        # checkpoints
        self.ckpt_dir = os.path.join("ckpts", self.name)

        # outputs
        self.output_dir_recon = os.path.join("outputs", self.name + '_recon')
        # dir for reconstructed images
        self.output_dir_synth = os.path.join("outputs", self.name + '_synth')
        # dir for synthesized images
        self.output_dir_trvsl = os.path.join("outputs", self.name + '_trvsl')

        #### create a new model or load a previously saved model
        
        self.ckpt_load_iter = args.ckpt_load_iter
        self.n_pts = args.n_pts
        self.n_data = args.n_data

        if self.ckpt_load_iter == 0:  # create a new model

            self.encoderA = EncoderA(self.zA_dim, self.zS_dim)
            self.encoderB = EncoderB(self.zB_dim, self.zS_dim)
            self.decoderA = DecoderA(self.zA_dim, self.zS_dim)
            self.decoderB = DecoderB(self.zB_dim, self.zS_dim)
            self.alpha = Alpha()
            self.DA = Discriminator(self.zA_dim + self.zS_dim)
            self.DB = Discriminator(self.zB_dim + self.zS_dim)
                        
        else:  # load a previously saved model
            
            print('Loading saved models (iter: %d)...' % self.ckpt_load_iter)
            self.load_checkpoint()
            print('...done')
            
        if self.use_cuda:
            print('Models moved to GPU...')
            self.encoderA = self.encoderA.cuda()
            self.encoderB = self.encoderB.cuda()            
            self.decoderA = self.decoderA.cuda()
            self.decoderB = self.decoderB.cuda()
            self.alpha = self.alpha.cuda()
            self.DA = self.DA.cuda()
            self.DB = self.DB.cuda()
            print('...done')
        
        # get VAE parameters
        vae_params = \
            list(self.encoderA.parameters()) + \
            list(self.encoderB.parameters()) + \
            list(self.decoderA.parameters()) + \
            list(self.decoderB.parameters()) + \
            list(self.alpha.parameters())

        dis_params = list(self.DA.parameters()) + list(self.DB.parameters())

        # create optimizers
        self.optim_vae = optim.Adam( 
            vae_params, 
            lr=self.lr_VAE, 
            betas=[self.beta1_VAE, self.beta2_VAE] 
        )

        self.optim_dis = optim.Adam( dis_params,
          lr=self.lr_D, betas=[self.beta1_D, self.beta2_D] )




    ####
    def train(self):
        
        self.set_mode(train=True)
            
        # prepare dataloader (iterable)
        print('Start loading data...')
        dset = DIGIT('./data', train=True)
        self.data_loader = torch.utils.data.DataLoader(dset, batch_size=self.batch_size, shuffle=True)
        test_dset = DIGIT('./data', train=False)
        self.test_data_loader = torch.utils.data.DataLoader(test_dset, batch_size=self.batch_size, shuffle=True)
        print('test: ', len(test_dset))
        self.N = len(self.data_loader.dataset)
        print('...done')
        
        # iterators from dataloader
        iterator1 = iter(self.data_loader)
        iterator2 = iter(self.data_loader)

        iter_per_epoch = min(len(iterator1), len(iterator2))
        
        start_iter = self.ckpt_load_iter + 1
        epoch = int(start_iter / iter_per_epoch)
        
        for iteration in range(start_iter, self.max_iter+1):

            # reset data iterators for each epoch
            if iteration % iter_per_epoch == 0:
                print('==== epoch %d done ====' % epoch)
                epoch+=1
                iterator1 = iter(self.data_loader)
                iterator2 = iter(self.data_loader)
                
            #============================================
            #          TRAIN THE VAE (ENC & DEC)
            #============================================

            # sample a mini-batch
            XA, XB, index = next(iterator1)  # (n x C x H x W)
            NA = 1*28*28
            NB = 3*32*32


            index = index.cpu().detach().numpy()
            if self.use_cuda:
                XA = XA.cuda()
                XB = XB.cuda()

            # zA, zS = encA(xA)
            muA_infA, stdA_infA, logvarA_infA, \
            muS_infA, stdS_infA, logvarS_infA = self.encoderA(XA)

            # zB, zS = encB(xB)
            muB_infB, stdB_infB, logvarB_infB, \
            muS_infB, stdS_infB, logvarS_infB = self.encoderB(XB)

            # read current values
            logalpha, logalphaA, logalphaB = self.alpha()

            # zS = encAB(xA,xB) via POE
            muS_POE, stdS_POE, logvarS_POE = apply_poe(
                self.use_cuda, muS_infA, logvarS_infA, muS_infB, logvarS_infB, logalpha, logalphaA, logalphaB
            )

            # kl losses

            loss_kl_infA = -0.5*(
              (1 + logvarA_infA - muA_infA**2 - stdA_infA**2).sum(1).mean() +
              (1 + logvarS_infA - muS_infA**2 - stdS_infA**2).sum(1).mean()
            )
            loss_kl_infB = -0.5*(
              (1 + logvarB_infB - muB_infB**2 - stdB_infB**2).sum(1).mean() +
              (1 + logvarS_infB - muS_infB**2 - stdS_infB**2).sum(1).mean()
            )
            loss_kl_POE = -0.5*(
              (1 + logvarA_infA - muA_infA**2 - stdA_infA**2).sum(1).mean() +
              (1 + logvarB_infB - muB_infB**2 - stdB_infB**2).sum(1).mean() +
              (1 + logvarS_POE - muS_POE**2 - stdS_POE**2).sum(1).mean()
            )
            ###################################################### CONDITIONAL ######################################################
            loss_kl_infA_givenB = -0.5 * (
                    (1 + logvarA_infA - muA_infA ** 2 - stdA_infA ** 2).sum(1).mean() +
                    (1 + logvarS_POE - logvarS_infB - ((muS_POE - muS_infB) ** 2 + stdS_POE ** 2) / stdS_infB ** 2).sum(1).mean()
            )

            loss_kl_infB_givenA = -0.5 * (
                    (1 + logvarB_infB - muB_infB ** 2 - stdB_infB ** 2).sum(1).mean() +
                    (1 + logvarS_POE - logvarS_infA - ((muS_POE - muS_infA) ** 2 + stdS_POE ** 2) / stdS_infA ** 2).sum(1).mean()
            )
            ##########################################################################################################################

            loss_kl = loss_kl_infA

            # encoder samples (for training)
            ZA_infA = sample_gaussian(self.use_cuda, muA_infA, stdA_infA)
            ZB_infB = sample_gaussian(self.use_cuda, muB_infB, stdB_infB)
            ZS_POE = sample_gaussian(self.use_cuda, muS_POE, stdS_POE)

            # encoder samples (for cross-modal prediction)
            ZS_infA = sample_gaussian(self.use_cuda, muS_infA, stdS_infA)
            ZS_infB = sample_gaussian(self.use_cuda, muS_infB, stdS_infB)

            # reconstructed samples (given joint modal observation)
            XA_POE_recon = self.decoderA(ZA_infA, ZS_POE)
            XB_POE_recon = self.decoderB(ZB_infB, ZS_POE)

            # reconstructed samples (given single modal observation)
            XA_infA_recon = self.decoderA(ZA_infA, ZS_infA)
            XB_infB_recon = self.decoderB(ZB_infB, ZS_infB)

            loss_recon_infA = F.l1_loss(torch.sigmoid(XA_infA_recon), XA, reduction='sum').div(XA.size(0))
            #
            loss_recon_infB = F.l1_loss(torch.sigmoid(XB_infB_recon), XB, reduction='sum').div(XB.size(0))
            #
            loss_recon_POE = \
                F.l1_loss(torch.sigmoid(XA_POE_recon), XA, reduction='sum').div(XA.size(0)) + \
                F.l1_loss(torch.sigmoid(XB_POE_recon), XB, reduction='sum').div(XB.size(0))
            #

            ###################################################### CONDITIONAL ######################################################
            # conditional
            loss_recon_infA_givenB = F.l1_loss(torch.sigmoid(XA_POE_recon), XA, reduction='sum').div(XA.size(0))


            loss_recon_infB_givenA = F.l1_loss(torch.sigmoid(XB_POE_recon), XB, reduction='sum').div(XB.size(0))
            ##########################################################################################################################

            loss_recon = loss_recon_infA

            ###################################################### TC ######################################################
            ZA = torch.cat([ZA_infA, ZS_infA], dim=1)
            ZB = torch.cat([ZB_infB, ZS_infB], dim=1)

            # dis(Z)
            DZA = self.DA(ZA)
            DZB = self.DB(ZB)

            # tc loss
            tcA_loss = (DZA[:, 0] - DZA[:, 1]).mean()
            tcB_loss = (DZB[:, 0] - DZB[:, 1]).mean()

            ##########################################################################################################################

            # total loss for vae
            vae_loss = loss_recon + self.lamkl * loss_kl + self.beta1 * tcA_loss
            # vae_loss = self.beta1 * tcA_loss + self.beta2 * tcB_loss

            # update vae
            self.optim_vae.zero_grad()
            vae_loss.backward()
            self.optim_vae.step()

            # ============================================
            #          TRAIN THE DISCRIMINATOR
            # ============================================

            # sample a mini-batch
            XA2, XB2, _ = next(iterator2)  # (n x C x H x W)
            if self.use_cuda:
                XA2 = XA2.cuda()
                XB2 = XB2.cuda()

            # zA, zS = encA(xA)
            muA_infA2, stdA_infA2, _, \
            muS_infA2, stdS_infA2, _ = self.encoderA(XA2)

            # zB, zS = encB(xB)
            muB_infB2, stdB_infB2, _, \
            muS_infB2, stdS_infB2, _ = self.encoderB(XB2)

            ZA_infA2 = sample_gaussian(self.use_cuda, muA_infA2, stdA_infA2)
            ZB_infB2 = sample_gaussian(self.use_cuda, muB_infB2, stdB_infB2)

            # encoder samples (for cross-modal prediction)
            ZS_infA2 = sample_gaussian(self.use_cuda, muS_infA2, stdS_infA2)
            ZS_infB2 = sample_gaussian(self.use_cuda, muS_infB2, stdS_infB2)

            ZA2 = torch.cat([ZA_infA2, ZS_infA2], dim=1)
            ZB2 = torch.cat((ZB_infB2, ZS_infB2), dim=1)

            DZA2 = self.DA(ZA2)
            DZB2 = self.DB(ZB2)

            # A: dim-wise permutated Z over the mini-batch
            perm_ZA = []
            for zj in ZA2.split(1, 1):
                idx = torch.randperm(ZA2.size(0))
                perm_zj = zj[idx]
                perm_ZA.append(perm_zj)
            ZA_perm = torch.cat(perm_ZA, 1)
            ZA_perm = ZA_perm.detach()

            # dis(Z_perm)
            DZA_perm = self.DA(ZA_perm)

            ones = torch.ones(ZA2.size(0), dtype=torch.long)
            zeros = torch.zeros(ZA2.size(0), dtype=torch.long)
            if self.use_cuda:
                ones = ones.cuda()
                zeros = zeros.cuda()
            # discriminator loss
            disA_loss = 0.5 * (F.cross_entropy(DZA2, zeros) +
                               F.cross_entropy(DZA_perm, ones))

            # B
            # perm_ZB = []
            # for zj in ZB2.split(1, 1):
            #     idx = torch.randperm(ZB2.size(0))
            #     perm_zj = zj[idx]
            #     perm_ZB.append(perm_zj)
            # ZB_perm = torch.cat(perm_ZB, 1)
            # ZB_perm = ZB_perm.detach()
            #
            # # dis(Z_perm)
            # DZB_perm = self.DB(ZB_perm)
            #
            # ones = torch.ones(ZB2.size(0), dtype=torch.long)
            # zeros = torch.zeros(ZB2.size(0), dtype=torch.long)
            # if self.use_cuda:
            #     ones = ones.cuda()
            #     zeros = zeros.cuda()
            #
            # # discriminator loss
            # disB_loss = 0.5 * (F.cross_entropy(DZB2, zeros) +
            #                    F.cross_entropy(DZB_perm, ones))





            dis_loss = disA_loss
            # update discriminator
            self.optim_dis.zero_grad()
            dis_loss.backward()
            self.optim_dis.step()

            tcS_loss = tcA_loss
            ####################

            ####################

            # print the losses
            if iteration % self.print_iter == 0:
                prn_str = ( \
                  '[iter %d (epoch %d)] vae_loss: %.3f ' + \
                  '(recon: %.3f, kl: %.3f)\n' + \
                  '    rec_infA = %.3f, rec_infB = %.3f, rec_POE = %.3f\n' + \
                  '    kl_infA = %.3f, kl_infB = %.3f, kl_POE = %.3f\n' + \
                  '    kl_A/B = %.3f, kl_B/A = %.3f\n' + \
                  '    log(XA|XB) = %.3f, log(XB|XA) = %.3f\n' + \
                  '    tcA_loss = %.3f, tcB_loss = %.3f, tcS_loss = %.3f\n'
                              ) % \
                  (iteration, epoch,
                    vae_loss.item(), loss_recon.item(), loss_kl.item(),
                    loss_recon_infA.item(), loss_recon_infB.item(),
                      loss_recon_POE.item(),
                    loss_kl_infA.item(), loss_kl_infB.item(),
                      loss_kl_POE.item(), loss_kl_infA_givenB.item(), loss_kl_infB_givenA.item(),
                     -loss_recon_infA_givenB.item(), -loss_recon_infB_givenA.item(),
                    tcA_loss, tcB_loss, tcS_loss
                  )
                print(prn_str)
                if self.record_file:
                    record = open(self.record_file, 'a')
                    record.write('%s\n' % (prn_str,))
                    record.close()


            # save model parameters
            if iteration % self.ckpt_save_iter == 0:
               self.save_checkpoint(iteration)
               
            # save output images (recon, synth, etc.)
            if iteration % self.output_save_iter == 0:

                # self.save_embedding(iteration, index, muA_infA, muB_infB, muS_infA, muS_infB, muS_POE)

                # 1) save the recon images
                self.save_recon(iteration, logalpha, logalphaA, logalphaB)


                # self.save_recon2(iteration, index, XA, XB,
                #     torch.sigmoid(XA_infA_recon).data,
                #     torch.sigmoid(XB_infB_recon).data,
                #     torch.sigmoid(XA_POE_recon).data,
                #     torch.sigmoid(XB_POE_recon).data,
                #     muA_infA, muB_infB, muS_infA, muS_infB, muS_POE,
                #     logalpha, logalphaA, logalphaB
                # )
                z_A, z_B, z_S = self.get_stat(logalpha, logalphaA, logalphaB)

                #
                #
                #
                # # 2) save the pure-synthesis images
                # # self.save_synth_pure( iteration, howmany=100 )
                # #
                # # 3) save the cross-modal-synthesis images
                # self.save_synth_cross_modal(iteration, z_A, z_B, howmany=3)
                #
                # # 4) save the latent traversed images
                # self.save_traverse(iteration, logalpha, logalphaA, logalphaB, z_A, z_B, z_S)

                # self.get_loglike(logalpha, logalphaA, logalphaB)


                # # 3) save the latent traversed images
                # if self.dataset.lower() == '3dchairs':
                #     self.save_traverse(iteration, limb=-2, limu=2, inter=0.5)
                # else:
                #     self.save_traverse(iteration, limb=-3, limu=3, inter=0.1)

            if iteration % self.eval_metrics_iter == 0:
                self.save_synth_cross_modal(iteration, z_A, z_B, train=False, howmany=3)

            # (visdom) insert current line stats
            if self.viz_on and (iteration % self.viz_ll_iter == 0):
                self.line_gather.insert( iter=iteration, 
                    recon_both=loss_recon_POE.item(), 
                    recon_A=loss_recon_infA.item(), 
                    recon_B=loss_recon_infB.item(),
                    kl_both=loss_kl_POE.item(),
                    kl_A=loss_kl_infA.item(),
                    kl_B=loss_kl_infB.item(),
                    kl_A_givenB = loss_kl_infA_givenB.item(),
                    kl_B_givenA = loss_kl_infB_givenA.item(),
                    logalpha=logalpha.item(),
                    logalphaA=logalphaA.item(),
                    logalphaB=logalphaB.item(),
                 tcA=tcA_loss.item(),
                 tcB=tcB_loss.item(),
                 tcS=tcS_loss.item(),
                )

            # (visdom) visualize line stats (then flush out)
            if self.viz_on and (iteration % self.viz_la_iter == 0):
                self.visualize_line()
                self.line_gather.flush()

            # evaluate metrics
            # if self.eval_metrics and (iteration % self.eval_metrics_iter == 0):
            #
            #     metric1, _ = self.eval_disentangle_metric1()
            #     metric2, _ = self.eval_disentangle_metric2()
            #
            #     prn_str = ( '********\n[iter %d (epoch %d)] ' + \
            #       'metric1 = %.4f, metric2 = %.4f\n********' ) % \
            #       (iteration, epoch, metric1, metric2)
            #     print(prn_str)
            #     if self.record_file:
            #         record = open(self.record_file, 'a')
            #         record.write('%s\n' % (prn_str,))
            #         record.close()
            #
            #     # (visdom) visulaize metrics
            #     if self.viz_on:
            #         self.visualize_line_metrics(iteration, metric1, metric2)
            #

    def get_loglike(self, logalpha, logalphaA, logalphaB):
        self.set_mode(train=False)


        # fixed_idxs = [1000, 7246, 21440, 1000]
        np.random.seed(3)
        fixed_idxs = np.random.randint(0,self.N,1000)
        print(fixed_idxs[:10])

        fixed_XA = [0] * len(fixed_idxs)
        fixed_XB = [0] * len(fixed_idxs)


        for i, idx in enumerate(fixed_idxs):

            fixed_XA[i], fixed_XB[i] = \
                self.data_loader.dataset.__getitem__(idx)[0:2]

            if self.use_cuda:
                fixed_XA[i] = fixed_XA[i].cuda()
                fixed_XB[i] = fixed_XB[i].cuda()

        fixed_XA = torch.stack(fixed_XA)
        fixed_XB = torch.stack(fixed_XB)

        muA_infA, stdA_infA, logvarA_infA, \
        muS_infA, stdS_infA, logvarS_infA = self.encoderA(fixed_XA)

        # zB, zS = encB(xB)
        muB_infB, stdB_infB, logvarB_infB, \
        muS_infB, stdS_infB, logvarS_infB = self.encoderB(fixed_XB)


        # zS = encAB(xA,xB) via POE
        muS_POE, stdS_POE, logvarS_POE = apply_poe(
            self.use_cuda, muS_infA, logvarS_infA, muS_infB, logvarS_infB, logalpha, logalphaA, logalphaB
        )

        ZA_infA = sample_gaussian(self.use_cuda, muA_infA, stdA_infA)
        ZB_infB = sample_gaussian(self.use_cuda, muB_infB, stdB_infB)
        ZS_POE = sample_gaussian(self.use_cuda, muS_POE, stdS_POE)
        if self.use_cuda:
            ZA_infA = ZA_infA.cuda()
            ZB_infB = ZB_infB.cuda()
            ZS_POE = ZS_POE.cuda()

        XA_POE_recon = self.decoderA(ZA_infA, ZS_POE)
        XB_POE_recon = self.decoderB(ZB_infB, ZS_POE)

        ###################################################### CONDITIONAL ######################################################
        loss_kl_infA_givenB = -0.5 * (
                (1 + logvarA_infA - muA_infA ** 2 - stdA_infA ** 2).sum(1).mean() +
                (1 + logvarS_POE - logvarS_infB - ((muS_POE - muS_infB) ** 2 + stdS_POE ** 2) / stdS_infB ** 2).sum(
                    1).mean()
        )

        loss_kl_infB_givenA = -0.5 * (
                (1 + logvarB_infB - muB_infB ** 2 - stdB_infB ** 2).sum(1).mean() +
                (1 + logvarS_POE - logvarS_infA - ((muS_POE - muS_infA) ** 2 + stdS_POE ** 2) / stdS_infA ** 2).sum(
                    1).mean()
        )
        # conditional
        loss_recon_infA_givenB = F.l1_loss(torch.sigmoid(XA_POE_recon), fixed_XA, reduction='sum').div(fixed_XA.size(0))

        loss_recon_infB_givenA = F.l1_loss(torch.sigmoid(XB_POE_recon), fixed_XB, reduction='sum').div(fixed_XB.size(0))
        ##########################################################################################################################

        xa_xb, xb_xa = -(loss_recon_infA_givenB.item() + loss_kl_infA_givenB.item()),\
                       -(loss_recon_infB_givenA.item() + loss_kl_infB_givenA.item())
        print('xa_xb:',xa_xb)
        print('xb_xa:',xb_xa)



    ####
    def eval_disentangle_metric1(self):
        
        # some hyperparams
        num_pairs = 800  # # data pairs (d,y) for majority vote classification
        bs = 50  # batch size
        nsamps_per_factor = 100  # samples per factor
        nsamps_agn_factor = 5000  # factor-agnostic samples
        
        self.set_mode(train=False)
        
        # 1) estimate variances of latent points factor agnostic
        
        dl = DataLoader( 
          self.data_loader.dataset, batch_size=bs,
          shuffle=True, num_workers=self.args.num_workers, pin_memory=True )
        iterator = iter(dl)
        
        M = []
        for ib in range(int(nsamps_agn_factor/bs)):
            
            # sample a mini-batch
            XAb, XBb, _, _, _ = next(iterator)  # (bs x C x H x W)
            if self.use_cuda:
                XAb = XAb.cuda()
                XBb = XBb.cuda()
                
            # z = encA(xA)
            mu_infA, _, logvar_infA = self.encoderA(XAb)
            
            # z = encB(xB)
            mu_infB, _, logvar_infB = self.encoderB(XBb)
              
            # z = encAB(xA,xB) via POE
            mu_POE, _, _ = apply_poe(
                self.use_cuda, mu_infA, logvar_infA, mu_infB, logvar_infB,
            )
            
            mub = mu_POE
                        
            M.append(mub.cpu().detach().numpy())
            
        M = np.concatenate(M, 0)
        
        # estimate sample vairance and mean of latent points for each dim
        vars_agn_factor = np.var(M, 0)

        # 2) estimatet dim-wise vars of latent points with "one factor fixed"

        factor_ids = range(0, len(self.latent_sizes))  # true factor ids
        vars_per_factor = np.zeros([num_pairs, self.z_dim])  
        true_factor_ids = np.zeros(num_pairs, np.int)  # true factor ids

        # prepare data pairs for majority-vote classification
        i = 0
        for j in factor_ids:  # for each factor

            # repeat num_paris/num_factors times
            for r in range(int(num_pairs/len(factor_ids))):

                # a true factor (id and class value) to fix
                fac_id = j
                fac_class = np.random.randint(self.latent_sizes[fac_id])

                # randomly select images (with the fixed factor)
                indices = np.where( 
                  self.latent_classes[:,fac_id]==fac_class )[0]
                np.random.shuffle(indices)
                idx = indices[:nsamps_per_factor]
                M = []
                for ib in range(int(nsamps_per_factor/bs)):
                    XAb, XBb, _, _, _ = dl.dataset[ idx[(ib*bs):(ib+1)*bs] ]
                    if XAb.shape[0]<1:  # no more samples
                        continue;
                    if self.use_cuda:
                        XAb = XAb.cuda()
                        XBb = XBb.cuda()                    
                    mu_infA, _, logvar_infA = self.encoderA(XAb)
                    mu_infB, _, logvar_infB = self.encoderB(XBb)
                    mu_POE, _, _ = apply_poe( self.use_cuda, 
                        mu_infA, logvar_infA, mu_infB, logvar_infB,
                    )
                    mub = mu_POE
                    M.append(mub.cpu().detach().numpy())
                M = np.concatenate(M, 0)
                                
                # estimate sample var and mean of latent points for each dim
                if M.shape[0]>=2:
                    vars_per_factor[i,:] = np.var(M, 0)
                else:  # not enough samples to estimate variance
                    vars_per_factor[i,:] = 0.0                
                
                # true factor id (will become the class label)
                true_factor_ids[i] = fac_id

                i += 1
                
        # 3) evaluate majority vote classification accuracy
 
        # inputs in the paired data for classification
        smallest_var_dims = np.argmin(
          vars_per_factor / (vars_agn_factor + 1e-20), axis=1 )
    
        # contingency table
        C = np.zeros([self.z_dim,len(factor_ids)])
        for i in range(num_pairs):
            C[ smallest_var_dims[i], true_factor_ids[i] ] += 1
        
        num_errs = 0  # # misclassifying errors of majority vote classifier
        for k in range(self.z_dim):
            num_errs += np.sum(C[k,:]) - np.max(C[k,:])
        
        metric1 = (num_pairs - num_errs) / num_pairs  # metric = accuracy
        
        self.set_mode(train=True)

        return metric1, C
    
    
    ####
    def eval_disentangle_metric2(self):
        
        # some hyperparams
        num_pairs = 800  # # data pairs (d,y) for majority vote classification
        bs = 50  # batch size
        nsamps_per_factor = 100  # samples per factor
        nsamps_agn_factor = 5000  # factor-agnostic samples     
        
        self.set_mode(train=False)
        
        # 1) estimate variances of latent points factor agnostic
        
        dl = DataLoader( 
          self.data_loader.dataset, batch_size=bs,
          shuffle=True, num_workers=self.args.num_workers, pin_memory=True )
        iterator = iter(dl)
        
        M = []
        for ib in range(int(nsamps_agn_factor/bs)):
            
            # sample a mini-batch
            XAb, XBb, _, _, _ = next(iterator)  # (bs x C x H x W)
            if self.use_cuda:
                XAb = XAb.cuda()
                XBb = XBb.cuda()
                
            # z = encA(xA)
            mu_infA, _, logvar_infA = self.encoderA(XAb)
            
            # z = encB(xB)
            mu_infB, _, logvar_infB = self.encoderB(XBb)
              
            # z = encAB(xA,xB) via POE
            mu_POE, _, _ = apply_poe(
                self.use_cuda, mu_infA, logvar_infA, mu_infB, logvar_infB,
            )
            
            mub = mu_POE

            M.append(mub.cpu().detach().numpy())
            
        M = np.concatenate(M, 0)
        
        # estimate sample vairance and mean of latent points for each dim
        vars_agn_factor = np.var(M, 0)

        # 2) estimatet dim-wise vars of latent points with "one factor varied"

        factor_ids = range(0, len(self.latent_sizes))  # true factor ids
        vars_per_factor = np.zeros([num_pairs, self.z_dim])
        true_factor_ids = np.zeros(num_pairs, np.int)  # true factor ids

        # prepare data pairs for majority-vote classification
        i = 0
        for j in factor_ids:  # for each factor

            # repeat num_paris/num_factors times
            for r in range(int(num_pairs/len(factor_ids))):
                                
                # randomly choose true factors (id's and class values) to fix
                fac_ids = list(np.setdiff1d(factor_ids,j))
                fac_classes = \
                  [ np.random.randint(self.latent_sizes[k]) for k in fac_ids ]

                # randomly select images (with the other factors fixed)
                if len(fac_ids)>1:
                    indices = np.where( 
                      np.sum(self.latent_classes[:,fac_ids]==fac_classes,1)
                      == len(fac_ids) 
                    )[0]
                else:
                    indices = np.where(
                      self.latent_classes[:,fac_ids]==fac_classes 
                    )[0]
                np.random.shuffle(indices)
                idx = indices[:nsamps_per_factor]
                M = []
                for ib in range(int(nsamps_per_factor/bs)):                    
                    XAb, XBb, _, _, _ = dl.dataset[ idx[(ib*bs):(ib+1)*bs] ]
                    if XAb.shape[0]<1:  # no more samples
                        continue;
                    if self.use_cuda:
                        XAb = XAb.cuda()
                        XBb = XBb.cuda()
                    mu_infA, _, logvar_infA = self.encoderA(XAb)
                    mu_infB, _, logvar_infB = self.encoderB(XBb)
                    mu_POE, _, _ = apply_poe( self.use_cuda, 
                        mu_infA, logvar_infA, mu_infB, logvar_infB,
                    )
                    mub = mu_POE
                    M.append(mub.cpu().detach().numpy())
                M = np.concatenate(M, 0)
                
                # estimate sample var and mean of latent points for each dim
                if M.shape[0]>=2:
                    vars_per_factor[i,:] = np.var(M, 0)
                else:  # not enough samples to estimate variance
                    vars_per_factor[i,:] = 0.0
                    
                # true factor id (will become the class label)
                true_factor_ids[i] = j

                i += 1
                
        # 3) evaluate majority vote classification accuracy
            
        # inputs in the paired data for classification
        largest_var_dims = np.argmax(
          vars_per_factor / (vars_agn_factor + 1e-20), axis=1 )
    
        # contingency table
        C = np.zeros([self.z_dim,len(factor_ids)])
        for i in range(num_pairs):
            C[ largest_var_dims[i], true_factor_ids[i] ] += 1
        
        num_errs = 0  # # misclassifying errors of majority vote classifier
        for k in range(self.z_dim):
            num_errs += np.sum(C[k,:]) - np.max(C[k,:])
    
        metric2 = (num_pairs - num_errs) / num_pairs  # metric = accuracy    
        
        self.set_mode(train=True)

        return metric2, C



    def save_recon(self, iters, logalpha, logalphaA, logalphaB):
        self.set_mode(train=False)


        mkdirs(self.output_dir_recon)

        fixed_idxs = [3246, 7000, 14305, 19000, 27444, 33100, 38000, 45231, 51000, 55121]

        fixed_idxs60 = []
        for idx in fixed_idxs:
            for i in range(6):
                fixed_idxs60.append(idx + i)


        XA = [0] * len(fixed_idxs60)
        XB = [0] * len(fixed_idxs60)


        for i, idx in enumerate(fixed_idxs60):
            XA[i], XB[i] = \
                self.data_loader.dataset.__getitem__(idx)[0:2]

            if self.use_cuda:
                XA[i] = XA[i].cuda()
                XB[i] = XB[i].cuda()

        XA = torch.stack(XA)
        XB = torch.stack(XB)

        muA_infA, stdA_infA, logvarA_infA, \
        muS_infA, stdS_infA, logvarS_infA = self.encoderA(XA)

        # zB, zS = encB(xB)
        muB_infB, stdB_infB, logvarB_infB, \
        muS_infB, stdS_infB, logvarS_infB = self.encoderB(XB)


        # zS = encAB(xA,xB) via POE
        muS_POE, stdS_POE, logvarS_POE = apply_poe(
            self.use_cuda, muS_infA, logvarS_infA, muS_infB, logvarS_infB, logalpha, logalphaA, logalphaB
        )

        # encoder samples (for training)
        ZA_infA = sample_gaussian(self.use_cuda, muA_infA, stdA_infA)
        ZB_infB = sample_gaussian(self.use_cuda, muB_infB, stdB_infB)
        ZS_POE = sample_gaussian(self.use_cuda, muS_POE, stdS_POE)

        # encoder samples (for cross-modal prediction)
        ZS_infA = sample_gaussian(self.use_cuda, muS_infA, stdS_infA)
        ZS_infB = sample_gaussian(self.use_cuda, muS_infB, stdS_infB)

        # reconstructed samples (given joint modal observation)
        XA_POE_recon = torch.sigmoid(self.decoderA(ZA_infA, ZS_POE))
        XB_POE_recon = torch.sigmoid(self.decoderB(ZB_infB, ZS_POE))

        # reconstructed samples (given single modal observation)
        XA_infA_recon = torch.sigmoid(self.decoderA(ZA_infA, ZS_infA))
        XB_infB_recon = torch.sigmoid(self.decoderB(ZB_infB, ZS_infB))



        WS = torch.ones(XA.shape)
        if self.use_cuda:
            WS = WS.cuda()

        n = XA.shape[0]
        perm = torch.arange(0, 4 * n).view(4, n).transpose(1, 0)
        perm = perm.contiguous().view(-1)

        ## img
        # merged = torch.cat(
        #     [ XA, XB, XA_infA_recon, XB_infB_recon,
        #       XA_POE_recon, XB_POE_recon, WS ], dim=0
        # )
        merged = torch.cat(
            [XA, XA_infA_recon, XA_POE_recon, WS], dim=0
        )
        merged = merged[perm, :].cpu()

        # save the results as image
        fname = os.path.join(self.output_dir_recon, 'reconA_%s.jpg' % iters)
        mkdirs(self.output_dir_recon)
        save_image(
            tensor=merged, filename=fname, nrow=7 * int(np.sqrt(n)),
            pad_value=1
        )

        WS = torch.ones(XB.shape)
        if self.use_cuda:
            WS = WS.cuda()

        n = XB.shape[0]
        perm = torch.arange(0, 4 * n).view(4, n).transpose(1, 0)
        perm = perm.contiguous().view(-1)

        ## ingr
        merged = torch.cat(
            [XB, XB_infB_recon, XB_POE_recon, WS], dim=0
        )
        merged = merged[perm, :].cpu()

        # save the results as image
        fname = os.path.join(self.output_dir_recon, 'reconB_%s.jpg' % iters)
        mkdirs(self.output_dir_recon)
        save_image(
            tensor=merged, filename=fname, nrow=7 * int(np.sqrt(n)),
            pad_value=1
        )
        self.set_mode(train=True)




    def save_recon2(self, iters, index, XA, XB,
        XA_infA_recon, XB_infB_recon, XA_POE_recon, XB_POE_recon, muA_infA, muB_infB, muS_infA, muS_infB, muS_POE,
                   logalpha, logalphaA, logalphaB):

        muA_infA, muB_infB, muS_infA, muS_infB, muS_POE = muA_infA.cpu().detach().numpy(), muB_infB.cpu().detach().numpy(), \
                                                          muS_infA.cpu().detach().numpy(), muS_infB.cpu().detach().numpy(), \
                                                          muS_POE.cpu().detach().numpy()

        emb_info = {}
        emb_info['muA_infA'] = muA_infA.tolist()
        emb_info['muB_infB'] = muB_infB.tolist()
        emb_info['muS_infA'] = muS_infA.tolist()
        emb_info['muS_infB'] = muS_infB.tolist()
        emb_info['muS_POE'] = muS_POE.tolist()

        fname = os.path.join(self.output_dir_recon, 'embedding_%s.json' % iters)
        mkdirs(self.output_dir_recon)

        with open(fname, 'w') as outfile:
            json.dump(emb_info, outfile)

        WS = torch.ones(XA.shape)
        if self.use_cuda:
            WS = WS.cuda()

        n = XA.shape[0]
        perm = torch.arange(0, 4 * n).view(4, n).transpose(1, 0)
        perm = perm.contiguous().view(-1)

        ## img
        # merged = torch.cat(
        #     [ XA, XB, XA_infA_recon, XB_infB_recon,
        #       XA_POE_recon, XB_POE_recon, WS ], dim=0
        # )
        merged = torch.cat(
            [XA, XA_infA_recon, XA_POE_recon, WS], dim=0
        )
        merged = merged[perm, :].cpu()

        # save the results as image
        fname = os.path.join(self.output_dir_recon, 'reconA_%s.jpg' % iters)
        mkdirs(self.output_dir_recon)
        save_image(
            tensor=merged, filename=fname, nrow=7 * int(np.sqrt(n)),
            pad_value=1
        )

        WS = torch.ones(XB.shape)
        if self.use_cuda:
            WS = WS.cuda()

        n = XB.shape[0]
        perm = torch.arange(0, 4 * n).view(4, n).transpose(1, 0)
        perm = perm.contiguous().view(-1)

        ## ingr
        merged = torch.cat(
            [XB, XB_infB_recon, XB_POE_recon, WS], dim=0
        )
        merged = merged[perm, :].cpu()

        # save the results as image
        fname = os.path.join(self.output_dir_recon, 'reconB_%s.jpg' % iters)
        mkdirs(self.output_dir_recon)
        save_image(
            tensor=merged, filename=fname, nrow=7 * int(np.sqrt(n)),
            pad_value=1
        )




    ####
    def save_synth_pure(self, iters, howmany=100):
        
        self.set_mode(train=False)

        decoderA = self.decoderA
        decoderB = self.decoderB
        
        Z = torch.randn(howmany, self.z_dim)
        if self.use_cuda:
            Z = Z.cuda()
    
        # do synthesis 
        XA = torch.sigmoid(decoderA(Z)).data
        XB = torch.sigmoid(decoderB(Z)).data
        
        WS = torch.ones(XA.shape)
        if self.use_cuda:
            WS = WS.cuda()
        
        perm = torch.arange(0,3*howmany).view(3,howmany).transpose(1,0)
        perm = perm.contiguous().view(-1)
        merged = torch.cat([XA, XB, WS], dim=0)
        merged = merged[perm,:].cpu()
    
        # save the results as image
        fname = os.path.join(
            self.output_dir_synth, 'synth_pure_%s.jpg' % iters
        )
        mkdirs(self.output_dir_synth)
        save_image( 
          tensor=merged, filename=fname, nrow=3*int(np.sqrt(howmany)), 
          pad_value=1
        )

        self.set_mode(train=True)
        
    
    ####
    def save_synth_cross_modal(self, iters, z_A_stat, z_B_stat, train=True, howmany=3):

        self.set_mode(train=False)


        if train:
            data_loader = self.data_loader
            fixed_idxs = [3246, 7001, 14308, 19000, 27447, 33103, 38002, 45232, 51000, 55125]
        else:
            data_loader = self.test_data_loader
            fixed_idxs = [2, 982, 2300, 3400, 4500, 5500, 6500, 7500, 8500, 9500]



        fixed_XA = [0] * len(fixed_idxs)
        fixed_XB = [0] * len(fixed_idxs)


        for i, idx in enumerate(fixed_idxs):

            fixed_XA[i], fixed_XB[i] = \
                data_loader.dataset.__getitem__(idx)[0:2]

            if self.use_cuda:
                fixed_XA[i] = fixed_XA[i].cuda()
                fixed_XB[i] = fixed_XB[i].cuda()

        fixed_XA = torch.stack(fixed_XA)
        fixed_XB = torch.stack(fixed_XB)

        _, _, _, \
        muS_infA, stdS_infA, logvarS_infA = self.encoderA(fixed_XA)
        _, _, _, \
        muS_infB, stdS_infB, logvarS_infB = self.encoderB(fixed_XB)

        ZS_infA = sample_gaussian(self.use_cuda, muS_infA, stdS_infA)
        ZS_infB = sample_gaussian(self.use_cuda, muS_infB, stdS_infB)

        if self.use_cuda:
            ZS_infA = ZS_infA.cuda()
            ZS_infB = ZS_infB.cuda()

        decoderA = self.decoderA
        decoderB = self.decoderB


        # mkdirs(os.path.join(self.output_dir_synth, str(iters)))


        fixed_XA_3ch = []
        for i in range(len(fixed_XA)):
            each_XA = fixed_XA[i].clone().squeeze()
            fixed_XA_3ch.append(torch.stack([each_XA, each_XA, each_XA]))

        fixed_XA_3ch = torch.stack(fixed_XA_3ch)

        WS = torch.ones(fixed_XA_3ch.shape)
        if self.use_cuda:
            WS = WS.cuda()

        n = len(fixed_idxs)

        perm = torch.arange(0, (howmany + 2) * n).view(howmany + 2, n).transpose(1, 0)
        perm = perm.contiguous().view(-1)

        ######## 1) generate xB from given xA (A2B) ########



        merged = torch.cat([fixed_XA_3ch], dim=0)
        for k in range(howmany):
            # z_B_stat = np.array(z_B_stat)
            # z_B_stat_mean = np.mean(z_B_stat, 0)
            # ZB = torch.Tensor(z_B_stat_mean)
            # ZB_list = []
            # for _ in range(n):
            #     ZB_list.append(ZB)
            # ZB = torch.stack(ZB_list)


            ZB = torch.randn(n, self.zB_dim)
            z_B_stat = np.array(z_B_stat)
            z_B_stat_mean = np.mean(z_B_stat, 0)
            ZB = ZB + torch.Tensor(z_B_stat_mean)

            if self.use_cuda:
                ZB = ZB.cuda()
            XB_synth = torch.sigmoid(decoderB(ZB, ZS_infA))  # given XA
            # merged = torch.cat([merged, fixed_XA_3ch], dim=0)
            merged = torch.cat([merged, XB_synth], dim=0)
        merged = torch.cat([merged, WS], dim=0)
        merged = merged[perm, :].cpu()

        # save the results as image
        if train:
            fname = os.path.join(
                self.output_dir_synth,
                'synth_cross_modal_A2B_%s.jpg' % iters
            )
        else:
            fname = os.path.join(
                self.output_dir_synth,
                'eval_synth_cross_modal_A2B_%s.jpg' % iters
            )
        mkdirs(self.output_dir_synth)
        save_image(
            tensor=merged, filename=fname, nrow=(howmany + 2) * int(np.sqrt(n)),
            pad_value=1
        )



        ######## 2) generate xA from given xB (B2A) ########
        merged = torch.cat([fixed_XB], dim=0)
        for k in range(howmany):
            # z_A_stat = np.array(z_A_stat)
            # z_A_stat_mean = np.mean(z_A_stat, 0)
            # ZA = torch.Tensor(z_A_stat_mean)
            # ZA_list = []
            # for _ in range(n):
            #     ZA_list.append(ZA)
            # ZA = torch.stack(ZA_list)



            ZA = torch.randn(n, self.zA_dim)
            z_A_stat = np.array(z_A_stat)
            z_A_stat_mean = np.mean(z_A_stat, 0)
            ZA = ZA + torch.Tensor(z_A_stat_mean)


            if self.use_cuda:
                ZA = ZA.cuda()
            XA_synth = torch.sigmoid(decoderA(ZA, ZS_infB))  # given XB

            XA_synth_3ch = []
            for i in range(len(XA_synth)):
                each_XA = XA_synth[i].clone().squeeze()
                XA_synth_3ch.append(torch.stack([each_XA, each_XA, each_XA]))

            # merged = torch.cat([merged, fixed_XB[:,:,2:30, 2:30]], dim=0)
            merged = torch.cat([merged, torch.stack(XA_synth_3ch)], dim=0)
        merged = torch.cat([merged, WS], dim=0)
        merged = merged[perm, :].cpu()

        # save the results as image
        if train:
            fname = os.path.join(
                self.output_dir_synth,
                'synth_cross_modal_B2A_%s.jpg' % iters
            )
        else:
            fname = os.path.join(
                self.output_dir_synth,
                'eval_synth_cross_modal_B2A_%s.jpg' % iters
            )
        mkdirs(self.output_dir_synth)
        save_image(
            tensor=merged, filename=fname, nrow=(howmany + 2) * int(np.sqrt(n)),
            pad_value=1
        )

        self.set_mode(train=True)

    def get_stat(self, logalpha, logalphaA, logalphaB):
        encoderA = self.encoderA
        encoderB = self.encoderB

        z_A, z_B, z_S =[], [], []
        for _ in range(10000):
            rand_i = np.random.randint(self.N)
            random_XA, random_XB = self.data_loader.dataset.__getitem__(rand_i)[0:2]
            if self.use_cuda:
                random_XA = random_XA.cuda()
                random_XB = random_XB.cuda()
            random_XA = random_XA.unsqueeze(0)
            random_XB = random_XB.unsqueeze(0)

            random_zmuA, random_zstdA, _, \
            muS_infA, stdS_infA, logvarS_infA = encoderA(random_XA)
            random_zmuB, random_zstdB, _, \
            muS_infB, stdS_infB, logvarS_infB = encoderB(random_XB)
            random_zmuS, random_zstdS, _ = apply_poe(
                self.use_cuda, muS_infA, logvarS_infA, muS_infB, logvarS_infB, logalpha, logalphaA, logalphaB
            )
            z_A.append(random_zmuA.cpu().detach().numpy()[0])
            z_B.append(random_zmuB.cpu().detach().numpy()[0])
            z_S.append(random_zmuS.cpu().detach().numpy()[0])
        return z_A, z_B, z_S

    def save_traverse(self, iters, logalpha, logalphaA, logalphaB, z_A, z_B, z_S, loc=-1):

        self.set_mode(train=False)

        encoderA = self.encoderA
        encoderB = self.encoderB
        decoderA = self.decoderA
        decoderB = self.decoderB
        # interpolationA = torch.tensor(np.linspace(np.min(np.array(z_A)) - 0.5, np.max(np.array(z_A)) + 0.5, 30))
        # interpolationB = torch.tensor(np.linspace(np.min(np.array(z_B)) - 0.5, np.max(np.array(z_B)) + 0.5, 30))
        # interpolationS = torch.tensor(np.linspace(np.min(np.array(z_S)) - 0.5, np.max(np.array(z_S)) + 0.5, 30))

        interpolationA = torch.tensor(np.linspace(-6, 6, 30))
        interpolationB = torch.tensor(np.linspace(-6, 6, 30))
        interpolationS = torch.tensor(np.linspace(-6, 6, 30))

        print('------------ traverse interpolation ------------')
        print('interpolationA: ', np.min(np.array(z_A)), np.max(np.array(z_A)))
        print('interpolationB: ', np.min(np.array(z_B)), np.max(np.array(z_B)))
        print('interpolationS: ', np.min(np.array(z_S)), np.max(np.array(z_S)))


        i = np.random.randint(self.N)
        random_XA, random_XB = self.data_loader.dataset.__getitem__(i)[0:2]
        if self.use_cuda:
            random_XA = random_XA.cuda()
            random_XB = random_XB.cuda()
        random_XA = random_XA.unsqueeze(0)
        random_XB = random_XB.unsqueeze(0)
        #
        random_zmuA, _, _, \
        muS_infA, stdS_infA, logvarS_infA = encoderA(random_XA)
        random_zmuB, _, _, \
        muS_infB, stdS_infB, logvarS_infB = encoderB(random_XB)
        random_zmuS, _, _ = apply_poe(
            self.use_cuda, muS_infA, logvarS_infA, muS_infB, logvarS_infB, logalpha, logalphaA, logalphaB
        )

        prn_str = '(latent traversal) random image id = %d' % i
        print(prn_str)
        if self.record_file:
            record = open(self.record_file, 'a')
            record.write('%s\n' % (prn_str,))
            record.close()

        ####

            # fixed_idxs = [10306, 7246, 21440]
            fixed_idxs = [3246, 7001, 14308, 19000, 27447, 33103, 38002, 45232, 51000, 55125]
            # fixed_idxs = [3246, 7000, 14305, 19000, 27444, 33100, 38000, 45231, 51000, 55121]

            fixed_XA = [0] * len(fixed_idxs)
            fixed_XB = [0] * len(fixed_idxs)
            fixed_zmuA = [0] * len(fixed_idxs)
            fixed_zmuB = [0] * len(fixed_idxs)
            fixed_zmuS = [0] * len(fixed_idxs)

            for i, idx in enumerate(fixed_idxs):

                fixed_XA[i], fixed_XB[i] = \
                    self.data_loader.dataset.__getitem__(idx)[0:2]
                if self.use_cuda:
                    fixed_XA[i] = fixed_XA[i].cuda()
                    fixed_XB[i] = fixed_XB[i].cuda()
                fixed_XA[i] = fixed_XA[i].unsqueeze(0)
                fixed_XB[i] = fixed_XB[i].unsqueeze(0)

                fixed_zmuA[i], _, _, \
                muS_infA, stdS_infA, logvarS_infA = encoderA(fixed_XA[i])
                fixed_zmuB[i], _, _, \
                muS_infB, stdS_infB, logvarS_infB = encoderB(fixed_XB[i])
                fixed_zmuS[i], _, _ = apply_poe(
                    self.use_cuda,
                    muS_infA, logvarS_infA, muS_infB, logvarS_infB, logalpha, logalphaA, logalphaB
                )

            IMG = {}

            fixed_XA_3ch = []
            for i in range(len(fixed_XA)):
                each_XA = fixed_XA[i].clone().squeeze()
                fixed_XA_3ch.append(torch.stack([each_XA, each_XA, each_XA]).unsqueeze(0))


            for i, idx in enumerate(fixed_idxs):
                IMG['fixed' + str(i)] = \
                    torch.cat([fixed_XA_3ch[i], fixed_XB[i]], dim=2)
            each_XA = random_XA.clone().squeeze()
            IMG['random'] = torch.cat([torch.stack([each_XA, each_XA, each_XA]).unsqueeze(0), random_XB], dim=2)

            Z = {}
            for i, idx in enumerate(fixed_idxs):
                Z['fixed' + str(i)] = \
                    [fixed_zmuA[i], fixed_zmuB[i], fixed_zmuS[i]]
            Z['random'] = [random_zmuA, random_zmuB, random_zmuS]

        ####

        WS = torch.ones(IMG['fixed1'].shape)
        if self.use_cuda:
            WS = WS.cuda()

        # do traversal and collect generated images
        gifs = []
        for key in Z:

            zA_ori, zB_ori, zS_ori = Z[key]

            ############# traversal over zA #############
            for val in interpolationA:
                gifs.append(WS)
            for row in range(self.zA_dim):
                if loc != -1 and row != loc:
                    continue
                zA = zA_ori.clone()
                for val in interpolationA:
                    zA[:, row] = val
                    sampleA = torch.sigmoid(decoderA(zA, zS_ori)).data
                    sampleB = torch.sigmoid(decoderB(zB_ori, zS_ori)).data

                    sampleA_3ch = []
                    for i in range(sampleA.size(0)):
                        each_XA = sampleA[i].clone().squeeze()
                        sampleA_3ch.append(torch.stack([each_XA, each_XA, each_XA]))
                    sample = torch.cat([torch.stack(sampleA_3ch), sampleB], dim=2)
                    gifs.append(sample)

            ############# traversal over zS #############
            for val in interpolationS:
                gifs.append(WS)
            for row in range(self.zS_dim):
                if loc != -1 and row != loc:
                    continue
                zS = zS_ori.clone()
                for val in interpolationS:
                    zS[:, row] = val
                    sampleA = torch.sigmoid(decoderA(zA_ori, zS)).data
                    sampleB = torch.sigmoid(decoderB(zB_ori, zS)).data

                    sampleA_3ch = []
                    for i in range(sampleA.size(0)):
                        each_XA = sampleA[i].clone().squeeze()
                        sampleA_3ch.append(torch.stack([each_XA, each_XA, each_XA]))
                    sample = torch.cat([torch.stack(sampleA_3ch), sampleB], dim=2)
                    gifs.append(sample)

            # traversal over zB
            for val in interpolationB:
                gifs.append(WS)
            for row in range(self.zB_dim):
                if loc != -1 and row != loc:
                    continue
                zB = zB_ori.clone()
                for val in interpolationB:
                    zB[:, row] = val
                    sampleA = torch.sigmoid(decoderA(zA_ori, zS_ori)).data
                    sampleB = torch.sigmoid(decoderB(zB, zS_ori)).data

                    sampleA_3ch = []
                    for i in range(sampleA.size(0)):
                        each_XA = sampleA[i].clone().squeeze()
                        sampleA_3ch.append(torch.stack([each_XA, each_XA, each_XA]))
                    sample = torch.cat([torch.stack(sampleA_3ch), sampleB], dim=2)
                    gifs.append(sample)

        ####

        # save the generated files, also the animated gifs
        out_dir = os.path.join(self.output_dir_trvsl, str(iters))
        mkdirs(self.output_dir_trvsl)
        mkdirs(out_dir)
        gifs = torch.cat(gifs)
        gifs = gifs.view(
            len(Z), 1 + self.zA_dim + 1 + self.zS_dim + 1 + self.zB_dim,
            len(interpolationA), self.nc, 2 * 32, 32
        ).transpose(1, 2)
        for i, key in enumerate(Z.keys()):
            for j, val in enumerate(interpolationA):
                I = torch.cat([IMG[key], gifs[i][j]], dim=0)
                save_image(
                    tensor=I.cpu(),
                    filename=os.path.join(out_dir, '%s_%03d.jpg' % (key, j)),
                    nrow=1 + 1 + self.zA_dim + 1 + self.zS_dim + 1 + self.zB_dim,
                    pad_value=1)
            # make animated gif
            grid2gif(
                out_dir, key, str(os.path.join(out_dir, key + '.gif')), delay=10
            )

        self.set_mode(train=True)

    def viz_init(self):

        self.viz.close(env=self.name + '/lines', win=self.win_id['recon'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['kl'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['alpha'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['tc'])

        # if self.eval_metrics:
        #     self.viz.close(env=self.name+'/lines', win=self.win_id['metrics'])

    ####
    def visualize_line(self):

        # prepare data to plot
        data = self.line_gather.data
        iters = torch.Tensor(data['iter'])
        recon_both = torch.Tensor(data['recon_both'])
        recon_A = torch.Tensor(data['recon_A'])
        recon_B = torch.Tensor(data['recon_B'])
        kl_both = torch.Tensor(data['kl_both'])
        kl_A = torch.Tensor(data['kl_A'])
        kl_B = torch.Tensor(data['kl_B'])
        kl_A_givenB = torch.Tensor(data['kl_A_givenB'])
        kl_B_givenA = torch.Tensor(data['kl_B_givenA'])

        logalpha = torch.Tensor(data['logalpha'])
        logalphaA = torch.Tensor(data['logalphaA'])
        logalphaB = torch.Tensor(data['logalphaB'])

        tcA = torch.Tensor(data['tcA'])
        tcB = torch.Tensor(data['tcB'])
        tcS = torch.Tensor(data['tcS'])

        recons = torch.stack(
            [recon_both.detach(), recon_A.detach(), recon_B.detach()], -1
        )
        kls = torch.stack(
            [kl_both.detach(), kl_A.detach(), kl_B.detach(), kl_A_givenB.detach(), kl_B_givenA.detach()], -1
        )

        alphas = torch.stack(
            [logalpha.detach(), logalphaA.detach(), logalphaB.detach()], -1
        )

        TCs = torch.stack(
            [tcA.detach(), tcB.detach(), tcS.detach()], -1
        )

        self.viz.line(
            X=iters, Y=recons, env=self.name + '/lines',
            win=self.win_id['recon'], update='append',
            opts=dict(xlabel='iter', ylabel='recon losses',
                      title='Recon Losses', legend=['both', 'A', 'B'])
        )

        self.viz.line(
            X=iters, Y=kls, env=self.name + '/lines',
            win=self.win_id['kl'], update='append',
            opts=dict(xlabel='iter', ylabel='kl losses',
                      title='KL Losses', legend=['both', 'A', 'B', 'kl_A_givenB', 'kl_B_givenA']),
        )

        self.viz.line(
            X=iters, Y=alphas, env=self.name + '/lines',
            win=self.win_id['alpha'], update='append',
            opts=dict(xlabel='iter', ylabel='logalpha',
                      title='Alpha', legend=['logalpha', 'logalphaA', 'logalphaB']),
        )

        self.viz.line(
            X=iters, Y=TCs, env=self.name + '/lines',
            win=self.win_id['tc'], update='append',
            opts=dict(xlabel='iter', ylabel='TC',
                      title='TC', legend=['tcA', 'tcB', 'tcS']),
        )

    ####
    def visualize_line_metrics(self, iters, metric1, metric2):
        
        # prepare data to plot
        iters = torch.tensor([iters], dtype=torch.int64).detach()
        metric1 = torch.tensor([metric1])
        metric2 = torch.tensor([metric2])
        metrics = torch.stack([metric1.detach(), metric2.detach()], -1)
        
        self.viz.line(
          X=iters, Y=metrics, env=self.name+'/lines', 
          win=self.win_id['metrics'], update='append',
          opts=dict( xlabel='iter', ylabel='metrics', 
            title='Disentanglement metrics', 
            legend=['metric1','metric2'] ) 
        )

    def set_mode(self, train=True):

        if train:
            self.encoderA.train()
            self.encoderB.train()
            self.decoderA.train()
            self.decoderB.train()
            self.DA.train()
            self.DB.train()
            self.alpha.train()
        else:
            self.encoderA.eval()
            self.encoderB.eval()
            self.decoderA.eval()
            self.decoderB.eval()
            self.DA.eval()
            self.DB.eval()
            self.alpha.eval()

    ####
    def save_checkpoint(self, iteration):

        encoderA_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_encoderA.pt' % iteration
        )
        encoderB_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_encoderB.pt' % iteration
        )
        decoderA_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_decoderA.pt' % iteration
        )
        decoderB_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_decoderB.pt' % iteration
        )
        alpha_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_alpha.pt' % iteration
        )
        DA_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_DA.pt' % iteration
        )
        DB_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_DB.pt' % iteration
        )

        mkdirs(self.ckpt_dir)

        torch.save(self.encoderA, encoderA_path)
        torch.save(self.encoderB, encoderB_path)
        torch.save(self.decoderA, decoderA_path)
        torch.save(self.decoderB, decoderB_path)
        torch.save(self.DA, DA_path)
        torch.save(self.DB, DB_path)
        torch.save(self.alpha, alpha_path)

    ####
    def load_checkpoint(self):

        encoderA_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_encoderA.pt' % self.ckpt_load_iter
        )
        encoderB_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_encoderB.pt' % self.ckpt_load_iter
        )
        decoderA_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_decoderA.pt' % self.ckpt_load_iter
        )
        decoderB_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_decoderB.pt' % self.ckpt_load_iter
        )
        alpha_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_alpha.pt' % self.ckpt_load_iter
        )
        DA_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_DA.pt' % self.ckpt_load_iter
        )
        DB_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_DB.pt' % self.ckpt_load_iter
        )
        if self.use_cuda:
            self.encoderA = torch.load(encoderA_path)
            self.encoderB = torch.load(encoderB_path)
            self.decoderA = torch.load(decoderA_path)
            self.decoderB = torch.load(decoderB_path)
            self.DA = torch.load(DA_path)
            self.DB = torch.load(DB_path)
            self.alpha = torch.load(alpha_path)
        else:
            self.encoderA = torch.load(encoderA_path, map_location='cpu')
            self.encoderB = torch.load(encoderB_path, map_location='cpu')
            self.decoderA = torch.load(decoderA_path, map_location='cpu')
            self.decoderB = torch.load(decoderB_path, map_location='cpu')
            self.DA = torch.load(DA_path, map_location='cpu')
            self.DB = torch.load(DB_path, map_location='cpu')
            self.alpha = torch.load(alpha_path, map_location='cpu')

