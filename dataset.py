import os, random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

###############################################################################
    
class CustomImageFolder(ImageFolder):
    
    '''
    Dataset when it is inefficient to load all data items to memory
    '''
    
    ####
    def __init__(self, root, transform=None):
        
        '''
        root = directory that contains data (images)
        '''
        
        super(CustomImageFolder, self).__init__(root, transform)


    ####
    def __getitem__(self, i):

        img = self.loader(self.imgs[i][0])
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, i


###############################################################################
        
class CustomTensorDataset(Dataset):
    
    '''
    Dataset when it is possible and efficient to load all data items to memory
    '''
    
    ####
    def __init__(self, data_tensor, transform=None):
        
        '''
        data_tensor = actual data items; (N x C x H x W)
        '''
        
        self.data_tensor = data_tensor
        self.transform = transform

    ####
    def __getitem__(self, i):
        
        img = self.data_tensor[i]
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, i

    ####
    def __len__(self):
        
        return self.data_tensor.size(0)
    

###############################################################################
        
class IDAZ_ELLI_PairedTensorDataset(Dataset):
    
    '''
    ID-Azimuth-Elevation-Lighting Paired Dataset 
      (possible and efficient to load all data items to memory)    
    '''
    
    ####
    def __init__(self, data_tensor, factors_info, transform=None):
        
        '''
        data_tensor = actual data items; (N x C x H x W)
        factors_info = latent_classes; (N x 4) -- (ID, AZ, EL, LI)
        '''
        
        self.data_tensor = data_tensor
        self.factors_info = factors_info
        self.transform = transform

    ####
    def __getitem__(self, i):
        
        i1, i2 = self._get_mode_specific_indices(i)
 
        img1 = self.data_tensor[i1]
        img2 = self.data_tensor[i2]
        
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        return img1, img2, i1, i2, i

    ####
    def __len__(self):
        
        return self.data_tensor.size(0)
    
    ####
    def _get_mode_specific_indices(self, ii):
        
        '''
        ii could contain multiple indices
        '''
        
        if not isinstance(ii, np.ndarray):
            
            i = ii
        
            cf = self.factors_info[i]  # current factor values (classes)
            
            # for modality 1
            i1 = int( np.where(  # elevation = neutral (5) fixed
                (self.factors_info[:,0]==cf[0]) * \
                (self.factors_info[:,1]==cf[1]) * \
                (self.factors_info[:,2]==cf[2]) * \
                (self.factors_info[:,3]==5) \
            )[0] )
            
            # for modality 2
            i2 = int( np.where(  # lighting = neutral (5) fixed
                (self.factors_info[:,0]==cf[0]) * \
                (self.factors_info[:,1]==cf[1]) * \
                (self.factors_info[:,2]==5) * \
                (self.factors_info[:,3]==cf[3]) \
            )[0] )
            
            return i1, i2
        
        else:
            
            i1 = np.zeros(ii.shape)
            i2 = np.zeros(ii.shape)
            
            for i0, i in enumerate(ii): 
    
                cf = self.factors_info[i]  # current factor values (classes)
            
                # for modality 1
                i1[i0] = int( np.where(  # elevation = neutral (5) fixed
                    (self.factors_info[:,0]==cf[0]) * \
                    (self.factors_info[:,1]==cf[1]) * \
                    (self.factors_info[:,2]==cf[2]) * \
                    (self.factors_info[:,3]==5) \
                )[0] )
                
                # for modality 2
                i2[i0] = int( np.where(  # lighting = neutral (5) fixed
                    (self.factors_info[:,0]==cf[0]) * \
                    (self.factors_info[:,1]==cf[1]) * \
                    (self.factors_info[:,2]==5) * \
                    (self.factors_info[:,3]==cf[3]) \
                )[0] )
                
            return i1, i2


###############################################################################
        
def create_dataloader(args, williams=None):
    
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    assert image_size == 64, 'currently only image size of 64 is supported'

    transform = transforms.Compose([
      transforms.Resize((image_size, image_size)),
      transforms.ToTensor(),])

    drop_last = True
    
    if name.lower() == 'celeba':
        
        root = os.path.join(dset_dir, 'celeba') #/img_align_crop_64x64')
        train_kwargs = {'root': root, 'transform': transform}
        dset = CustomImageFolder
    
    elif name.lower() == '3dchairs':
    
        root = os.path.join(dset_dir, '3DChairs')
        train_kwargs = {'root': root, 'transform': transform}
        dset = CustomImageFolder
    
    elif name.lower() == 'dsprites':
    
        root = os.path.join( dset_dir, 'dsprites-dataset/imgs.npy' )
        imgs = np.load(root, encoding='latin1')
        data = torch.from_numpy(imgs).unsqueeze(1).float()
        train_kwargs = {'data_tensor': data}
        dset = CustomTensorDataset
    
    elif name.lower() == 'oval_dsprites':
    
        latent_classes = np.load( os.path.join( dset_dir, 
           'dsprites-dataset/latents_classes.npy'), encoding='latin1' )
        idx = np.where(latent_classes[:,1]==1)[0]  # "oval" shape only
        root = os.path.join( dset_dir, 'dsprites-dataset/imgs.npy' )
        imgs = np.load(root, encoding='latin1')
        imgs = imgs[idx]
        data = torch.from_numpy(imgs).unsqueeze(1).float()
        train_kwargs = {'data_tensor': data}
        dset = CustomTensorDataset
        
    elif name.lower() == 'ov_scro_spr':
    
        latent_classes = np.load( os.path.join( dset_dir, 
           'dsprites-dataset/latents_classes.npy'), encoding='latin1' )
        idx = np.where(latent_classes[:,1]==1)[0]  # "oval" shape only
        root = os.path.join( dset_dir, 'dsprites-dataset/imgs.npy' )
        imgs = np.load(root, encoding='latin1')
        imgs = imgs[idx]
        latent_classes = latent_classes[idx,:]
        latent_classes = latent_classes[:,[2,3,4,5]]
        data = torch.from_numpy(imgs).unsqueeze(1).float()
        train_kwargs = {'data_tensor': data, 'factors_info': latent_classes}
        dset = OV_SCRO_SPR_PairedTensorDataset
        
    elif name.lower() == 'idaz_elli_3df':
    
        latent_classes, latent_values = np.load( os.path.join( 
            dset_dir, '3d_faces/rtqichen/gt_factor_labels.npy' ) )
        root = os.path.join( dset_dir, 
                            '3d_faces/rtqichen/basel_face_renders.pth' )
        data = torch.load(root).float().div(255)  # (50x21x11x11x64x64)
        data = data.view(-1,64,64).unsqueeze(1)  # (127050 x 1 x 64 x 64)
        train_kwargs = {'data_tensor': data, 'factors_info': latent_classes}
        dset = IDAZ_ELLI_PairedTensorDataset

        
    elif name.lower() == '3dfaces':
    
        root = os.path.join( dset_dir, 
                            '3d_faces/rtqichen/basel_face_renders.pth' )
        data = torch.load(root).float().div(255)  # (50x21x11x11x64x64)
        data = data.view(-1,64,64).unsqueeze(1)  # (127050 x 1 x 64 x 64)
        train_kwargs = {'data_tensor': data}
        dset = CustomTensorDataset

    elif name.lower() == 'sinusoid':
        dset = Sinusoid(n_pts, partition='train')
    else:
        raise NotImplementedError

    dataset = dset(**train_kwargs)
    
    dataloader = DataLoader( dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True, drop_last=drop_last )

    return dataloader
