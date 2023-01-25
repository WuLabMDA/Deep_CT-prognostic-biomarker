import torch
import numpy as np
import nibabel as nib
from skimage.io import imread
from torch.utils import data
from tqdm.notebook import tqdm




class SegmentationDataSet2(data.Dataset):
    """Image segmentation dataset with caching and pretransforms."""
    def __init__(self,
                 inputs: list,
                 targets: list,
                 transform=None,
                 use_cache=False,
                 pre_transform=None,
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        #self.targets_dtype = torch.long
        self.targets_dtype = torch.float32
        self.use_cache = use_cache
        self.pre_transform = pre_transform

        if self.use_cache:
            self.cached_data = []

            progressbar = tqdm(range(len(self.inputs)), desc='Caching')
            for i, img_name, tar_name in zip(progressbar, self.inputs, self.targets):
                #img, tar = imread(str(img_name)), imread(str(tar_name))
                img, tar = nib.load(img_name), nib.load(tar_name)
                # -----------newly added--------------------
                new_data = np.copy(img.get_fdata())
                hd =img.header
                #new_dtype = np.double
                new_dtype = np.float32
                new_data = new_data.astype(new_dtype)
                img.set_data_dtype(new_dtype)
                img=new_data
                
                new_tar = np.copy(tar.get_fdata())
                hd =tar.header
                new_dtype = np.float32
               # new_dtype = np.long
                new_tar = new_tar.astype(new_dtype)
                tar.set_data_dtype(new_dtype)
                tar=new_tar
                # ------------------------------------
                
                if self.pre_transform is not None:
                    img, tar = self.pre_transform(img, tar)

                self.cached_data.append((img, tar))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        if self.use_cache:
            x, y = self.cached_data[index]
        else:
            # Select the sample
            input_ID = self.inputs[index]
            target_ID = self.targets[index]

            # Load input and target
            #x, y = imread(str(input_ID)), imread(str(target_ID))
            x, y = nib.load(input_ID), (target_ID)

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)

        return x, y












