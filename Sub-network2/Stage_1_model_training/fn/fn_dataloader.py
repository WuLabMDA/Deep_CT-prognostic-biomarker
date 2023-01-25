
import pathlib
import torch
from skimage.io import imread
from torch.utils import data
from tqdm.notebook import tqdm

import albumentations
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from skimage.transform import resize

from fn.fn_niftiReader import SegmentationDataSet2
from fn.fn_transformations import ComposeDouble, AlbuSeg2d, FunctionWrapperDouble, normalize_01, create_dense_target


# data directory
root = pathlib.Path.cwd() / 'Lung_3D' # Lung_3D is the folder that should contains your 3D volumetric data



def get_filenames_of_path(path: pathlib.Path, ext: str = '*'):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in path.glob(ext) if file.is_file()]
    return filenames

# Training input and target files
inputs_train = get_filenames_of_path(root / 'Train/Input')
targets_train = get_filenames_of_path(root / 'Train/Target')

# Validation input and target files
inputs_valid = get_filenames_of_path(root / 'Valid/Input')
targets_valid = get_filenames_of_path(root / 'Valid/Target')


#pre-transformation (resize input image to 128 128)

pre_transforms = ComposeDouble([
    FunctionWrapperDouble(resize,
                          input=True,
                          target=False,
                          output_shape=(128,128,128,1)),
                          #output_shape=(128, 128)),
    FunctionWrapperDouble(resize,
                          input=False,
                          target=True,
                          # output_shape=(1,128, 128),
                          output_shape=(128,128,128,1),
                          #order=0,
                          #anti_aliasing=False,
                          #preserve_range=True
                          ),
])

# training transformations and augmentations
transforms_training = ComposeDouble([
    AlbuSeg2d(albumentations.HorizontalFlip(p=0.5)),
    #FunctionWrapperDouble(create_dense_target, input=False, target=True), # need this for segmentation netwok
    FunctionWrapperDouble(np.moveaxis, input=True, target=True, source=-1, destination=0),
    #FunctionWrapperDouble(normalize_01)
])


# validation transformations
transforms_validation = ComposeDouble([
    #FunctionWrapperDouble(create_dense_target, input=False, target=True),
    FunctionWrapperDouble(np.moveaxis, input=True, target=True, source=-1, destination=0),
    #FunctionWrapperDouble(normalize_01)
])




# dataset training
dataset_train = SegmentationDataSet2(inputs=inputs_train,
                                    targets=targets_train,
                                    transform=transforms_training,
                                    use_cache=True,
                                    pre_transform=pre_transforms)

# dataset validation
dataset_valid = SegmentationDataSet2(inputs=inputs_valid,
                                    targets=targets_valid,
                                    transform=transforms_validation,
                                    use_cache=True,
                                    pre_transform=pre_transforms)

# dataloader training
dataloader_training = DataLoader(dataset=dataset_train,
                                 batch_size=6,
                                 shuffle=True)

# dataloader validation
dataloader_validation = DataLoader(dataset=dataset_valid,
                                   batch_size=6,
                                   shuffle=True)

