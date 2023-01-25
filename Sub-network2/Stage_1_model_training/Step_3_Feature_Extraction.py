
import os, sys
import argparse, shutil
import torch
import numpy as np
import nibabel as nib
from skimage.transform import resize
from sklearn.preprocessing import normalize
from fn_unet_noskip import UNet
import pathlib

import matplotlib.pyplot as plt
import scipy.io as sio

from fn_feature_extraction import FeatureExtraction


def set_args():
    parser = argparse.ArgumentParser(description="3D U-Net Reconstruction")
    parser.add_argument('--cuda_id',               type=str, default="1")
    parser.add_argument('--session',               type=str, default="recon_1")
    parser.add_argument('--best_model_name',       type=str, default="lung_recon_1.pt")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = set_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id  
    # load model
  
    '''--Load your trained model here -------'''

    model = UNet(in_channels=1,
             out_channels=1,
             n_blocks=4,
             start_filters=32,
             activation='relu',
             normalization='batch',
             conv_mode='same',
             dim=3).cuda()

    
    root = pathlib.Path.cwd()
    model_path = os.path.join(root, "Best_Models", args.session, args.best_model_name) # Best_Models is a folder created to store the .pt files
    model_weights = torch.load(model_path)
    model.load_state_dict(model_weights)
    
    '''-- Feature extraction starts here -------'''
    ''' Train_3D, Valid_3D, and Test_3D are folders with images for feature extraction in 3D format'''
    
    train_dir = os.path.join(root, "Train_3D")
    valid_dir = os.path.join(root, "Valid_3D")
    test_dir = os.path.join(root, "Test_3D") 
    
    print("----Processing Training Cohort----")
    features = FeatureExtraction(model_path,train_dir)
    train_features = features.data
    train_features=np.array(train_features)

    
    print("----Processing Validation Cohort----")
    features = FeatureExtraction(model_path,valid_dir)
    valid_features = features.data
    valid_features=np.array(valid_features)

     
    print("----Processing Testing Cohort----")
    features = FeatureExtraction(model_path,test_dir)
    test_features = features.data
    test_features=np.array(test_features)

    # save outputs into matlab files
    sio.savemat('lung_train.mat', {'train_features':train_features})
    sio.savemat('lung_valid.mat', {'valid_features':valid_features})
    sio.savemat('lung_test.mat', {'test_features':test_features})
  
    
    
    

    