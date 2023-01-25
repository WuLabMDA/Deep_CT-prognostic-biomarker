
import os, sys
import argparse, shutil
import torch
import numpy as np
import nibabel as nib
from skimage.transform import resize
from fn_unet_noskip import UNet
import pathlib

def set_args():
    parser = argparse.ArgumentParser(description="3D U-Net Reconstruction")
    parser.add_argument('--cuda_id',               type=str, default="1") 
    parser.add_argument('--session',               type=str, default="recon_1") # 
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
    model_path = os.path.join(root, "Best_Models",args.session, args.best_model_name) # Best_Models is a folder created to store the .pt files
    model_weights = torch.load(model_path)
    model.load_state_dict(model_weights)
    
    
    
 '''--Reconstruction starts here -------'''
    
    recon_dir = os.path.join(root, "Reconstructed_Data") # Reconstructed_Data is a folder created to store recon results
    if os.path.exists(recon_dir):
        shutil.rmtree(recon_dir)
    os.makedirs(recon_dir)
    
    
    # traverse images
    input_dir = os.path.join(root, "Lung_3D", "Test", "Input")
    image_names = sorted([ele for ele in os.listdir(input_dir) if ele.endswith(".nii.gz")])
    image_paths = [os.path.join(input_dir, ele) for ele in image_names]
    for ind, cur_img_path in enumerate(image_paths):
        file_name = os.path.basename(cur_img_path).split('.', 1)[0]
        print("Recon {} {:3d}/{:3d}".format(file_name, ind+1, len(image_paths)))
        image = nib.load(cur_img_path).get_fdata().astype(np.float32)
        image = resize(image, output_shape=(128, 128, 128))
        # save the resized image
        resize_nifty = nib.Nifti1Image(image, affine=np.eye(4))
        nib.save(resize_nifty, os.path.join(recon_dir, file_name + "_original.nii.gz"))
        # model recontruction
        image = np.expand_dims(image, axis=(0, 1))
        image = torch.from_numpy(image).type(torch.float32).cuda()
        recon = model(image)
        recon = np.squeeze(recon.detach().cpu().numpy())
        recon_nifty = nib.Nifti1Image(recon, affine=np.eye(4))
        nib.save(recon_nifty, os.path.join(recon_dir, file_name + "_recon.nii.gz"))
        
        
