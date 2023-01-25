import torch
import torch.nn as nn
from fn_unet_noskip import UNet
import pathlib
import os
import nibabel as nib
import numpy as np
from skimage.transform import resize
import torchvision.transforms as transforms
from torch.autograd import Variable



class FeatureExtraction:
    """Load the trained model and extract features."""
    def __init__(self,model_path,data_dir):
        self.data = []
        model = UNet(in_channels=1, out_channels=1, n_blocks=4, start_filters=32, activation="relu",
                 normalization="batch", conv_mode="same", dim=3).cuda()
        
        model_weights = torch.load(model_path)

        model.load_state_dict(model_weights)
        layer = model._modules.get('down_blocks')[0].norm2
        model.cuda().eval()
        #model.cpu().eval()  ##-------------------changed this later in DGX!!!!!!!!!
        
        # input paths
        images_names = sorted([ele for ele in os.listdir(data_dir) if ele.endswith(".nii.gz")])
        image_paths = [os.path.join(data_dir, ele) for ele in images_names]
        images = [nib.load(os.path.join(data_dir,img_name)) for img_name in images_names]
        
        # dtype 
        for i in range(len(images_names)):
            image = nib.load((os.path.join(data_dir,images_names[i])))
            new_data = np.copy(image.get_fdata())
            #hd = image_header
            new_dtype = np.float32
            new_data = new_data.astype(new_dtype)
            image.set_data_dtype(new_dtype)
            images[i]=new_data
        
        # resize images as per training and validation
        images_res = [resize(img, (128, 128,128)) for img in images]
         
        
    #-----------------------------------------------------------------
        def get_vector(image_name):
         img = image_name
         #to_tensor = transforms.ToTensor()
         t_img = torch.from_numpy(img)
         t_img = t_img.unsqueeze(0).unsqueeze(0)
         t_img = t_img.cuda()
         
         my_embedding = torch.zeros(1,32,128,128,128)
         #my_embedding = torch.zeros(1,64,64,64,64)
         #my_embedding = torch.zeros(1,128,32,32,32)
         #my_embedding = torch.zeros(1,256,16,16,16)
         
         def copy_data(m, i, o):
             my_embedding.copy_(o.data)
             
         h = layer.register_forward_hook(copy_data)
         model(t_img)
         h.remove()
         return my_embedding
     #---------------------------------------------       
        List =[]
        for i in range (len(images_res)):
            file_name=(os.path.basename(images_names[i]).split('.', 1)[0])
            print("Extracting features for {} {:3d}/{:3d}".format(file_name, i+1, len(images_res)))  ######
            vector = get_vector(images_res[i])
            #vector = torch.flatten(vector)
            vector = vector.numpy()
            vector = np.mean(vector, axis=(0,2,3,4)) # mean pooling on the last 3 channels
            vector = np.squeeze(vector)  # remove batch dim and channel dim -> [H, W]
            self.data.append(vector)
            #List.append(vector)
            #features=np.array(List)


       

            

        
        

     
  
     
    

        
        
    