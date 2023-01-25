## Main

import torch
import argparse
import os
from torch.optim import lr_scheduler
from fn.fn_unet import UNet
from fn.fn_trainer import Trainer
from fn.fn_dataloader import dataloader_training, dataloader_validation
import numpy as np



np.random.seed(1234)
_ = torch.manual_seed(123)



'''--set your parameters such learning rate, number of epoch, etc -------'''

def set_args():
    parser = argparse.ArgumentParser(description="3D U-Net Reconstruction")
    parser.add_argument('--cuda_id',               type=str, default="1") # Change GPU id here
    parser.add_argument('--session',               type=str, default="recon_1") # Change to any session name here
    parser.add_argument('--lr',                    type=float, default=0.001) # Learning rate set by default at 0.001
    parser.add_argument('--lr_decay_epoch',        type=int, default=10)  # optional
    parser.add_argument('--lr_decay_ratio',        type=int, default=0.8) # optional
    parser.add_argument('--num_epoch',             type=int, default=200) # epoch by default set at 200
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    args = set_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id 
    
    '''--UNET structure -------'''
    # model
    model = UNet(in_channels=1, out_channels=1, n_blocks=4, start_filters=32, activation="relu",
                 normalization="batch", conv_mode="same", dim=3).cuda()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_epoch, gamma=args.lr_decay_ratio)
    print("----Start Training----")
    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer,
                      training_DataLoader=dataloader_training,
                      validation_DataLoader=dataloader_validation,
                      lr_scheduler=scheduler, epochs=args.num_epoch)
    trainer.run_trainer(args)




