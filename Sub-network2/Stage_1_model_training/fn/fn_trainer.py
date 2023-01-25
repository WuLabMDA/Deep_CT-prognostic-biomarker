import os
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
import pathlib

root = pathlib.Path.cwd()


class Trainer:
    def __init__(self, model: torch.nn.Module, criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_DataLoader: torch.utils.data.Dataset,
                 validation_DataLoader: torch.utils.data.Dataset = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 epochs: int = 100):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.epochs = epochs
        self.min_val_loss = 1.0

        
        

        self.training_loss = []
        self.validation_loss = []  
        self.learning_rate = []


    def run_trainer(self, args):
        # tensorboard
        log_dir = os.path.join(root, "TensorBoard", "lung_recon_noskip", args.session)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)
        # training
        for ind in range(self.epochs):
            """Training block"""
            self._train()
            """Validation block"""
            if self.validation_DataLoader is not None:
                self._validate(args)
            writer.add_scalar("learning rate", self.learning_rate[ind], ind)
            writer.add_scalar("training loss", self.training_loss[ind], ind)
            #writer.add_scalar("training combined loss",self.training_mixed_loss[ind], ind)
            writer.add_scalar("validation loss", self.validation_loss[ind], ind)
           # writer.add_scalar("validation combined loss",self.validation_mixed_loss[ind], ind)
            print("Time: {} Epoch {:3d} lr:{:.4f} training loss: {:.3f} validation loss: {:.3f}".format(
                datetime.now().strftime("%H:%M:%S"), ind, self.learning_rate[ind], self.training_loss[ind], self.validation_loss[ind]))
            # Learning rate schedule
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()  # learning rate scheduler step
          

    def _train(self):
        self.model.train()  # train mode
        self.model.cuda()
        train_losses = []  # accumulate the losses here
        for i, (input, target) in enumerate(self.training_DataLoader):
            input, target = input.cuda(), target.cuda()
            self.optimizer.zero_grad()  # zerograd the parameters
            out = self.model(input)  # one forward pass
            loss = self.criterion(out, target)  # calculate loss
            loss_value = loss.item()
            train_losses.append(loss_value)
            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters
        self.training_loss.append(np.mean(train_losses))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])


    def _validate(self, args):
        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses here
        for i, (input, target) in enumerate(self.validation_DataLoader):
            input, target = input.cuda(), target.cuda()
            with torch.no_grad():
                out = self.model(input)
                loss = self.criterion(out, target)
                loss_value = loss.item()
                valid_losses.append(loss_value)
        mean_val_loss = np.mean(valid_losses)
        self.validation_loss.append(mean_val_loss)
        
        # save the current best model
        if mean_val_loss <= self.min_val_loss:
            self.min_val_loss = mean_val_loss # if you want to save only current best model
            model_dir = os.path.join(root, "Best_Models", "lung_recon_noskip", args.session)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            model_path = os.path.join(model_dir, "lung_recon_noskip_{:.3f}.pt".format(mean_val_loss))
            torch.save(self.model.state_dict(), model_path)
            

                    
                    
                    
                
                
