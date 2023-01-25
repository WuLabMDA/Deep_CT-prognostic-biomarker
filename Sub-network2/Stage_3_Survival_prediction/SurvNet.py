'''================== Import packages======================================'''
import numpy as np
import matplotlib.pyplot
import pathlib
import nibabel as nib
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader 
import torchvision.transforms as transforms
from torch.autograd import Variable
from skimage.transform import resize
import scipy.io as sio

'''=============For preprocessing========================================='''
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

'''=====================For network buidling=============================='''
import torch
from torch import nn
import torch.nn.functional as F
import torchtuples as tt

'''============PyCox survival packages===================================='''
from pycox.datasets import metabric
from pycox.models import LogisticHazard
from pycox.models import CoxPH
# from pycox.models import PMF
from pycox.models.loss import NLLLogistiHazardLoss
from pycox.evaluation import EvalSurv

np.random.seed(1234)
_ = torch.manual_seed(123)


'''===========================Dataset==================================='''
root_dir = pathlib.Path.cwd();

input_train = os.path.join(root_dir, "Sample_data","train_features.mat")
input_valid = os.path.join(root_dir, "Sample_data","valid_features.mat")
input_test  = os.path.join(root_dir, "Sample_data","test_features.mat")


# train features
train_dict=sio.loadmat(input_train)
sorted(train_dict.keys())
x_train = train_dict['train_features'];

# valid features
valid_dict=sio.loadmat(input_valid)
sorted(valid_dict.keys())
x_valid = valid_dict['valid_features'];

# test features
test_dict=sio.loadmat(input_test)
sorted(test_dict.keys())
x_test = test_dict['test_features'];
#x_test = test_dict['pos_features'];

'''=================Label Transformations================================'''

# read event data from csv files
df_train = pd.read_csv('train_events.csv')
get_target = lambda df_train: (df_train['OS_time'].values, df_train['OS_status'].values)
y_train = get_target(df_train)

df_valid = pd.read_csv('valid_events.csv')
get_target = lambda df_valid: (df_valid['OS_time'].values, df_valid['OS_status'].values)
y_valid = get_target(df_valid)

df_test = pd.read_csv('test_events.csv')
get_target = lambda df_test: (df_test['OS_time'].values, df_test['OS_status'].values)



train = (x_train, y_train)
valid = (x_valid, y_valid)

# We don't need to transform the test labels
durations_test, events_test = get_target(df_test)
durations_train, events_train = get_target(df_train)
durations_valid, events_valid = get_target(df_valid)

'''=================Feed forward NN============================='''

in_features = x_train.shape[1] # initialize num of features
out_features = 1  


net = torch.nn.Sequential(
     torch.nn.Linear(in_features,256),
     torch.nn.ReLU(),
     torch.nn.BatchNorm1d(256),
     torch.nn.Dropout(0.5),
     
     torch.nn.Linear(256, 128),
     torch.nn.ReLU(),
     torch.nn.BatchNorm1d(128),
     torch.nn.Dropout(0.5),
     
     torch.nn.Linear(128, 64),
     torch.nn.ReLU(),
     torch.nn.BatchNorm1d(64),
     torch.nn.Dropout(0.5),
     

     torch.nn.Linear(64, 32),
     torch.nn.ReLU(),
     torch.nn.BatchNorm1d(32),
     torch.nn.Dropout(0.5),
     
     torch.nn.Linear(32, 16),
     torch.nn.ReLU(),
     torch.nn.BatchNorm1d(16),
     torch.nn.Dropout(0.5),
    
     torch.nn.Linear(16, out_features)
 )



'''==============================Training================================='''
model = CoxPH(net, tt.optim.Adam)
batch_size =32
epochs = 200
#callbacks = [tt.cb.EarlyStopping()]
log = model.fit(x_train, y_train, batch_size, epochs,val_data=valid)

_ = log.plot()
_ = model.compute_baseline_hazards() 

surv_test = model.predict_surv_df(x_test)  
surv_train = model.predict_surv_df(x_train)  
surv_valid = model.predict_surv_df(x_valid)  

'''============================Prediction=================================='''
ev_train = EvalSurv(surv_train, durations_train, events_train, censor_surv='km')
perf_train=ev_train.concordance_td('antolini')
print( "Train CI: ", round(perf_train,2))

ev_valid = EvalSurv(surv_valid, durations_valid, events_valid, censor_surv='km')
perf_valid=ev_valid.concordance_td('antolini')
print( "Valid CI: ", round(perf_valid,2))

ev_test = EvalSurv(surv_test, durations_test, events_test, censor_surv='km')
perf_test=ev_test.concordance_td('antolini')
print( "Test CI: ", round(perf_test,2))


'''============================Risk scores ================================='''
risk_test = model.predict(x_test)
risk_test = risk_test.mean(axis=1)

risk_train = model.predict(x_train)
risk_train = risk_train.mean(axis=1)

risk_valid = model.predict(x_valid)
risk_valid = risk_valid.mean(axis=1)
