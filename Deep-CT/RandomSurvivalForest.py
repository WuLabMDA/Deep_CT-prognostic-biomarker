import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pathlib
import os, sys
import scipy.io as sio
import warnings
warnings.simplefilter(action='ignore')
from sksurv.ensemble import RandomSurvivalForest
from sklearn.preprocessing import StandardScaler


root_dir = pathlib.Path.cwd()
'''-----Training data-----'''
train_path = os.path.join(root_dir,'sample_data',"train_data.csv")
df_train = pd.read_csv(train_path)

x_train = df_train.loc[:,'M1':'M4']
x_train = x_train.to_numpy('float32')

## convert y labels to structured array
os_status = df_train['OS_status'].tolist()
os_time = df_train['OS_time'].tolist()
y_train = np.zeros(len(df_train), dtype={'names':('os_status', 'os_time'),
                          'formats':('bool', 'f8')})
y_train['os_status'] = os_status
y_train['os_time'] = os_time

del os_status,os_time,df_train

'''-----validation data-----'''
valid_path = os.path.join(root_dir,'sample_data',"valid_data.csv")
df_valid = pd.read_csv(valid_path)

x_valid = df_valid.loc[:,'M1':'M4']
x_valid = x_valid.to_numpy('float32')

## convert y labels to structured array
os_status = df_valid['OS_status'].tolist()
os_time = df_valid['OS_time'].tolist()
y_valid = np.zeros(len(df_valid), dtype={'names':('os_status', 'os_time'),
                          'formats':('bool', 'f8')})
y_valid['os_status'] = os_status
y_valid['os_time'] = os_time

del os_status,os_time,df_valid

'''-----testing data-----'''
test_path = os.path.join(root_dir,'sample_data',"test_data.csv")
df_test = pd.read_csv(test_path)

x_test = df_test.loc[:,'M1':'M4']
x_test = x_test.to_numpy('float32')

## convert y labels to structured array
os_status = df_test['OS_status'].tolist()
os_time = df_test['OS_time'].tolist()
y_test = np.zeros(len(df_test), dtype={'names':('os_status', 'os_time'),
                          'formats':('bool', 'f8')})
y_test['os_status'] = os_status
y_test['os_time'] = os_time

del os_status,os_time,df_test


'''--fit RSF---'''
random_state = 20
rsf = RandomSurvivalForest(n_estimators=1000,
                           min_samples_split=10,
                           min_samples_leaf=15,
                           max_features="sqrt",
                           n_jobs=-1,
                           random_state=random_state)
rsf.fit(x_train, y_train)

train_ci = rsf.score(x_train, y_train)
valid_ci = rsf.score(x_valid, y_valid)
test_ci = rsf.score(x_test, y_test)

print( "Train CI: ", round(train_ci,2))
print( "Valid CI: ", round(valid_ci,2))
print( "Test CI: ", round(test_ci,2))




risk_train = rsf.predict(x_train)
risk_valid = rsf.predict(x_valid)
risk_test = rsf.predict(x_test)



