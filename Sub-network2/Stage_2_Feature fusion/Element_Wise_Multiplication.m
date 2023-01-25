 clear
 close all
 clc
%% read data

load ([pwd '\Sample_data\lung_train']);
load ([pwd '\Sample_data\media_train']);
load ([pwd '\Sample_data\raw_train']);

load ([pwd '\Sample_data\lung_valid']);
load ([pwd '\Sample_data\media_valid']);
load ([pwd '\Sample_data\raw_valid']);

load ([pwd '\Sample_data\lung_test']);
load ([pwd '\Sample_data\media_test']);
load ([pwd '\Sample_data\raw_test']);

EWM_train = lung_train.*media_train.*raw_train;
EWM_valid = lung_valid.*media_valid.*raw_valid;
EWM_test = lung_test.*media_test.*raw_test;

clear lung_test lung_train lung_valid
clear media_test media_train media_valid
clear raw_test raw_train raw_valid

save EWM_train EWM_train
save EWM_valid EWM_valid
save EWM_test EWM_test

