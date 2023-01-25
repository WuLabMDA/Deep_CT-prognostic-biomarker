 clear
 close all
 clc
%% load DCA features
load ([pwd '\DCA_train.mat']);
load ([pwd '\DCA_valid.mat']);
load ([pwd '\DCA_test.mat']);

%% load EWM features
load ([pwd '\EWM_train.mat']);
load ([pwd '\EWM_valid.mat']);
load ([pwd '\EWM_test.mat']);

%% Joint two types of feature fusion by concatenate them

train_features = [EWM_train, DCA_train];
valid_features = [EWM_valid, DCA_valid];
test_features = [EWM_test, DCA_test];

%% Save
save train_features train_features
save valid_features valid_features
save test_features test_features
