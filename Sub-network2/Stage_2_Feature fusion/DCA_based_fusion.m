 clear
 close all
 clc
 
%% Training data
load ([pwd '\Sample_data\lung_train']);
load ([pwd '\Sample_data\media_train']);
load ([pwd '\Sample_data\raw_train']);
load ([pwd '\Sample_labels\train_labels']);

%-LM features---
X=lung_train';
Y=media_train';
[Ax, Ay, Xdca, Ydca] = dcaFuse(X, Y, train_labels);
LM_features = [Xdca ; Ydca]';

%--LR features---
X=lung_train';
Y=raw_train';
[Ax, Ay, Xdca, Ydca] = dcaFuse(X, Y, train_labels);
LR_features = [Xdca ; Ydca]';

%--MR features---
X=media_train';
Y=raw_train';
[Ax, Ay, Xdca, Ydca] = dcaFuse(X, Y, train_labels);
MR_features = [Xdca ; Ydca]';

%--final features--
DCA_train = [LM_features,LR_features,MR_features];

clear Ax Ay lung_train media_train raw_train
clear X Xdca Y Ydca train_labels
clear LM_features LR_features MR_features


%% Validation data
load ([pwd '\Sample_data\lung_valid']);
load ([pwd '\Sample_data\media_valid']);
load ([pwd '\Sample_data\raw_valid']);
load ([pwd '\Sample_labels\valid_labels']);

%-LM features---
X=lung_valid';
Y=media_valid';
[Ax, Ay, Xdca, Ydca] = dcaFuse(X, Y, valid_labels);
LM_features = [Xdca ; Ydca]';

%--LR features---
X=lung_valid';
Y=raw_valid';
[Ax, Ay, Xdca, Ydca] = dcaFuse(X, Y, valid_labels);
LR_features = [Xdca ; Ydca]';

%--MR features---
X=media_valid';
Y=raw_valid';
[Ax, Ay, Xdca, Ydca] = dcaFuse(X, Y, valid_labels);
MR_features = [Xdca ; Ydca]';

%--final features--
DCA_valid = [LM_features,LR_features,MR_features];

clear Ax Ay lung_valid media_valid raw_valid
clear X Xdca Y Ydca valid_labels
clear LM_features LR_features MR_features



%% Testing data
load ([pwd '\Sample_data\lung_test']);
load ([pwd '\Sample_data\media_test']);
load ([pwd '\Sample_data\raw_test']);
load ([pwd '\Sample_labels\test_labels']);

%-LM features---
X=lung_test';
Y=media_test';
[Ax, Ay, Xdca, Ydca] = dcaFuse(X, Y, test_labels);
LM_features = [Xdca ; Ydca]';

%--LR features---
X=lung_test';
Y=raw_test';
[Ax, Ay, Xdca, Ydca] = dcaFuse(X, Y, test_labels);
LR_features = [Xdca ; Ydca]';

%--MR features---
X=media_test';
Y=raw_test';
[Ax, Ay, Xdca, Ydca] = dcaFuse(X, Y, test_labels);
MR_features = [Xdca ; Ydca]';

%--final features--
DCA_test = [LM_features,LR_features,MR_features];

clear Ax Ay lung_test media_test raw_test
clear X Xdca Y Ydca test_labels
clear LM_features LR_features MR_features


%% Save
save DCA_train DCA_train
save DCA_valid DCA_valid
save DCA_test DCA_test
