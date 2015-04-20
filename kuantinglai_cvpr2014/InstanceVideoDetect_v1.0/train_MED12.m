function train_MED12(run_name, C1Params, C2Params, max_neg)

addpath(genpath('pSVM-master'));
addpath('liblinear-1.95/matlab');

% Load 25 event videos from MED12
load('med12_GT_info.mat');      % Ground Truth matrix: fileList, Label, TrnInd, TstInd
load('med12MBH_BOW_20s.mat');	% featNum, featMat

% Prepare training vectors
TrainVec = cell2mat(featMat(TrnInd)');
TrnFeatNum = featNum(TrnInd);
VidLabel = Label(TrnInd, :);
OUT_NAME = sprintf('models/med12_psvm_mbh_20s_%s_mneg%d', run_name, max_neg);

%C1 = [1];             % Cost parameter for liblinear
%C2 = [0.01, 1, 100];
%C2 = [1];
Proportion = 1;

psvm_train(VidLabel, TrnFeatNum, TrainVec, Proportion, C1Params, C2Params, OUT_NAME, max_neg, 'liblinear');

clear TrainVec;