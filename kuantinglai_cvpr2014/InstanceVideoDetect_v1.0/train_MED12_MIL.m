clear; clc

addpath(genpath('pSVM-master'));
addpath('liblinear-1.95/matlab');

% Load 25 event videos from MED12
load('med12_GT_info.mat');      % Ground Truth matrix: fileList, Label, TrnInd, TstInd
load('med12MBH_BOW_20s.mat');	% featNum, featMat

% Prepare training vectors
TrainVec = cell2mat(featMat(TrnInd)');
TrnFeatNum = featNum(TrnInd);
VidLabel = Label(TrnInd, :);
OUT_NAME = 'models/med12_psvm_mbh_20s';

C1 = [1];             % Cost parameter for liblinear
C2 = [0.01, 1, 100];
Proportion = 1;

psvm_train_MIL(VidLabel, TrnFeatNum, TrainVec, Proportion, C1, C2, OUT_NAME);
