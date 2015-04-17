function [bags, num_data, num_feature, trainindex, testindex] = MIL_Data_Load_MED2012(event_idx)

%% param event_idx: 1-25
global preprocess;

filename='/net/per610a/export/das11f/plsang/codes/opensource/kuantinglai_cvpr2014/InstanceVideoDetect_v1.0/med12MBH_BOW_20s.mat';

load(filename);
       
num_data = size(featNum, 2);

for ii=1:num_data,
    bags(ii).name = fileList{ii};
    bags(ii).inst_name = arrayfun(@(x) sprintf('HVC1000-%d', x), [1:featNum(ii)], 'UniformOutput', false);
    bags(ii).label = Label(ii, event_idx);
    bags(ii).inst_label = bags(ii).label * ones(1, featNum(ii));
    bags(ii).instance = featMat{ii};
end

num_feature = size(bags(1).instance, 2);

testindex = TstInd';
trainindex = TrnInd';

% normalize the data set
if (preprocess.Normalization == 1) 
    bags = MIL_Scale(bags);
end;

% randomize the data
rand('state',sum(100*clock));
if (preprocess.Shuffled == 1) %Shuffle the datasets
    Vec_rand = rand(num_data, 1);
    [B, Index] = sort(Vec_rand);
    bags = bags(Index);
end;

clear featMat;
