function [bags, num_data, num_feature, trainindex, testindex] = MIL_Data_Load_MED2012(event_idx)

%% param event_idx: 1-25
global preprocess;

filename='/net/per610a/export/das11f/plsang/codes/opensource/kuantinglai_cvpr2014/InstanceVideoDetect_v1.0/med12MBH_BOW_20s.mat';

load(filename);
       
num_data = size(featNum, 2);

max_negative = 30;

for ii=1:num_data,
    bags(ii).name = fileList{ii};
    bags(ii).inst_name = arrayfun(@(x) sprintf('HVC1000-%d', x), [1:featNum(ii)], 'UniformOutput', false);
    bags(ii).label = Label(ii, event_idx);
    bags(ii).inst_label = bags(ii).label * ones(1, featNum(ii));
    bags(ii).instance = featMat{ii};
end

sub_train_idx = find(Label(TrnInd, event_idx) == 1);
for ii=1:size(Label, 2),
    if ii==event_idx, continue; end;
    
    neg_idx_ii = find(Label(TrnInd, ii) == 1);
    randidx = randperm(length(neg_idx_ii));
    sel_idx = randidx(1:max_negative);
    sub_train_idx = [sub_train_idx; neg_idx_ii(sel_idx)];
end

all_idx = 1:num_data;

testindex = all_idx(TstInd);
trainindex = sub_train_idx';

num_data = length([trainindex, testindex]);
num_feature = size(bags(1).instance, 2);

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
