function [bags, num_data, num_feature, trainindex, testindex] = MIL_Data_Load_MED2012(event_idx)

%% param event_idx: 1-25
global preprocess;

filename='/net/per610a/export/das11f/plsang/codes/opensource/kuantinglai_cvpr2014/InstanceVideoDetect_v1.0/med12MBH_BOW_20s.mat';

medmd = load(filename);
       
num_data = size(medmd.featNum, 2);

max_negative = preprocess.max_neg;

for ii=1:num_data,
    bags(ii).name = medmd.fileList{ii};
    bags(ii).inst_name = arrayfun(@(x) sprintf('HVC1000-%d', x), [1:medmd.featNum(ii)], 'UniformOutput', false);
    bags(ii).label = medmd.Label(ii, event_idx);
    bags(ii).inst_label = bags(ii).label * ones(1, medmd.featNum(ii));
    bags(ii).instance = medmd.featMat{ii};
end

sub_train_idx = find(medmd.Label(medmd.TrnInd, event_idx) == 1);
for ii=1:size(medmd.Label, 2),
    if ii==event_idx, continue; end;
    
    neg_idx_ii = find(medmd.Label(medmd.TrnInd, ii) == 1);
    
    if max_negative > length(neg_idx_ii),
        sub_train_idx = [sub_train_idx; neg_idx_ii];
    else
        randidx = randperm(length(neg_idx_ii));
        sel_idx = randidx(1:max_negative);
        sub_train_idx = [sub_train_idx; neg_idx_ii(sel_idx)];
    end
end

all_idx = 1:num_data;

testindex = all_idx(medmd.TstInd);
trainindex = sub_train_idx';

num_data = length([trainindex, testindex]);
num_feature = size(bags(1).instance, 2);

