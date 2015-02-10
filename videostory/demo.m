function demo(am_file)

% add libsvm path for training event classifiers

root_dir='/net/per610a/export/das11f/plsang/dataset/videostory';
output_file=fullfile(root_dir, 'results', [am_file, '.txt']);

if exist(output_file, 'file'),
    fprintf('File (%s) already exist\n', output_file);
    return;
end

addpath(fullfile(root_dir, '/libs/libsvm-3.17/matlab'));

% load event train data - They should be l2 normalized
load(fullfile(root_dir, '/datasets/event_train/feature_mbh.mat'));
trnData = data';

% load event train labels for 10Ex setting
load(fullfile(root_dir,'/datasets/event_train/labels10Ex.mat'));
trnLabel = labels';

% load event test data - They should be l2 normalized
load(fullfile(root_dir,'/datasets/event_test/feature_mbh.mat'));
tstData = data';

% load event test labels
load(fullfile(root_dir,'/datasets/event_test/labels.mat'));
tstLabel = labels';

% augment the features with a bias parameter of 0.1 fixed in our
% experiments
trnData = [trnData ; 0.1*ones(1, size(trnData, 2))];
tstData = [tstData ; 0.1*ones(1, size(tstData, 2))];

% load the TRAINED video story mappings A and W for k = 1024, which is the
% optimal value as shown in the paper (Figure 5)
K = 1024;
%load(['./AW_k' num2str(K) '.mat']);
%load(['./AW_k' num2str(K) '_epoch9.mat']);

load(fullfile(root_dir, 'code', [am_file, '.mat']));

% obtain the videostory represnetation (S) for event train and test data by
% projecting them through the visual mapping W
trnDataX = W' * trnData;
tstDataX = W' * tstData;

% l2 normalization of train data
for i = 1:size(trnData, 2)
    trnDataX(:, i) = trnDataX(:, i) ./ norm(trnDataX(:, i), 2);
end
trnDataX(isnan(trnDataX)) = 0;

% l2 normalization of test data
for i = 1:size(tstDataX, 2)
    tstDataX(:, i) = tstDataX(:, i) ./ norm(tstDataX(:, i), 2);
end
tstDataX(isnan(tstDataX)) = 0;

%% perform event detection
aps = zeros(1, size(trnLabel, 1));

% for each event
for e = 1:size(trnLabel, 1)
    % prepare the event training data
    label = trnLabel(e, :);
    trnDataX2 = trnDataX(:, label ~= 0);
    label_train = label(label ~= 0);
    
    % prepare the event testing data
    label_test = tstLabel(e, :);
    tstDataX2 = tstDataX(:, label_test ~= 0);
    label_test = label_test(label_test ~= 0);        
    
    % train a RBF kernel SVM event classifier
    model = svmtrain(cast(label_train', 'double'), cast(trnDataX2', 'double'), '-t 2 -g 1 -q -c 100');    
    
    % apply the trained event classifier
    [~, ~, conf] = svmpredict(zeros(size(tstDataX2(1, :)))', cast(tstDataX2', 'double'), model);    
    
    % calculate the average precision
    aps(e) = calcMap(conf, label_test');
end

aps
mean(aps)
aps = [aps, mean(aps)];

dmlwrite(output_file, aps);

end
