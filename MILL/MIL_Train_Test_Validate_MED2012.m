% Input pararmeter: 
% D: data array, including the feature data and output class
% outputfile: the output file name of classifiers
function run = MIL_Train_Test_Validate_MED2012(input_file, classifier_wrapper_handle, classifier)

    global preprocess;

    filename='/net/per610a/export/das11f/plsang/codes/opensource/kuantinglai_cvpr2014/InstanceVideoDetect_v1.0/med12MBH_BOW_20s.mat';
    fprintf('Loading meta file <%s>\n', filename);
    medmd = load(filename);

    output_dir = '/net/per610a/export/das11f/plsang/trecvidmed12/experiments/MIL';
    
    [classifier_name, para, additional_classifier] = ParseCmd(classifier, '--');
    
    output_file = sprintf('%s/%s.linear.maxneg%d.mat', output_dir, classifier_name, preprocess.max_neg);
    
    runs = cell(preprocess.end_event-preprocess.start_event + 1, 1);
    
    for ii=preprocess.start_event:preprocess.end_event,
        [bags, num_data, num_feature, trainindex, testindex] = Load_MED2012(medmd, ii, preprocess.max_neg);
        run = feval(classifier_wrapper_handle, bags, trainindex, testindex, classifier);
        fprintf('** EventID = %d, AP = %f \n', ii, run.BagAccuMED);
        runs{ii-preprocess.start_event+1} = run;
    end
    
    fprintf('**** mAP = %f\n',  mean(cellfun(@(x) x.BagAccuMED, runs)));
    fprintf('Saving to <%s>\n', output_file);
    save(output_file, 'runs');
    
    clear medmd;
    
end



function [bags, num_data, num_feature, trainindex, testindex] = Load_MED2012(medmd, event_id, max_neg)
           
    num_data = size(medmd.featNum, 2);

    for ii=1:num_data,
        bags(ii).name = medmd.fileList{ii};
        bags(ii).inst_name = arrayfun(@(x) sprintf('HVC1000-%d', x), [1:medmd.featNum(ii)], 'UniformOutput', false);
        bags(ii).label = medmd.Label(ii, event_id);
        bags(ii).inst_label = bags(ii).label * ones(1, medmd.featNum(ii));
        bags(ii).instance = medmd.featMat{ii};
    end

    sub_train_idx = find(medmd.Label(medmd.TrnInd, event_id) == 1);
    for ii=1:size(medmd.Label, 2),
        if ii==event_id, continue; end;
        
        neg_idx_ii = find(medmd.Label(medmd.TrnInd, ii) == 1);
        
        if max_neg > length(neg_idx_ii),
            sub_train_idx = [sub_train_idx; neg_idx_ii];
        else
            randidx = randperm(length(neg_idx_ii));
            sel_idx = randidx(1:max_neg);
            sub_train_idx = [sub_train_idx; neg_idx_ii(sel_idx)];
        end
    end

    all_idx = 1:num_data;

    testindex = all_idx(medmd.TstInd);
    trainindex = sub_train_idx';

    num_data = length([trainindex, testindex]);
    num_feature = size(bags(1).instance, 2);
end
