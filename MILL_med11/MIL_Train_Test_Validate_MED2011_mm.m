% Input pararmeter: 
% D: data array, including the feature data and output class
% outputfile: the output file name of classifiers
function run = MIL_Train_Test_Validate_MED2011_mm(input_file, classifier_wrapper_handle, classifier)

    global preprocess;
    
    feat_name = preprocess.feat_name;
    feat_dim = preprocess.feat_dim;
    num_agg = preprocess.num_agg;  % min_seg =4s, multiply by num_agg to form new seg
    
    if ~isempty(strfind(feat_name, 'bow')),
        MEDMD = load_metadata(feat_name, feat_dim, num_agg);
    elseif ~isempty(strfind(feat_name, 'fisher')),    
        MEDMD = load_metadata_fisher(feat_name, feat_dim, num_agg)
    else
        error('unknown feature <%s> \n', feat_name);
    end
 
    output_dir = sprintf('/net/per610a/export/das11f/plsang/trecvidmed11/experiments/MIL/%s', feat_name);
    if ~exist(output_dir, 'file'),
        mkdir(output_dir);
    end
    
    [classifier_name, para, additional_classifier] = ParseCmd(classifier, '--');
    
    output_file = sprintf('%s/%s.linear.start%d.end%d.neg%d.nagg%d.pool%s.mat', output_dir, classifier_name, preprocess.start_event, preprocess.end_event, preprocess.max_neg, preprocess.num_agg,preprocess.pool);
    
    runs = cell(preprocess.end_event-preprocess.start_event + 1, 1);
    
    for ii=preprocess.start_event:preprocess.end_event,
        [bags, num_data, num_feature, trainindex, testindex] = Load_MED2011(MEDMD, preprocess.max_neg, ii);
        run = feval(classifier_wrapper_handle, bags, trainindex, testindex, classifier);
        fprintf('** EventID = %d, AP = %f \n', ii, run.BagAccuMED);
        runs{ii-preprocess.start_event+1} = run;
    end
    
    fprintf('**** mAP = %f\n',  mean(cellfun(@(x) x.BagAccuMED, runs)));
    fprintf('Saving to <%s>\n', output_file);
    save(output_file, 'runs');
    
    clear MEDMD;
    
end



function MEDMD = load_metadata(feat_name, feat_dim, num_agg)

    filename='/net/per610a/export/das11f/plsang/trecvidmed/metadata/med11/medmd_2011.mat';
    fprintf('Loading meta file <%s>\n', filename);
    load(filename, 'MEDMD');

    root_fea_dir = '/net/per610a/export/das11f/plsang/trecvidmed/feature/med.pooling.seg4';
    fea_dir = sprintf('%s/%s', root_fea_dir, feat_name);
    
    fprintf('Loading feature...');
    %featMat, featNum
    MEDMD.featMat = cell(length(MEDMD.clips), 1);
    MEDMD.featNum = zeros(length(MEDMD.clips), 1);
    for ii=1:length(MEDMD.clips),
        if ~mod(ii, 100), fprintf('%d ', ii); end;
        
        video_id = MEDMD.clips{ii};
		
		if ~isfield(MEDMD.info, video_id),
			continue;
		end
		
        feat_pat = MEDMD.info.(video_id).loc;
        feat_file = sprintf('%s/%s.mat', fea_dir, feat_pat(1:end-4));
		if ~exist(feat_file, 'file'), continue; end;
		
        load(feat_file, 'code');
        total_unit_seg = size(code, 2);
        idxs = 1:num_agg:total_unit_seg;
        featMat_ = zeros(feat_dim, length(idxs));
        
        remove_last_seg = 0;
        for jj=1:length(idxs),
            start_idx = idxs(jj);
            end_idx = start_idx + num_agg - 1;
            if end_idx > total_unit_seg, end_idx = total_unit_seg; end;
            code_ = code(:, start_idx:end_idx);
            
            if any(any(isnan(code_), 1)),
                code_ = code_(:, ~any(isnan(code_), 1));
            end
            
            if isempty(code_) && end_idx == total_unit_seg,
                remove_last_seg = 1;
                break;
            end
            
            featMat_(:, jj) = sum(code_, 2);
            clear code_;
        end
        
        if remove_last_seg,
            fprintf('Last seg of video <%s> contains NaN. Removing...\n', feat_pat);
            featMat_(start_idx:end, :) = [];
        end
        
		non_zero_idx = any(featMat_);
		featMat_ = featMat_(:, non_zero_idx);
        featMat_ = l2_norm_matrix(featMat_);
        MEDMD.featMat{ii} = featMat_';
        MEDMD.featNum(ii) = size(featMat_, 2);
        
        clear code featMat_;
    end

end


function MEDMD = load_metadata_fisher(feat_name, feat_dim, num_agg)

    addpath('/net/per610a/export/das11f/plsang/codes/common/gmm-fisher/matlab');
    addpath('/net/per610a/export/das11f/plsang/codes/tvmed-framework-v2.0');

    fisher_params = struct;
    fisher_params.grad_weights = false;		% "soft" BOW
    fisher_params.grad_means = true;		% 1st order
    fisher_params.grad_variances = true;	% 2nd order
    fisher_params.alpha = single(1.0);		% power normalization (set to 1 to disable)
    fisher_params.pnorm = single(0.0);		% norm regularisation (set to 0 to disable)

    [~, param_dict] = get_coding_params();
    feat_key = strrep(feat_name, '.', '');
    if ~isfield(param_dict, feat_key),
        error('unknown feature <%s> \n', feat_name);
    end
       
    codebook = param_dict.(feat_key).codebook;
       
    filename='/net/per610a/export/das11f/plsang/trecvidmed/metadata/med12/medmd_2012.mat';
    fprintf('Loading meta file <%s>\n', filename);
    load(filename, 'MEDMD');

    root_fea_dir = '/net/per610a/export/das11f/plsang/trecvidmed/feature/med.pooling.seg4';
    fea_dir = sprintf('%s/%s', root_fea_dir, feat_name);
    
    fprintf('Loading feature...');
    %featMat, featNum
    MEDMD.featMat = cell(length(MEDMD.clips), 1);
    MEDMD.featNum = zeros(length(MEDMD.clips), 1);
    for ii=1:length(MEDMD.clips),
        if ~mod(ii, 100), fprintf('%d ', ii); end;
        
        video_id = MEDMD.clips{ii};
        feat_pat = MEDMD.info.(video_id).loc;
        feat_file = sprintf('%s/%s.stats.mat', fea_dir, feat_pat(1:end-4));
        load(feat_file, 'code');
        total_unit_seg = size(code, 2);
        idxs = 1:num_agg:total_unit_seg;
        featMat_ = zeros(feat_dim, length(idxs));
        error_idx = [];
        for jj=1:length(idxs),
            start_idx = idxs(jj);
            end_idx = start_idx + num_agg - 1;
            if end_idx > total_unit_seg, end_idx = total_unit_seg; end;
            code_ = code(:, start_idx:end_idx);
            
            if any(any(isnan(code_), 1)),
                code_ = code_(:, ~any(isnan(code_), 1));
            end
            
            stats = sum(code_, 2);
            cpp_handle = mexFisherEncodeHelperSP('init', codebook, fisher_params);
            code_= mexFisherEncodeHelperSP('getfkstats', cpp_handle, stats);
            mexFisherEncodeHelperSP('clear', cpp_handle);
            code_ = sign(code_) .* sqrt(abs(code_));    
            
            if any(isnan(code_)) || ~any(code_),
                error_idx = [error_idx, jj];
            else
                featMat_(:, jj) = code_;
            end
            
            clear code_ stats;
        end
        
        featMat_(:, error_idx) = [];
        featMat_ = l2_norm_matrix(featMat_);
        MEDMD.featMat{ii} = featMat_';
      
        MEDMD.featNum(ii) = length(idxs) - length(error_idx);
        clear code featMat_;
    end

end

function [bags, num_data, num_feature, trainindex, testindex] = Load_MED2011(medmd, max_neg, event_id)
    
    num_data = size(medmd.clips, 2);
    bags(num_data)=struct();
    for ii=1:num_data,
        bags(ii).name = medmd.clips{ii};
        bags(ii).inst_name = arrayfun(@(x) sprintf('HVC1000-%d', x), [1:medmd.featNum(ii)], 'UniformOutput', false);
        bags(ii).label = medmd.Label(ii, event_id);
        bags(ii).inst_label = bags(ii).label * ones(1, medmd.featNum(ii));
        bags(ii).instance = medmd.featMat{ii};
    end

    all_idx = 1:num_data;
    
    sub_train_idx = all_idx((medmd.Label(:, event_id) == 1) & medmd.TrnInd);
    
    for ii=1:size(medmd.Label, 2),
        if ii==event_id, continue; end;
        
        neg_idx_ii = all_idx((medmd.Label(:, ii) == 1) & medmd.TrnInd);
        
        if max_neg > length(neg_idx_ii),
            sub_train_idx = [sub_train_idx, neg_idx_ii];
        else
            randidx = randperm(length(neg_idx_ii));
            sel_idx = randidx(1:max_neg);
            sub_train_idx = [sub_train_idx, neg_idx_ii(sel_idx)];
        end
    end

    testindex = all_idx(medmd.TstInd);
    trainindex = sub_train_idx;

    num_data = length([trainindex, testindex]);
    num_feature = size(bags(1).instance, 2);
end


function X = l2_norm_matrix(X),
    for ii=1:size(X, 2),
        if any(X(:,ii) ~= 0), 
            X(:,ii) = X(:,ii) / norm(X(:,ii), 2);
        end
    end
end