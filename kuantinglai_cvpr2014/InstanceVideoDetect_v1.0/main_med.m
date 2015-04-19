
%tic;
function main_med(run_name, max_neg)

    %C1Params = [0.1, 1, 10];
    %C2Params = [0.1, 1, 10];
    C1Params = [1];
    C2Params = [0.1, 1, 10];
    Proportion = 1;

    addpath(genpath('pSVM-master'));
    addpath('liblinear-1.95/matlab');

    feat_name = 'idensetraj.hoghof.hardbow.cb4000';
    feat_dim = 4000;
    num_agg = 5;  % min_seg =4s, multiply by num_agg to form new seg

    MEDMD = load_meddata(feat_name, feat_dim, num_agg);

    % Prepare training vectors
    TrainVec = cell2mat(MEDMD.featMat(MEDMD.TrnInd));
    size(TrainVec)
    TrnFeatNum = MEDMD.featNum(MEDMD.TrnInd);
    VidLabel = MEDMD.Label(MEDMD.TrnInd, :);
    OUT_NAME = sprintf('models/med12_psvm_mbh_20s_%s_mneg%d', run_name, max_neg);

    psvm_train(VidLabel, TrnFeatNum, TrainVec, Proportion, C1Params, C2Params, OUT_NAME, max_neg);

    TstLabel = MEDMD.Label(MEDMD.TstInd, :);
    featMat = MEDMD.featMat(MEDMD.TstInd);
    featNum = MEDMD.featNum(MEDMD.TstInd);

    psvm_test(TstLabel, featMat, featNum, run_name, C1Params, C2Params, max_neg);

end

function MEDMD = load_meddata(feat_name, feat_dim, num_agg)

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
        feat_file = sprintf('%s/%s.mat', fea_dir, feat_pat(1:end-4));
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
            featMat(start_idx:end, :) = [];
        end
        
        featMat_ = l2_norm_matrix(featMat_);
        MEDMD.featMat{ii} = featMat_';
        MEDMD.featNum(ii) = length(idxs);
        
        clear code featMat_;
    end

end
    
function X = l2_norm_matrix(X),
    for ii=1:size(X, 2),
        if any(X(:,ii) ~= 0), 
            X(:,ii) = X(:,ii) / norm(X(:,ii), 2);
        end
    end
end    