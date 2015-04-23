function count_num_seg (feat_name, feat_dim)
    filename='/net/per610a/export/das11f/plsang/trecvidmed/metadata/med12/medmd_2012.mat';
    fprintf('Loading meta file <%s>\n', filename);
    load(filename, 'MEDMD');
    root_fea_dir = '/net/per610a/export/das11f/plsang/trecvidmed/feature/med.pooling.seg4';
    fea_dir = sprintf('%s/%s', root_fea_dir, feat_name);
    
    fprintf('Loading feature...');
    %featMat, featNum
    MEDMD.featMat = cell(length(MEDMD.clips), 1);
    MEDMD.featNum = zeros(length(MEDMD.clips), 1);
    
    total_unit_seg = 0;
    for ii=1:length(MEDMD.clips),
        if ~mod(ii, 100), fprintf('%d ', ii); end;
        
        video_id = MEDMD.clips{ii};
        feat_pat = MEDMD.info.(video_id).loc;
        feat_file = sprintf('%s/%s.mat', fea_dir, feat_pat(1:end-4));
        load(feat_file, 'code');
        total_unit_seg = total_unit_seg + size(code, 2);
        
    end
    fprintf('total_unit_seg = %g \n', total_unit_seg);
    total_duration = sum(cellfun(@(x) MEDMD.info.(x).duration, MEDMD.clips));
    fprintf('total duration = %g \n', total_duration);
    fprintf('mean duration = %g \n', total_duration/total_unit_seg);
end
