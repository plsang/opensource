function check_med12
    load('med12_GT_info.mat');

    video_dir = '/net/per610a/export/das11f/plsang/dataset/MED/LDCDIST-RSZ';
    fea_dir = '/net/per610a/export/das11f/plsang/trecvidmed/feature/med.pooling.seg4/idensetraj.mbh.softbow.cb4000';
    
    %kf_fea_dir = '/net/per610a/export/das11f/plsang/trecvidmed/feature/keyframes/deepcaffe_1000';
    kf_fea_dir = '/net/per610a/export/das11f/plsang/trecvidmed/feature/keyframes/placehybridCNN.full';
    
  	fprintf('Loading metadata...\n');
	medmd_file = '/net/per610a/export/das11f/plsang/trecvidmed14/metadata/medmd_2014_devel_ps.mat';
	
	load(medmd_file, 'MEDMD'); 
	metadata = MEDMD.lookup;

    %% check if video exists?
    % count = 0;
    % for ii=1:length(fileList),
        % video_id = fileList{ii};
        % video_file = fullfile(video_dir, metadata.(video_id));
        % if ~exist(video_file),
            % fprintf('File <%s> does not exist\n', video_file);
            % count = count+1;
        % end
    % end
    % fprintf('Count video = %d \n', count);
    
    % count = 0;
    % for ii=1:length(fileList),
        % video_id = fileList{ii};
        
        % fea_file = fullfile(fea_dir, [metadata.(video_id)(1:end-4), '.mat']);
        % if ~exist(fea_file),
        %    fprintf('File <%s> does not exist\n', fea_file);
            % count = count + 1;
        % end
        
    % end
    % fprintf('Count mbh feature = %d \n', count);
    
    count = 0;
    output_file = 'med2012.txt';
    fh = fopen(output_file, 'w');
    for ii=1:length(fileList),
        video_id = fileList{ii};
        
        fea_file = fullfile(kf_fea_dir, fullfile(fileparts(metadata.(video_id)), video_id));
        if ~exist(fea_file),
            fprintf('File <%s> does not exist\n', fea_file);
            %fprintf(fh, '%s %s\n', video_id, metadata.(video_id));
            count = count + 1;
        end
        
    end
    fclose(fh);
    fprintf('Count deep feature = %d \n', count);    
    
end
