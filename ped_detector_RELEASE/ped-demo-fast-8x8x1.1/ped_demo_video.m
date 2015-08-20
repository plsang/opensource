function ped_demo_video(video_file, out_dir),
    if ~exist('out_dir', 'var'),
        out_dir = '/net/per920a/export/das14a/satoh-lab/plsang/yfcc100m/detect_human';
    end
    %% extract video files into frames
    tmp_dir = '/tmp';
    fprintf('--- step 1: extract video files into frames...\n');
    rate = 1;
    [~, video_name] = fileparts(video_file);
    
    out_file = sprintf('%s/%s.mp4', out_dir, video_name);
    if exist(out_file, 'file'),
        fprintf('File already exist \n');
        return;
    end
    
    output_dir = sprintf('%s/%s', tmp_dir, video_name);
    if ~exist(output_dir, 'file'),  
        mkdir(output_dir); 
        cmd = sprintf('ffmpeg -i %s -loglevel quiet -r %f %s/%s-%%6d.jpg', video_file, rate, output_dir, video_name);
        system(cmd);
    end
    
    detected_output_dir = sprintf('%s-detected', output_dir);
    %% 
    fprintf('--- step 2: detect human on each frames...\n');
    frames = dir([output_dir, '/*.jpg']);
    num_boxes = zeros(1, length(frames));
    for ii=1:length(frames),
        image_file = fullfile(output_dir, frames(ii).name);
        output_file = fullfile(detected_output_dir, frames(ii).name);
        if exist(output_file, 'file'), continue; end;
        num_box = ped_demo(image_file);
        num_boxes(ii) = num_box;
    end
    
    %% 
    fprintf('--- step 3: concat frames to videos...\n');
    
    cmd = sprintf('ffmpeg -framerate %f -i %s/%s-%%6d.jpg -c:v libx264 -r 29.97 %s/%s.mp4', rate, detected_output_dir, video_name, out_dir, video_name);
    system(cmd);
    
    %% remove direction
    cmd = sprintf('rm -rf %s', output_dir);
    system(cmd);
    cmd = sprintf('rm -rf %s', detected_output_dir);
    system(cmd);
    
    meta_dir = '/net/per920a/export/das14a/satoh-lab/plsang/yfcc100m/detect_human_metadata';
    meta_file = sprintf('%s/%s.txt', meta_dir, video_name);
    fh = fopen(meta_file, 'w');
    fprintf(fh, '%f %f %f\n', max(num_boxes), min(num_boxes), mean(num_boxes));
    fclose(fh);
end
