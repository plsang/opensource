function draw_det(image, win_posw,win_posh,winw,winh,scores, threshold)

    tmp_dir = '/net/per920a/export/das14a/satoh-lab/plsang/codes/ped_detector_RELEASE/tmp';
    [image_dir, image_name] = fileparts(image);
    [~,video_name] = fileparts(image_dir);
    
    output_dir = sprintf('%s/%s-detected', tmp_dir, video_name);
    if ~exist(output_dir, 'file'), mkdir(output_dir); end;
    output_file = sprintf('%s/%s.jpg', output_dir, image_name);
    output_file
    
    if exist(output_file, 'file'), return; end;
    
    indx = find(scores > threshold);
    %draw the figure
    img = imread(image); 
    %edge_colors={'r','g','b','c','m','y'};
    for i = 1:length(indx),
        ii = indx(i);
        det_rect = [win_posw(ii), win_posh(ii), winw(ii), winh(ii)];
        img = draw_rectangle(img, det_rect);
    end
    
    imwrite(img, output_file);  % Save modified image
end

function img = draw_rectangle(img, rect),
    c = [255, 0, 0];
    rect = round(rect);
    rect(rect <= 0) = 1;
    
    x = rect(1);
    y = rect(2);
    w = rect(3);
    h = rect(4);
    
    img(y, x:x+w, :) = repmat(c, w+1, 1);
    img(y:y+h, x, :) = repmat(c, h+1, 1);
    img(y+h, x:x+w, :) = repmat(c, w+1, 1);
    img(y:y+h, x+w, :) = repmat(c, h+1, 1);
end

function draw_det2(image, win_posw,win_posh,winw,winh,scores, threshold)

    tmp_dir = '/net/per920a/export/das14a/satoh-lab/plsang/codes/ped_detector_RELEASE/tmp';
    [image_dir, image_name] = fileparts(image);
    [~,video_name] = fileparts(image_dir);
    
    output_dir = sprintf('%s/%s-detected', tmp_dir, video_name);
    if ~exist(output_dir, 'file'), mkdir(output_dir); end;
    output_file = sprintf('%s/%s.jpg', output_dir, image_name);
    if exist(output_file, 'file'), return; end;
    
    indx = find(scores > threshold);
    %draw the figure
    figure('Visible', 'off');
    imshow(imread(image)); 
    hold on;
    edge_colors={'r','g','b','c','m','y'};
    for i = 1:length(indx)
            ii = indx(i);
            det_rect = [win_posw(ii), win_posh(ii), winw(ii), winh(ii)];
            cindx = randperm(length(edge_colors));
            rectangle('Position',det_rect,'EdgeColor',edge_colors{cindx(1)},'LineWidth',2);
            text(win_posw(ii),win_posh(ii),sprintf('%0.2f',scores(ii)),'Color','y');
    end
    
    set(gca,'position',[0 0 1 1],'units','normalized');
    
    print('-djpeg', output_file);
end

function draw_det_new(image, win_posw,win_posh,winw,winh,scores, threshold)
    indx = find(scores > threshold);
    %draw the figure
    img = imread(image);
    
    edge_colors={'r','g','b','c','m','y'};
    
    for i = 1:length(indx)
        ii = indx(i);
        det_rect = [win_posw(ii), win_posh(ii), winw(ii), winh(ii)];
        cindx = randperm(length(edge_colors));
        
        %rectangle('Position',det_rect,'EdgeColor',edge_colors{cindx(1)},'LineWidth',2);
        frect = @() rectangle('Position',det_rect);
        params = {'EdgeColor', edge_colors{cindx(1)}, 'LineWidth', 2};
        img = insertInImage(img, frect, params);
        
        %text(win_posw(ii),win_posh(ii),sprintf('%0.2f',scores(ii)),'Color','y');
        img = insertInImage(img, @()text(win_posw(ii), win_posh(ii), sprintf('%0.2f',scores(ii))),{'Color','y'});
    end
    
    imshow(img);
end
