% Calculate mean AP on test dataset
function mAPs = psvm_test_save(MEDMD, TstLabel, featMat, featNum, ModelName, C1Params, C2Params)

ModelDir = 'models';
SaveDir = 'results';
%C1Params = [0.001, 1];
%C2Params = [0.01, 1, 100];
Proportion = [1];

StartEvent = 1;

proj_dir = '/net/per610a/export/das11f/plsang/trecvidmed';
kf_dir = sprintf('%s/keyframes/', proj_dir);
output_dir='/net/per610a/export/das11f/plsang/codes/opensource/kuantinglai_cvpr2014/selkf';
            
tst_kfidx = MEDMD.kfidx(MEDMD.TstInd);
tst_clips = MEDMD.clips(MEDMD.TstInd);
for c2 = 1:length(C2Params)
    for p = 1:length(Proportion)
    fname = sprintf('%s/%s.mat', ModelDir, ModelName);
    if (~exist(fname, 'file'))
        fprintf('Cannot load %s\n', fname);
        continue;
    end
    
    load(fname, 'model');
    c1 = size(model, 1);
        mAP = 0;
        pred_scr = zeros(length(TstLabel), 1);
        
        pred_save = zeros(length(TstLabel), 2);
        
        totalEvents = size(model, 2);
        for e = StartEvent:size(model, 2)
            for i = 1:length(TstLabel)
                % Average pooling
                if (featNum(i) > 0)
                    pred = double(featMat{i})*model{c1, e}.w;
                    pred_scr(i) = sum(pred) / featNum(i);
                    
                    [max_pred, max_ind] = max(pred);
                    pred_save(i, :) = [max_pred, max_ind];
                end
            end
            
            if (size(TstLabel, 2) > 1)
                posInd = find(TstLabel(:, e) == 1);
                negInd = find(TstLabel(:, e) ~= 1);
            else
                posInd = find(TstLabel == e);
                negInd = find(TstLabel ~= e);
            end
            tstInd = [negInd; posInd];
            tstLbl = [zeros(length(negInd),1); ones(length(posInd),1)];
            ap = cal_AP(pred_scr(tstInd), tstLbl);            
            
            if (isnan(ap))
                ap = 0;
                totalEvents = totalEvents - 1;
                fprintf('No test sampels for event %d\n', e);
            else
                %fprintf('The AP of event %d = %f\n', e, ap);
            end
            APVec(e) = ap;
            mAP = mAP + ap;
            
            %% sort score descend
            [~, sorted_idx] = sort(pred_save(:, 1), 'descend');
            
            output_dir_e = sprintf('%s/E%03d', output_dir, e);
            if ~exist(output_dir_e, 'file'), mkdir(output_dir_e); end;
            
            %% save top 20 frames
            for jj=1:20,
                vid_idx = sorted_idx(jj);
                seg_idx = pred_save(vid_idx, 2);
                
                kfidx = tst_kfidx{vid_idx};
                start_kf = kfidx(1, seg_idx);
                end_kf = kfidx(2, seg_idx); 
                middle_kf = round((start_kf + end_kf)/2);
                video_id = tst_clips{vid_idx};
                
                video_kf_dir = fullfile(kf_dir, fullfile(fileparts(MEDMD.info.(video_id).loc), video_id));
                kfs = dir([video_kf_dir, '/*.jpg']);
                filepath = fullfile(video_kf_dir, kfs(middle_kf).name);
                
                
                
                if TstLabel(vid_idx, e) == 1,
                    output_dir_e_pos = sprintf('%s/E%03d/pos', output_dir, e);
                    if ~exist(output_dir_e_pos, 'file'), mkdir(output_dir_e_pos); end;
                

                    new_filepath = fullfile(output_dir_e_pos, sprintf('%d-%s', jj, kfs(middle_kf).name));                
                    
                    cmd = sprintf('cp %s %s', filepath, new_filepath);
                    system(cmd);
                else
                    output_dir_e_neg = sprintf('%s/E%03d/neg', output_dir, e);
                    if ~exist(output_dir_e_neg, 'file'), mkdir(output_dir_e_neg); end;
                    
                    new_filepath = fullfile(output_dir_e_neg, sprintf('%d-%s', jj, kfs(middle_kf).name));                
                    
                    cmd = sprintf('cp %s %s', filepath, new_filepath);
                    system(cmd);
                end
            end
        
        end
        mAPs(c1, c2) = mAP / (totalEvents-StartEvent+1);
        APs{c1, c2} = APVec;
        fprintf('C1=%f, C2=%f, P=%f, Mean AP= %f\n', C1Params(c1), C2Params(c2), Proportion(p), mAP / (totalEvents-StartEvent+1));
        

        
    end
end

save(sprintf('%s/%s_AP.mat', SaveDir, ModelName), 'APs', 'C1Params', 'C2Params', 'Proportion');
