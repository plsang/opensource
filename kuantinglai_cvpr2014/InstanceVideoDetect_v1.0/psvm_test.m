% Calculate mean AP on test dataset
function mAPs = psvm_test(TstLabel, featMat, featNum, ModelName, C1Params, C2Params)

ModelDir = 'models';
SaveDir = 'results';
%C1Params = [0.001, 1];
%C2Params = [0.01, 1, 100];
Proportion = [1];

StartEvent = 1;

for c2 = 1:length(C2Params)
    for p = 1:length(Proportion)
    fname = sprintf('%s/%s.mat', ModelDir, ModelName);
    if (~exist(fname, 'file'))
        fprintf('Cannot load %s\n', fname);
        continue;
    end
    
    load(fname, 'model');
    for c1 = 1:size(model, 1)
        mAP = 0;
        pred_scr = zeros(length(TstLabel), 1);
        totalEvents = size(model, 2);
        for e = StartEvent:size(model, 2)
            for i = 1:length(TstLabel)
                % Average pooling
                if (featNum(i) > 0)
                    pred = double(featMat{i})*model{c1, e}.w;
                    pred_scr(i) = sum(pred) / featNum(i);
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
        end
        mAPs(c1, c2) = mAP / (totalEvents-StartEvent+1);
        APs{c1, c2} = APVec;
        fprintf('C1=%f, C2=%f, P=%f, Mean AP= %f\n', C1Params(c1), C2Params(c2), Proportion(p), mAP / (totalEvents-StartEvent+1));
    end
    end
end

save(sprintf('%s/%s_AP.mat', SaveDir, ModelName), 'APs', 'C1Params', 'C2Params', 'Proportion');
