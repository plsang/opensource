% Proportional SVM Training for Videos
%
% Author: Kuan-Ting Lai
%
% This code is based on C. J. Lin's libLinear and Felix Yu's proportional SVM 
% The videos are bags, while frames or video clips in a vide are instances


function psvm_train(VidLabel, BagInstNum, InstDataVec, Proportion, C1Params, C2Params, SaveName)

% VidLabel:     Labels of videos (bags) 
% BagInstNum:   Number of instances in each bag (video)
% InstDataVec:  Feature vecotrs of instances
% Proportion:   The target proportion
% C1Params:     Cost weight for supervised learning loss
% C2Params:     Cost weight for label proportion loss
% SaveName:     The prefix of output model's filename


CLS_NEG_NO = 30; % Number of negatvie samples selected from each class

% Check the ground truth's type
if (size(VidLabel, 2) == 1)
    EventNum = max(VidLabel);   % Event ID label. 0 means background
    bColLabel = false;
else
    EventNum = size(VidLabel, 2); % Multi-column label; Each column represents one event
    bColLabel = true;
end

TotalFeatN = sum(BagInstNum(1:length(VidLabel)));

if (TotalFeatN ~= size(InstDataVec, 1))
    fprintf('Total instance number are mismatched between BagInstNum %d and InstDataVec %d\n', TotalFeatN, size(InstDataVec,1));
    return ;
end

FrmBagInd = zeros(TotalFeatN,1);
f = 1;
for i = 1:length(VidLabel)
    n = BagInstNum(i);
    if n > 0
        FrmBagInd(f:f+n-1) = i;
        f = f + n;
    end
end

for C2 = C2Params
    for p = 1:length(Proportion)
        proport = Proportion(p);
        for i = 1:EventNum
            if (bColLabel == false)
                PosVidInd = find(VidLabel == i);
            else
                PosVidInd = find(VidLabel(:, i) == 1); % Multi-column label
            end
            
            % Select negative training samples randomly from each class
            NegVidInd = [];
            for j = 0:EventNum  % 0 is background video
                if (i == j) continue; end
                
                if (bColLabel == false)
                    ind = find(VidLabel == j);
                else
                    if (j==0) % Background video has all-zero row
                        ind = find(sum(VidLabel, 2) == 0);
                    else
                        ind = find(VidLabel(:, j) == 1);
                    end
                end
                
                ind = ind(randperm(length(ind))); % Randomize matrix indices
                if (CLS_NEG_NO < length(ind))
                    ind = ind(1:CLS_NEG_NO);
                end
                NegVidInd = [NegVidInd; ind];
            end
            
            % Select sub-training set
            VidInd = [PosVidInd; NegVidInd];
            subFrmBagInd = [];  subDataInd = [];
            for j = 1:length(VidInd)
                subFrmBagInd = [subFrmBagInd; j*ones(BagInstNum(VidInd(j)), 1)];
                subDataInd = [subDataInd; find(FrmBagInd == VidInd(j))];
            end
            subData = InstDataVec(subDataInd, :);
            
            % Target Proportion(p)% and 0% positive instances in positive and negative bags, respectively
            % instance = frame, bag = video
            bagPortion = [proport*ones(length(PosVidInd),1); 0*ones(length(NegVidInd),1)];
            bagLabels = [ones(length(PosVidInd),1); -1*ones(length(NegVidInd),1)];
            Params.init_y = bagLabels(subFrmBagInd);
            
            fprintf('Training Event %d by pSVM\n', i);
            Params.method = 'alter-pSVM'; Params.C_2=C2; Params.ep = 0; Params.verbose=0;
            Params.max_iter = 50;
            for n=1:length(C1Params)
                fprintf('C1=%g, C2=%g\n', C1Params(n), C2);
                Params.C=C1Params(n); 
                model{n, i} = alternating_svm_linear_SVM(sparse(double(subData)), subFrmBagInd, bagPortion, Params);
            end
        end
    	save(sprintf('%s_c%g_p%g.mat', SaveName, C2, proport), 'model');
    end
end
