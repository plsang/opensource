% Proportional SVM Training for Videos
%
% Author: Kuan-Ting Lai
%
% This code is based on C. J. Lin's libLinear and Felix Yu's proportional SVM 
% The videos are bags, while frames or video clips in a vide are instances


function psvm_train_sim(VidLabel, BagInstNum, InstDataVec, TrainSim, Proportion, C1Params, C2Params, SaveName, max_neg, svmlib, R)

% VidLabel:     Labels of videos (bags) 
% BagInstNum:   Number of instances in each bag (video)
% InstDataVec:  Feature vecotrs of instances
% Proportion:   The target proportion
% C1Params:     Cost weight for supervised learning loss
% C2Params:     Cost weight for label proportion loss
% SaveName:     The prefix of output model's filename

if ~exist('svmlib', 'var'),
    svmlib = 'libsvm';
end

if ~exist('max_neg', 'var'),
    CLS_NEG_NO = 30; % Number of negatvie samples selected from each class
else
    CLS_NEG_NO = max_neg;
end

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
            subSim = TrainSim(subDataInd, i);
            
            % Target Proportion(p)% and 0% positive instances in positive and negative bags, respectively
            % instance = frame, bag = video
            bagPortion = [proport*ones(length(PosVidInd),1); 0*ones(length(NegVidInd),1)];
            bagLabels = [ones(length(PosVidInd),1); -1*ones(length(NegVidInd),1)];
            Params.init_y = bagLabels(subFrmBagInd);
            
            min_sim = 0.01;
            % cal sim rank
            range = linspace(max(subSim), min_sim, R+1);
            simrank = zeros(length(subSim), 1);
            for ii=1:R,
                top = range(ii);
                if ii < R,
                    bottom = range(ii+1);
                else
                    bottom = min_sim;
                end 
                simrank(find(subSim <= top & subSim > bottom)) = ii;
            end
            
            range = linspace(0, min_sim, R+1);
            simrank_neg = zeros(length(subSim), 1);
            for ii=1:R,
                bottom = range(ii);
                if ii < R,
                    top = range(ii+1);
                else
                    top = min_sim;
                end 
                simrank_neg(find(subSim >= bottom & subSim < top)) = ii;
            end
            
            % init_y = double(simrank == 1);
            
            %% if there is no positive instance in the positive bag,
            %% choose the one that has the highest posibility
            % for j = 1:length(PosVidInd),
                % inst_idx = find(subFrmBagInd == j);
                % if ~any(init_y(inst_idx)),
                    % [~, max_inst_idx] = max(subSim(inst_idx));
                    % init_y(inst_idx(max_inst_idx)) = 1;
                    % find rank of the new 
                    % cur_rank = simrank(inst_idx(max_inst_idx));
                % end
            % end
            
            % init_y(init_y == 0) = -1;
            % Params.init_y = init_y;
            
            %% check positive bag without any positive instance label
            
            
            fprintf('Training Event %d by pSVM\n', i);
            Params.method = 'alter-pSVM'; Params.C_2=C2; Params.ep = 0; Params.verbose=0; Params.R = R;
            %Params.max_iter = 10;
            
            fprintf('cal train kernel...\n');
            train_kernel = subData*subData';
            fprintf('done...\n');
                
            for n=1:length(C1Params)
                fprintf('C1=%g, C2=%g\n', C1Params(n), C2);
                Params.C=C1Params(n); 
                model{n, i} = alternating_svm_linear_pre_sim(subData, train_kernel, subFrmBagInd, simrank, simrank_neg, bagPortion, Params); 
            end
        end
    	save(sprintf('%s.mat', SaveName), 'model');
    end
end
