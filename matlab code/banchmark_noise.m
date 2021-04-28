%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Benchmark on synthetic distributions                                    %
% 20190702 0936
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;
clear all;
clc;

addpath '.\utils';  %Utilities
addpath '.\libs\colorbrewer\';  %Visually different colors
addpath '.\libs\dpclustering\'; %CDP implementation

%% Settings
% Synthetic distribution
NOISE_LEVELS = 0:0.01:0.1;  % Background probability (Note: this is not, but affects, the radio pf points in the clusters w.r.t. the points in the background)
REPLICAS = 5;               % Number of samples for each noise level.
N_SAMPLES_TO_DRAW = 1000;   % Number of points to extract from the density distribution
TH_HALO = 0.01;             % Overlapping halo regions to be considered as noise.
DATASET_SIZE = 200;         % Discretization
mu1 = [50,100];             % mean of distribution
sigma1 = [50 0; 0 1000];    % variance of distribution
mu2 = [100,30];             % mean of distribution
sigma2 = [100 0; 0 100];    % variance of distribution
mu3 = [100,80];             % mean of distribution
sigma3 = [100 0; 0 100];    % variance of distribution

% CDP
DP_METHOD    = 'topmost';   % Method to select density peaks: 'manual', 'fixed', 'topmost', 'topmost_otsu'
DP_NUM_TOPMOST   = 3;       % Number of peaks to select when using DP_METHOD = 'topmost' or 'topmost_otsu'
DP_MIN_DELTA = 0;           % Used with DP_METHOD = 'fixed'
DP_MIN_RHO = 0;             % Used with DP_METHOD = 'fixed'
kernel = 'Gauss';           % Kernel to estimate density
percNeigh = 0.02;           % Neighborhood size in percentage

TH_PRUNING   = 1000;        % Prunes edges whose cost is higher. 0 no pruning.

% Training
W = 5;                      % Length of path fragment
NUM_PATHS = 100;            % Number of training path in each class.

% MISC
ENABLE_VISUALIZATION = false;

%% Initialization
x1 = 1:DATASET_SIZE;
x2 = 1:DATASET_SIZE;
[X1,X2] = meshgrid(x1,x2);
X = [X1(:), X2(:)];


%% Generation of density distributions
y1 = mvnpdf(X,mu1,sigma1);
y1 = reshape(y1,length(x2), length(x1));

y2 = mvnpdf(X,mu2,sigma2);
y2 = reshape(y2,length(x2), length(x1));

y3 = mvnpdf(X,mu3,sigma3);
y3 = reshape(y3,length(x2), length(x1));

coeff_y1 = 1./(max(y1(:)));
y1 = y1.*coeff_y1;

coeff_y2 = 1./(max(y2(:)));
y2 = y2.*coeff_y2;

coeff_y3 = 1./(max(y3(:)));
y3 = y3.*coeff_y3;

%% Combining density distributions
M = max(max(y1,y2), y3);

%% Benchmarking
F1s_DP = zeros(REPLICAS, numel(NOISE_LEVELS));
Js_DP = zeros(REPLICAS, numel(NOISE_LEVELS));
F1s_DI = zeros(REPLICAS, numel(NOISE_LEVELS));
Js_DI = zeros(REPLICAS, numel(NOISE_LEVELS));
F1s_DI_SVM = zeros(REPLICAS, numel(NOISE_LEVELS));
Js_DI_SVM = zeros(REPLICAS, numel(NOISE_LEVELS));

SNRs = zeros(REPLICAS, numel(NOISE_LEVELS));

%% Generate points from a true density distribution %%%%%%%%%%%%%%%%%%%%%%%
% 1. Generate density distributions.
for ni = 1:numel(NOISE_LEVELS)
    for repetition = 1:REPLICAS
        %% Add background-noise (scale peaks to be less than 1).
        % point_distribution = point_distribution*(1-BG_LEVEL)+BG_LEVEL; % or use max.
        BG_LEVEL = NOISE_LEVELS(ni);
        
        M = max(M, BG_LEVEL); %M is the matrix with the probability for each element to be extracted.

        points = zeros(N_SAMPLES_TO_DRAW, 2);
        ptsC_ref = zeros(N_SAMPLES_TO_DRAW, 1);
        
        count_drawn = 0;
        while (count_drawn < N_SAMPLES_TO_DRAW)
            x = [floor(DATASET_SIZE*rand(1,2))+1 rand(1,1)];
            xx = x(1);
            yy = x(2);
            pp = x(3);
            if (M(yy, xx) > pp)
                already_added = (nnz((sum(points(1:count_drawn, :) == [xx,yy], 2) > 1)) > 0);
                if ~already_added            
                    count_drawn = count_drawn + 1;
                    points(count_drawn, :) = [xx, yy];
                    curr_height = [y1(yy,xx); y2(yy,xx); y3(yy,xx)];
                    H = pdist2(curr_height, curr_height);
                    m = min(max(H)); %minimum difference from another distribution. if vvalue is similar, it is in the halo.
                    if(m > TH_HALO)
                        [~, L] = max(curr_height);
                    else
                        L = 0;
                    end
                    ptsC_ref(count_drawn) = L;
                end
            end
        end

        for dd = 1:size(points, 2)
            points(:,dd) = points(:,dd) ./ DATASET_SIZE;
        end
%% End point generation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Begin generation of training paths %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        num_clusters = numel(unique(ptsC_ref));
        num_points = size(points, 1);
        paths_good = zeros(NUM_PATHS, W);
        paths_bad = zeros(NUM_PATHS, W);

        G = pdist2(points, points); %A graph (can be avoided for optimization).

        % Good paths (not in noise)
        for nn = 1:NUM_PATHS
            curr_class = 0;
            while(curr_class == 0)
                order = zeros(num_points, 1);
                idx = floor(rand(1)*num_points) + 1;
                curr_class = ptsC_ref(idx);
            end
            same_class = ptsC_ref == ptsC_ref(idx);
            for ww = 1:W
                paths_good(nn,ww) = idx;
                order(idx) = ww;
                [~, idx2] = min(G(:,idx) + ((order>0)*9999) + ((~same_class)*9999));
                idx = idx2;
            end
        end

        % Bad paths cntaining one swap (from one cluster to another) error.
        for nn = 1:floor(NUM_PATHS/2)
            order = zeros(num_points, 1);
            curr_class = 0;
            while(curr_class == 0)
                order = zeros(num_points, 1);
                idx = floor(rand(1)*num_points) + 1;
                curr_class = ptsC_ref(idx);
            end
            same_class = ptsC_ref == ptsC_ref(idx);
            is_swapped = false;
            for ww = 1:W
                paths_bad(nn,ww) = idx;
                order(idx) = ww;
                swap_class = rand(1) > 0.8;
                if(swap_class)
                    is_swapped = true;
                end
                if((ww == W) && (is_swapped == false))
                    swap_class = 1;
                    is_swapped = true;
                end
                [~, idx2] = min(G(:,idx) + ((order>0)*9999) + (((swap_class.*(~same_class)) + ((~swap_class).*same_class))*9999));
                idx = idx2;
            end
        end

        %% Bad paths in noise
        for nn = floor(NUM_PATHS/2)+1:NUM_PATHS
            order = zeros(num_points, 1);
            curr_class = 9;
            while(curr_class ~= 0)
                order = zeros(num_points, 1);
                idx = floor(rand(1)*num_points) + 1;
                curr_class = ptsC_ref(idx);
            end
            same_class = ptsC_ref == ptsC_ref(idx);
            for ww = 1:W
                paths_bad(nn,ww) = idx;
                order(idx) = ww;
                [~, idx2] = min(G(:,idx) + ((order>0)*9999) + ((~same_class)*9999));
                idx = idx2;
            end
        end

        %% VISUAIZATION (I) - TRAINING PATHS
        if(ENABLE_VISUALIZATION)
            figure;
            subplot(121);
            colors = lines(NUM_PATHS);
            plot(points(:,1), points(:,2), 'o' );%, 'MarkerFaceColor', 'b'
            for ii = 1:NUM_PATHS
                hold on;
                plot(points(paths_good(ii,:),1), points(paths_good(ii,:),2), '-', 'LineWidth', 2, 'Color', colors(ii,:));
            end
            axis equal; axis square;
            title('correct paths');

            subplot(122);
            colors = lines(NUM_PATHS);
            plot(points(:,1), points(:,2), 'o');%, 'MarkerFaceColor', 'b');
            for ii = 1:NUM_PATHS
                hold on;
                plot(points(paths_bad(ii,:),1), points(paths_bad(ii,:),2), '-', 'LineWidth', 2, 'Color', colors(ii,:));
            end
            axis equal; axis square;
            title('wrong paths');
        end

        %% PATH DESCRIPTOR USING DENSITY PROFILE
        [dc, rho] = paraSet(G, percNeigh, kernel); % From CDP
        paths_good_features = rho(paths_good);
        paths_bad_features = rho(paths_bad);

        %% TRAINING
        X = [paths_good_features; paths_bad_features];
        y = [zeros(size(paths_good_features,1),1); ones(size(paths_bad_features,1),1)];

        SVMModel = fitcsvm(X,y);
        CVSVMModel = crossval(SVMModel);
        classLoss = kfoldLoss(CVSVMModel); %Generalization rate
%% End generation of training paths %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %% DP
        if(strcmp(DP_METHOD, 'manual'))
            figure;
        end
        [centers_DP, ptsC_DP, delta, haloInd] = dpeuclidean(points, DP_MIN_RHO, DP_MIN_DELTA, DP_METHOD, DP_NUM_TOPMOST);
        seeds_orig = centers_DP;
        ptsC_DP = ptsC_DP.*(haloInd' > 0);

        %% Proposed algorithm with minimax
        [centers_DI, ptsC_DI, path_DI, dist_DI] = dpshortest_minimax(points, centers_DP, TH_PRUNING);

        %% Proposed algorithm with path classifier
        [centers_DI_SVM, ptsC_DI_SVM, path_DI_SVM, dist_SVM] = dpshortest_svm(points, centers_DP, TH_PRUNING, rho, SVMModel, W);

        %% Halo assignment
        dist_DI = dist_DI(1:end-1);
        th_dist_DI = graythresh(dist_DI);
        ptsC_DI(dist_DI > th_dist_DI) = 0;

        dist_SVM = dist_SVM(1:end-1);
        th_dist_SVM = graythresh(dist_SVM);
        ptsC_DI_SVM(dist_SVM > th_dist_SVM) = 0;
        
        [TP_DP,FP_DP,TN_DP,FN_DP,precision_DP,recall_DP,F1_DP,J_DP] = evaluateClusteringResults(ptsC_DP(ptsC_ref ~=0), ptsC_ref(ptsC_ref ~=0));
        [TP_DI,FP_DI,TN_DI,FN_DI,precision_DI,recall_DI,F1_DI,J_DI] = evaluateClusteringResults(ptsC_DI(ptsC_ref ~=0), ptsC_ref(ptsC_ref ~=0)); %maybe use the original one and exclude the noise from the evaluation
        [TP_DI_SVM,FP_DI_SVM,TN_DI_SVM,FN_DI_SVM,precision_DI_SVM,recall_DI_SVM,F1_DI_SVM,J_DI_SVM] = evaluateClusteringResults(ptsC_DI_SVM(ptsC_ref ~=0), ptsC_ref(ptsC_ref ~=0));
        
        F1s_DP(repetition, ni) = F1_DP;
        Js_DP(repetition, ni) = J_DP;
        
        F1s_DI(repetition, ni) = F1_DI;
        Js_DI(repetition, ni) = J_DI;
        
        F1s_DI_SVM(repetition, ni) = F1_DI_SVM;
        Js_DI_SVM(repetition, ni) = J_DI_SVM;
        
        if(ENABLE_VISUALIZATION)
            figure;
            subplot(221); plot_clustered_points(points, ptsC_ref, false); title(['BG_LEVEL = ',num2str(BG_LEVEL)]);
            subplot(222); plot_clustered_points(points, ptsC_DP, false); title(['CDP, F1 = ', num2str(F1_DP)]);
            subplot(223); plot_clustered_points(points, ptsC_DI, false); title(['Ours (generic), F1 = ', num2str(F1_DI)]);
            subplot(224); plot_clustered_points(points, ptsC_DI_SVM, false); title(['Ours (trained), F1 = ', num2str(F1_DI_SVM)]);
        end
        
        SNRs(repetition, ni) = nnz(ptsC_ref) ./ numel(ptsC_ref);
    end
end

COLOR_CDP = [169 209 142]./255;
COLOR_GENERIC = [255 153 0]./255;
COLOR_TRAINED = [91 155 213]./255;

figure;
plotshaded(NOISE_LEVELS, [min(F1s_DP); mean(F1s_DP); max(F1s_DP)], COLOR_CDP);
hold on;
plotshaded(NOISE_LEVELS, [min(F1s_DI); mean(F1s_DI); max(F1s_DI)], COLOR_GENERIC);
hold on;
plotshaded(NOISE_LEVELS, [min(F1s_DI); mean(F1s_DI_SVM); max(F1s_DI_SVM)], COLOR_TRAINED);

figure;
plotshaded(NOISE_LEVELS, [min(Js_DP); mean(Js_DP); max(Js_DP)], 'r');
hold on;
plotshaded(NOISE_LEVELS, [min(Js_DI); mean(Js_DI); max(Js_DI)], 'b');
hold on;
plotshaded(NOISE_LEVELS, [min(Js_DI); mean(Js_DI_SVM); max(Js_DI_SVM)], 'g');