%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script provides a demo of the clustering algorithm using a trainable
% path classifer. In this examples NUM_PATHS good and bad paths of length W
% are generated automatically from the ground truth.
% Then a SVM classifier is trained and integrated into the minimax
% path-cost function. Requires the Matlab Bioinformatics Toolbox
% Author: Pizzagalli D.U.
% Date: 2019-01-30
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all;
clear all;
clc;

%% SETTINGS
PATH_DATASETS = '..\data\clusteval_synthetic\'; % Path to datasets
FN           = '01_chang_pathbased.txt'; % Dataset name.
DP_METHOD    = 'manual'; % Method to select density peaks: 'manual', 'fixed', 'topmost', 'otsu_topmost'
DP_MIN_RHO   = 2;        % Used with DP_METHOD = 'fixed'
DP_MIN_DELTA = 10;       % Used with DP_METHOD = 'fixed'
TH_PRUNING   = 200;      % Prunes edges whose cost is higher. 0 no pruning.
W = 5;                   % Length of path fragment for training
NUM_PATHS = 50;          % Number of training path in each class.

%% DEPENDENCIES
addpath '.\utils';  %Utilities
addpath '.\libs\colorbrewer\';  %Visually different colors
addpath '.\libs\dpclustering\'; %CDP implementation
% Requires Matlab Bioinformatics Toolbox
%Licenses to 3rd parties code is added in LICENSE.txt in each folder.


%% INITIALIZATION
points = dlmread([PATH_DATASETS,FN], '\t', 5, 1);
ptsC_ref = dlmread([PATH_DATASETS,FN(1:end-3),'gs.txt'], '\t', 0, 1);
points_idx = randperm(size(points,1));
points = points(points_idx, :);
ptsC_ref = ptsC_ref(points_idx, :);


%% GENERATE PATHS FOR TRAINING
num_clusters = numel(unique(ptsC_ref));
num_points = size(points, 1);
paths_good = zeros(NUM_PATHS, W);
paths_bad = zeros(NUM_PATHS, W);

G = pdist2(points, points); %A graph (can be avoided for optimization).

% Good paths
for nn = 1:NUM_PATHS
    order = zeros(num_points, 1);
    idx = floor(rand(1)*num_points) + 1;
    curr_class = ptsC_ref(idx);
    same_class = ptsC_ref == ptsC_ref(idx);
    for ww = 1:W
        paths_good(nn,ww) = idx;
        order(idx) = ww;
        [~, idx2] = min(G(:,idx) + ((order>0)*9999) + ((~same_class)*9999));
        idx = idx2;
    end
end

% Bad paths cntaining one swap (from one cluster to another) error.
for nn = 1:NUM_PATHS
    order = zeros(num_points, 1);
    idx = floor(rand(1)*num_points) + 1;
    curr_class = ptsC_ref(idx);
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

%% VISUAIZATION (I) - TRAINING PATHS 
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

%% PATH DESCRIPTOR USING DENSITY PROFILE
percNeigh = 0.02;
kernel = 'Gauss';
[dc, rho] = paraSet(G, percNeigh, kernel); % From CDP
paths_good_features = rho(paths_good);
paths_bad_features = rho(paths_bad);

%% TRAINING
X = [paths_good_features; paths_bad_features];
y = [zeros(size(paths_good_features,1),1); ones(size(paths_bad_features,1),1)];

SVMModel = fitcsvm(X,y,'Standardize',true,'KernelFunction','RBF',...
    'KernelScale','auto');
CVSVMModel = crossval(SVMModel);
classLoss = kfoldLoss(CVSVMModel); %Generalization rate

%% CDP
figure;
[centers_DP, ptsC_DP] = dpeuclidean(points, DP_MIN_RHO, DP_MIN_DELTA, DP_METHOD);

%% Proposed algorithm with minimax
[centers_DI, ptsC_DI, path_DI] = dpshortest_minimax(points, centers_DP, TH_PRUNING);

%% Proposed algorithm with path classifier
[centers_DI_SVM, ptsC_DI_SVM, path_DI_SVM] = dpshortest_svm(points, centers_DP, TH_PRUNING, rho, SVMModel, W);


%% Evaluation
[TP_DP,FP_DP,TN_DP,FN_DP,precision_DP,recall_DP,F1_DP,J_DP] = evaluateClusteringResults(ptsC_DP, ptsC_ref);
[TP_DI,FP_DI,TN_DI,FN_DI,precision_DI,recall_DI,F1_DI,J_DI] = evaluateClusteringResults(ptsC_DI, ptsC_ref);
[TP_DI_SVM,FP_DI_SVM,TN_DI_SVM,FN_DI_SVM,precision_DI_SVM,recall_DI_SVM,F1_DI_SVM,J_DI_SVM] = evaluateClusteringResults(ptsC_DI_SVM, ptsC_ref);

disp(['F1 (ClusterDP) on ', FN, ' = ', num2str(F1_DP)]);
disp(['F1 (ClusterDP with Shortest Path) on ', FN, ' = ', num2str(F1_DI)]);
disp(['F1 (ClusterDP with Shortest Path + SVM) on ', FN, ' = ', num2str(F1_DI_SVM)]);

%% VISUALIZATION
figure;
subplot(221);
fn_str = strrep(FN, '_', '\_');
plot_clustered_points(points, ptsC_ref, false);
axis equal; axis square;
title('Ground truth');

subplot(222);
plot_clustered_points(points, ptsC_DP, false);
hold on;
plot(points(centers_DP, 1), points(centers_DP, 2), '*', 'MarkerEdgeColor', [0,0,0], 'MarkerSize', 4);
title(['ClusterDP F1=',num2str(F1_DP),' J=',num2str(J_DP)]);
axis equal; axis square;

subplot(223);
plot_clustered_points(points, ptsC_DI, false);
hold on;
plot(points(centers_DI, 1), points(centers_DI, 2), '*', 'MarkerEdgeColor', [0,0,0],  'MarkerSize', 4);
title(['DIJ minimax F1=',num2str(F1_DI),' J=',num2str(J_DI)]);
axis equal; axis square;

subplot(224);
plot_clustered_points(points, ptsC_DI_SVM, false);
hold on;
plot(points(centers_DI_SVM, 1), points(centers_DI_SVM, 2), '*', 'MarkerEdgeColor', [0,0,0],  'MarkerSize', 4);
title(['DIJ minimax SVM F1=',num2str(F1_DI_SVM),' J=',num2str(J_DI_SVM)]);
axis equal; axis square;