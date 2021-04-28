%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script provides a demo of the clustering algorithm using Spearman  %
% distance for edge cost and minimax path cost.                           %
% Author: Pizzagalli D.U.                                                 %
% Date: 2019-07-05                                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;
clear all;
clc;

%% SETTINGS
PATH_DATASETS = '..\data\clusteval_bonemarrow\'; % Path to datasets
FN           = 'ALB_ALT_AML.txt';   % Dataset name.
DP_METHOD    = 'manual';            % Method to select density peaks: 'manual', 'fixed', 'topmost', 'otsu_topmost'
DP_MIN_RHO   = 10;                  % Used with DP_METHOD = 'fixed'
DP_MIN_DELTA = 20;                  % Used with DP_METHOD = 'fixed'
TH_PRUNING   = 1000000;             % Prunes edges whose cost is higher. 0 no pruning.

%% DEPENDENCIES
addpath '.\utils';  %Utilities
addpath '.\libs\colorbrewer\';  %Visually different colors
addpath '.\libs\dpclustering\'; %CDP implementation
%Licenses to 3rd parties code is added in LICENSE.txt in each folder.


%% INITIALIZATION
points = dlmread([PATH_DATASETS,FN], '\t', 5, 1); %RAW points
ptsC_ref = dlmread([PATH_DATASETS,FN(1:end-3),'gs.txt'], '\t', 0, 1); %GT

%% Build a graph from points using a distance derived from Spearman correlation
G = pdist2(points, points, 'spearman');
G = G - diag(diag(G)); % Froce element on the diagonal to be 0

%% DP
[centers_DP, ptsC_DP] = dpeuclidean_G(G, DP_MIN_RHO, DP_MIN_DELTA, 'manual');

%% DP with Shortest Path
[centers_DI, ptsC_DI, path_DI, dist] = dpshortest_minimax_G(G, centers_DP, TH_PRUNING);

%% Evaluation
[TP_DP,FP_DP,TN_DP,FN_DP,precision_DP,recall_DP,F1_DP,J_DP] = evaluateClusteringResults(ptsC_DP, ptsC_ref);
[TP_DI,FP_DI,TN_DI,FN_DI,precision_DI,recall_DI,F1_DI,J_DI] = evaluateClusteringResults(ptsC_DI, ptsC_ref);

disp(['F1 (ClusterDP) on ', FN, ' = ', num2str(F1_DP)]);
disp(['F1 (ClusterDP with Shortest Path) on ', FN, ' = ', num2str(F1_DI)]);

%% Visualization
fn_str = strrep(FN, '_', '\_');
figure;
subplot(131);
plot_clustered_points(points, ptsC_ref, false);
title('Ground truth');
axis equal; axis square;

subplot(132);
plot_clustered_points(points, ptsC_DP, false);
hold on;
plot(points(centers_DP, 1), points(centers_DP, 2), '*', 'MarkerEdgeColor', [0,0,0], 'MarkerSize', 4);
axis equal; axis square;
title(['ClusterDP F1=',num2str(F1_DP),' J=',num2str(J_DP)]);

subplot(133);
plot_clustered_points(points, ptsC_DI, false);
title(['DIJ minimax F1=',num2str(F1_DI),' J=',num2str(J_DI)]);
hold on;
plot(points(centers_DP, 1), points(centers_DI, 2), '*', 'MarkerEdgeColor', [0,0,0],  'MarkerSize', 4);
axis equal; axis square;