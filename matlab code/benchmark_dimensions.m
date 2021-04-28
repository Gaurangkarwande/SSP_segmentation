%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Benchmark on multidimensional datasets                                  %
% Author: Pizzagalli D.U.                                                 %
% Date: 2019-07-05                                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;
clear all;
clc;

%% Dependencies
addpath './utils';  %Utilities
addpath './libs/colorbrewer/';  %Visually different colors
addpath './libs/dpclustering/'; %CDP implementation
addpath(genpath('./libs/mdcgen/config_build/src/'));                             
addpath(genpath('./libs/mdcgen/mdcgen/src'));
%Licenses to 3rd parties code is added in LICENSE.txt in each folder.

%% Settings
% Dataset generation
DIMENSIONS = [2, 10:10:100];    % Dimensions on which to test the algorithm
REPETITIONS = 15;               % Number of random datasets to generate foreach value of dimension
NUM_POINTS = 1000;              % Number of points to draw
NUM_CLUSTERS = 2;               % Number of clusters to include
TYPE_DISTRIBUTION = 2;          % Use multivariate gaussian distributions

% CDP initialization
DP_METHOD = 'topmost_otsu';     % Method to select density peaks: 'manual', 'fixed', 'topmost', 'topmost_otsu'
DP_MIN_RHO = 10;                % Used with DP_METHOD = 'fixed'
DP_MIN_DELTA = 0.5;             % Used with DP_METHOD = 'fixed'
kernel = 'Gauss';               % Kernel to estimate density
percNeigh = 0.02;               % Neighborhood size in percentage

% Graph settings
TH_PRUNING = 1000;

% Training
W = 5;                          % Length of path fragment
NUM_PATHS = 100;                % Number of training path in each class.

% MISC
VISUALIZATION_LEVEL = 1;        % 0: Do not show results. 1: Show final results, 2: Show intermediate results.
COLOR_CDP = [169 209 142]./255;
COLOR_GENERIC = [255 153 0]./255;
COLOR_TRAINED = [91 155 213]./255;

%% Initialization
MAX_DIMENSIONS = max(DIMENSIONS);
F1s_DP = zeros(REPETITIONS, MAX_DIMENSIONS);
Js_DP = zeros(REPETITIONS, MAX_DIMENSIONS);
F1s_DI = zeros(REPETITIONS, MAX_DIMENSIONS);
Js_DI = zeros(REPETITIONS, MAX_DIMENSIONS);  
F1s_DI_SVM = zeros(REPETITIONS, MAX_DIMENSIONS);
Js_DI_SVM = zeros(REPETITIONS, MAX_DIMENSIONS);

for rr = 1:REPETITIONS
    config.seed = rand()*1000000;
    for curr_d = DIMENSIONS
        config.nDatapoints = NUM_POINTS;                                         
        config.nDimensions = curr_d;                                            
        config.nClusters = NUM_CLUSTERS;
        config.distribution = zeros(1, config.nClusters)+TYPE_DISTRIBUTION;
        [ result ] = mdcgen( config );                                   

        if(VISUALIZATION_LEVEL > 1)
            scatter(result.dataPoints(:,1),result.dataPoints(:,2),10,'fill');  
            axis([0 1 0 1])                                                    
            points = result.dataPoints;
            ptsC_ref = result.label;
        end
        
    %% Begin generation of training paths %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        num_clusters = numel(unique(ptsC_ref));
        num_points = size(points, 1);
        paths_good = zeros(NUM_PATHS, W);
        paths_bad = zeros(NUM_PATHS, W);

        G = pdist2(points, points);

        % Good paths (Points in the same cluster)
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
        if(VISUALIZATION_LEVEL > 1)
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
%% End generation of training paths %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        [centers_DP, ptsC_DP, ~, haloInd] = dpeuclidean(points, DP_MIN_RHO, DP_MIN_DELTA, DP_METHOD, NUM_CLUSTERS);
        [centers_DI, ptsC_DI, path_DI, dist] = dpshortest_minimax(points, centers_DP, TH_PRUNING);
        [centers_DI_SVM, ptsC_DI_SVM, path_DI_SVM, dist_SVM] = dpshortest_svm(points, centers_DP, TH_PRUNING, rho, SVMModel, W);

        [TP_DP,FP_DP,TN_DP,FN_DP,precision_DP,recall_DP,F1_DP,J_DP] = evaluateClusteringResults(ptsC_DP, ptsC_ref);
        [TP_DI,FP_DI,TN_DI,FN_DI,precision_DI,recall_DI,F1_DI,J_DI] = evaluateClusteringResults(ptsC_DI, ptsC_ref); %maybe use the original one and exclude the noise from the evaluation
        [TP_DI_SVM,FP_DI_SVM,TN_DI_SVM,FN_DI_SVM,precision_DI_SVM,recall_DI_SVM,F1_DI_SVM,J_DI_SVM] = evaluateClusteringResults(ptsC_DI_SVM, ptsC_ref);

        F1s_DI(rr, curr_d) = F1_DI;
        Js_DI(rr, curr_d) = J_DI;

        F1s_DP(rr, curr_d) = F1_DP;
        Js_DP(rr, curr_d) = J_DP;

        F1s_DI_SVM(rr, curr_d) = F1_DI_SVM;
        Js_DI_SVM(rr, curr_d) = J_DI_SVM;

    end
end

F1s_DI = F1s_DI(:, DIMENSIONS);
F1s_DI_SVM = F1s_DI_SVM(:, DIMENSIONS);
F1s_DP = F1s_DP(:, DIMENSIONS);

%% Visualization
if(VISUALIZATION_LEVEL > 0)
    figure;
    NOISE_LEVELS = DIMENSIONS;
    plotshaded(NOISE_LEVELS, [min(F1s_DI_SVM); mean(F1s_DI_SVM); max(F1s_DI_SVM)], COLOR_TRAINED);
    hold on;
    plotshaded(NOISE_LEVELS, [min(F1s_DI); mean(F1s_DI); max(F1s_DI)], COLOR_GENERIC);
    hold on;
    plotshaded(NOISE_LEVELS, [min(F1s_DP); mean(F1s_DP); max(F1s_DP)], COLOR_CDP);
end