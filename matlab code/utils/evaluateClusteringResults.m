%% Evaluates clustering scores giving a clustering result PtsC and a reference labeling ptsC_ref.
% Matches clustering labels with the mode of the labels assigned to the true
% positive.
% Scores are accumulated throughout all the cluster classes.

function [TP,FP,TN,FN,precision,recall,F1,J] = evaluateClusteringResults(ptsC, ptsC_ref)
    clusters_ref = unique(ptsC_ref);
    TP = 0;
    FP = 0;
    TN = 0;
    FN = 0;
    for ii = 1:numel(clusters_ref)
        label_ref = clusters_ref(ii);
        idx = (ptsC_ref == label_ref);
        ptsCcurr = ptsC(idx);
        label_mode = mode(ptsCcurr);

        TP = TP + sum((ptsC == label_mode) & (ptsC_ref == label_ref));
        FP = FP + sum((ptsC == label_mode) & (ptsC_ref ~= label_ref));
        TN = TN + sum((ptsC ~= label_mode) & (ptsC_ref ~= label_ref));
        FN = FN + sum((ptsC ~= label_mode) & (ptsC_ref == label_ref));
    end
    precision = TP/(TP+FP);
    recall = TP/(TP+FN);
    F1 = 2*(precision*recall)/(precision+recall);
    
    %Jaccard index relates the true positives to the number of pairs that either belong to the same class or are in the same cluster
    J = TP/(TP+FN+FP);
end

%     TP = sum(ptsCcurr==label_mode);
%     FP = sum(ptsC(~idx)==label_mode);
%     TN = sum(ptsC(~idx)~=label_mode);
%     FN = sum(ptsC(idx)~=label_mode);