%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Do not use. Modified version of CDP forced to use seeds as density      %
% peaks for point-to-cluster assignment                                   %
% Modified by D.U.P. from fileexchange/53922-densityclust                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [peaks, ptsC_DP, delta, haloInd] = dpeuclidean_seeded(points, seeds, MIN_RHO, MIN_DELTA, METHOD)
%DPEUCLIDEAN Executes Density Peaks clustering
%   Detailed explanation goes here
    dist = pdist2(points, points);
    isHalo = 1;
    percNeigh = 0.02;%percNeigh = 0.02;
    kernel = 'Gauss';
    [dc, rho] = paraSet(dist, percNeigh, kernel); 
    [numClust, clustInd, centInd, haloInd, delta] = densityClust_seeded(dist, seeds, dc, rho, isHalo, METHOD , MIN_RHO, MIN_DELTA);
    ptsC_DP = clustInd';
    peaks = find(centInd>0)';
end

