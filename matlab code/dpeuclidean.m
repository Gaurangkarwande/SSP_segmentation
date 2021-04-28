function [peaks, ptsC_DP, delta, haloInd] = dpeuclidean(points, MIN_RHO, MIN_DELTA, METHOD, N_TOPMOST)
%DPEUCLIDEAN Executes Density Peaks clustering
%   Modified by D.U.P. from fileexchange/53922-densityclust
    if(nargin < 5)
        N_TOPMOST = 0;
    end
    dist = pdist2(points, points);
    isHalo = 1;
    percNeigh = 0.02;
    kernel = 'Gauss';
    [dc, rho] = paraSet(dist, percNeigh, kernel); 
    [numClust, clustInd, centInd, haloInd, delta] = densityClust(dist, dc, rho, isHalo, METHOD , MIN_RHO, MIN_DELTA, N_TOPMOST);
    ptsC_DP = clustInd';
    peaks = find(centInd>0)';
end

