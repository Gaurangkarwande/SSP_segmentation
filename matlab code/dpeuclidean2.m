function [peaks, ptsC_DP, delta, haloInd] = dpeuclidean2(points, centroids_id)
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
    [numClust, clustInd, centInd, haloInd, delta] = densityClust2(dist, dc, rho, isHalo, centroids_id);
    ptsC_DP = clustInd';
    peaks = find(centInd>0)';
end
