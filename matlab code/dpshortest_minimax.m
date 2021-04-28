%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Computes shortest path from a dummy start node, passing through         %
% peaks, to points.                                                       %
%                                                                         %
% Author: Diego Ulisse Pizzagalli                                         %
% Rev. 20190402-2311                                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [centers, labels, path, dist] = dpshortest_minimax(points, peaks, th_pruning)
%dpshortest assigns labels to points according with the shortest path from
%density peaks
%   points: array[num_pts x num_dim]. peaks: array[num_peaksx1] containing
%   the index of the centroids in points. th_pruning: scalar. 

%% Initialization and constants
    num_pts = size(points, 1);
    num_peaks = size(peaks, 1);
    labels = zeros(num_pts, 1);
    
    % array of weights
    w = pdist2(points, points);
    w = w(:);%.^2;
    EPS_START = min(w(w>0))*0.1;

    
    % array of from-to edge pairs
    pts_id = 1:size(points,1);
    [from, to] = meshgrid(pts_id, pts_id);
    from = from(:);
    to = to(:);
    
%% Graph pruning
    if(th_pruning > 0)
        from(w > th_pruning) = [];
        to(w > th_pruning)   = [];
        w(w > th_pruning)    = [];
    end
    max_pts_id = max([from(:); to(:)]);

%% Adding a dummy start node connected to peaks
    start_node_id = max_pts_id + 1; %dummy node
    start_edges_from = zeros(num_peaks,1) + start_node_id;
    start_edges_w = zeros(num_peaks,1) + EPS_START;
    
    from = [from; start_edges_from];
    to = [to; peaks];
    w = [w; start_edges_w];
    
    max_pts_id = max_pts_id + 1;
    
%% Executing Dijkstra SSSP from start, through peaks to points
    A = sparse(from, to, w, max_pts_id, max_pts_id);
    A = (A+A');
    [dist, path, pred] = dijkstra_SSSP_minimax(A, start_node_id);
    
%% Label assignment
    temp_labels = zeros(num_pts, 1);
    for ii = 1:num_pts
        curr_path = path{ii};
        if(numel(curr_path)>=2)
            temp_labels(ii) = curr_path(2);
        end
    end
    temp_labels(peaks) = peaks;
    
    unique_labels = unique(temp_labels);
    for ii = 1:numel(unique_labels)
        if unique_labels(ii) > 0
            labels(temp_labels == unique_labels(ii)) = ii;
        end
    end 
    centers = peaks;
end

