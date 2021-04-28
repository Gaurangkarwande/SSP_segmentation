%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Computes shortest path from a start node S_idx to all the nodes on G    %
% implementing Dijkstra's Single Source Shortest Path algorithm           %
% using a minimax path cost function.                                     %
% Author: Diego Ulisse Pizzagalli                                         %
% Rev. 20190402-2311                                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [best_distance, path, pred] = dijkstra_SSSP_minimax_20190125(G, S_idx)
	MAX_DIST = max(999999, full(max(G(:))*2));
	num_pts = max(size(G));
    best_distance = zeros(num_pts, 1) + MAX_DIST;
	pred = zeros(num_pts, 1, 'uint32') + MAX_DIST;
    path = cell(num_pts, 1);
    visited = false(num_pts, 1);
    
    best_distance(S_idx) = 0;
    pred(S_idx) = 0;
    
    path{S_idx} = S_idx;
    path_length = zeros(num_pts, 1) + MAX_DIST;

    count_it = 0;
    while(sum(visited) < num_pts)
        [~, temp_node] = min(best_distance + (visited * (MAX_DIST + 1))); %Excludes visited nodes.
        temp_distance = best_distance(temp_node);
        temp_idx = find(best_distance == temp_distance & ~visited);
        
        [~, curr_temp_node] = min(path_length(temp_idx));
        curr_node = temp_idx(curr_temp_node);
        
        connected_to_curr_node = (G(:,curr_node) > 0); %20181116 connected_to_curr_node = (G(curr_node,:) > 0)';
        connected_to_curr_node_and_not_visited = find(connected_to_curr_node & (~visited)); %Sparse & array       
        
        same_cost = best_distance(connected_to_curr_node_and_not_visited) == max(G(connected_to_curr_node_and_not_visited, curr_node)*0+best_distance(curr_node),G(connected_to_curr_node_and_not_visited, curr_node));
        but_shorter = (path_length(curr_node) + 1) < path_length(connected_to_curr_node_and_not_visited);
        
        same_cost_but_shorter = same_cost & but_shorter;
        
        [better_distance, updated] = min([best_distance(connected_to_curr_node_and_not_visited)'; max(G(connected_to_curr_node_and_not_visited, curr_node)*0+best_distance(curr_node),G(connected_to_curr_node_and_not_visited, curr_node))']);
        
        best_distance(connected_to_curr_node_and_not_visited) = better_distance; %not yet minimax
        updated = (updated - 1 > 0);
        updated = updated | same_cost_but_shorter';

        
        updated_nodes = connected_to_curr_node_and_not_visited(updated);
        pred(updated_nodes) = curr_node;
        for ii=1:numel(updated_nodes)
            path{updated_nodes(ii)} = [path{curr_node}, updated_nodes(ii)];
            path_length(updated_nodes(ii)) = path_length(curr_node) + 1;
        end
        visited(curr_node) = true;
        count_it = count_it + 1;
    end