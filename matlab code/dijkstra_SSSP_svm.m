%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Computes shortest path from a start node S_idx to all the nodes on G    %
% implementing Dijkstra's Single Source Shortest Path algorithm           %
% using a cost function determined by a trained path classifier		      %
% Author: Diego Ulisse Pizzagalli                                         %
% Rev. 20190402-2311                                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [best_distance, path, pred] = dijkstra_SSSP_svm(G, S_idx, rho, SVMModel,W_SIZE)
	MAX_DIST = max(999999, full(max(G(:))*2));
	num_pts = max(size(G));
    best_distance = zeros(num_pts, 1) + MAX_DIST;
	pred = zeros(num_pts, 1, 'uint32') + MAX_DIST;
    path = cell(num_pts, 1);
    visited = false(num_pts, 1);
    num_nodes = size(G,1);
    best_distance(S_idx) = 0;
    pred(S_idx) = 0;
    
    path{S_idx} = S_idx;

    count_it = 0;
    while(sum(visited) < num_pts)
        [~, curr_node] = min(best_distance + (visited * (MAX_DIST + 1))); %Excludes visited nodes.
        connected_to_curr_node = (G(curr_node,:) > 0)';
        connected_to_curr_node_and_not_visited = find(connected_to_curr_node & (~visited)); %Sparse & array
        curr_path = path{curr_node};
        
        if(numel(curr_path) > (W_SIZE-1))
            rho_1_W1 = rho(curr_path(end-(W_SIZE-2):end));
            proposed_paths_features = zeros(numel(connected_to_curr_node_and_not_visited), W_SIZE);
            proposed_paths_features(:,1:(W_SIZE-1)) = repmat(rho_1_W1, numel(connected_to_curr_node_and_not_visited), 1);
            proposed_paths_features(:,W_SIZE) = rho(connected_to_curr_node_and_not_visited);
            toAvoid = predict(SVMModel, proposed_paths_features);
        else
            toAvoid = zeros(numel(connected_to_curr_node_and_not_visited),1);
        end
        %foreach in connected neighborhood
        [better_distance, updated] = min([best_distance(connected_to_curr_node_and_not_visited)'; ...
            max( ...
                repmat(best_distance(curr_node), numel(connected_to_curr_node_and_not_visited), 1), ...
                G(connected_to_curr_node_and_not_visited, curr_node).*(toAvoid+1) ...
                )' ... %proposed = max(distance of curr node, distance of gap)
            ]); %comparison of proposed and best minimum
        
        best_distance(connected_to_curr_node_and_not_visited) = better_distance;
        updated = (updated - 1 > 0);
        updated_nodes = connected_to_curr_node_and_not_visited(updated);
        pred(updated_nodes) = curr_node;
        for ii=1:numel(updated_nodes)
            path{updated_nodes(ii)} = [path{curr_node}, updated_nodes(ii)];
        end
        visited(curr_node) = true;
        count_it = count_it + 1;
    end