function [numClust, centInd] = decisionGraph(rho, delta, selectMode, minRho, minDelta, numTopmost)
%%DECISIONGRAPH Decision graph for choosing the cluster centroids.
%   INPUT:
%       rho: local density [row vector]
%       delta: minimum distance between each point and any other point with higher density [row vector]
%       isManualSelect: 1 denote that all the cluster centroids are selected manually, otherwise 0
%  OUTPUT:
%       numClust: number of clusters
%       centInd:  centroid index vector
% Modified by D.U.P. from fileexchange/53922-densityclust

    NE = length(rho);
    numClust = 0;
    centInd = zeros(1, NE);
    
    if strcmp(selectMode, 'manual')
        
        fprintf('Manually select a proper rectangle to determine all the cluster centres (use Decision Graph)!\n');
        fprintf('The only points of relatively high *rho* and high  *delta* are the cluster centers!\n');
        plot(rho, delta, 's', 'MarkerSize', 7, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'b');
        title('Decision Graph', 'FontSize', 17);
        xlabel('\rho');
        ylabel('\delta');
        
        rectangle = getrect;
        minRho = rectangle(1);
        minDelta = rectangle(2);
        
        for i = 1 : NE
            if (rho(i) > minRho) && (delta(i) > minDelta)
                numClust = numClust + 1;
                centInd(i) = numClust;
            end
        end
        
    elseif strcmp(selectMode, 'fixed')
        for i = 1 : NE
            if (rho(i) > minRho) && (delta(i) > minDelta)
                numClust = numClust + 1;
                centInd(i) = numClust;
            end
        end
    elseif strcmp(selectMode, 'relative')
        rho_rel = double(rho ./ max(rho));
        delta_rel = double(delta ./ max(delta));
        for i = 1 : NE
            if (rho_rel(i) > minRho) && (delta_rel(i) > minDelta)
                numClust = numClust + 1;
                centInd(i) = numClust;
            end
        end
    elseif strcmp(selectMode, 'topmost')
        rho_rel = double(rho);%double(rho ./ max(rho));
        rho_rel(rho_rel < minRho) = 0;
        delta_rel = double(delta);%double(delta ./ max(delta));
        delta_rel(delta_rel < minDelta) = 0;
        gamma = rho_rel.*delta_rel;
        [~, idx_sorted] = sort(gamma, 'descend');
        
        for i = 1 : numTopmost
            numClust = numClust + 1;
            centInd(idx_sorted(i)) = numClust;
        end
    elseif strcmp(selectMode, 'topmost_otsu')
        minRho = graythresh(rho./max(rho))*max(rho);
        minDelta = graythresh(delta./max(delta))*max(delta);
        
        rho_rel = double(rho);%double(rho ./ max(rho));
        rho_rel(rho_rel < minRho) = rho_rel(rho_rel < minRho)*0.001;
        delta_rel = double(delta);%double(delta ./ max(delta));
        delta_rel(delta_rel < minDelta) = delta_rel(delta_rel < minDelta)*0.001;
        gamma = rho_rel.*delta_rel;
        [~, idx_sorted] = sort(gamma, 'descend');

        for i = 1 : numTopmost
            numClust = numClust + 1;
            centInd(idx_sorted(i)) = numClust;
        end
    elseif strcmp(selectMode, 'otsu')
        minRho = graythresh(rho./max(rho))*max(rho);
        minDelta = graythresh(delta./max(delta))*max(delta);
        for i = 1 : NE
            if (rho(i) > minRho) && (delta(i) > minDelta)
                numClust = numClust + 1;
                centInd(i) = numClust;
            end
        end
    end

end