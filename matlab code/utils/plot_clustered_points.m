function plot_clustered_points(points, ptsC, vargin)
%PLOT_CLUSTERED_POINTS plots points with colored labels
     colors = linspecer(numel(unique(ptsC)));    
%     colors_temp = colors;
% colors(6,:) = colors_temp(1,:);
% colors(1,:) = colors_temp(6,:);
% colors(5,:) = colors_temp(2,:);
% colors(4,:) = colors_temp(5,:);
% colors(2,:) = colors_temp(4,:);
%     if(size(colors,1) > 19)
%     colors2 = colors;
%     colors2(1,:) = colors(19,:);
%     colors2(10,:) = colors(1,:);
%     colors2(19,:) = colors(10,:);
%     colors = colors2;
%     end
    
    idx_noise = ptsC == 0;
    is_noisy = sum(idx_noise)>=1;
    
    unique_cluster_id = unique(ptsC);
    ptsC_unique = ptsC;
    for ii = 1:numel(unique_cluster_id)
        if(unique_cluster_id(ii) > 0)
            ptsC_unique(ptsC == unique_cluster_id(ii)) = ii-(1*is_noisy);
        end
    end
    
    plot(points(idx_noise,1), points(idx_noise,2), 'xr');
    for ii = 1:max(ptsC_unique)
        idx = find(ptsC_unique == ii);
        hold on;
        %plot(points(idx,1), points(idx,2), 'o', 'MarkerEdgeColor', getColor(ii, colors), 'MarkerFaceColor', getColor(ii, colors));
        scatter(points(idx,1), points(idx,2), 'filled', 'MarkerFaceColor', getColor(ii, colors));
    end
    
    show_legend = true;
    if (nargin>2) && (vargin(1)==false)
        show_legend = false;
    end
       
    if(show_legend)
        legends = cell(max(ptsC_unique) + (1*(is_noisy)), 1);

        if(is_noisy)
            legends{1} = 'Noise';
        end
        for ii=1:max(ptsC_unique)
            legends{ii+(1*is_noisy)} = ['Cluster ', num2str(ii)];
        end

        legend(legends);
    end
    
end

