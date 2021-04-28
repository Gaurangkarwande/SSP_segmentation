function color = getColor( ii, colors )
    sz_col = size(colors,1);
    curr_idx = mod(ii-1, sz_col)+1;
    color = colors(curr_idx, :);
end

