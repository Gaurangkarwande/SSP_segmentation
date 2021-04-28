warning on                                                         
warning('backtrace', 'off');                                     

addpath(genpath('config_build/src/'));                             
addpath(genpath('mdcgen/src'));                                  

config.nDatapoints = 2000;                                         
config.nDimensions = 30;                                            
config.nClusters = 3;                                              
%config.nOutliers = 4;                                              
config.distribution = zeros(1, config.nClusters)+2;         % use gaussian distributions                          

[ result ] = mdcgen( config );                                   

scatter(result.dataPoints(:,1),result.dataPoints(:,2),10,'fill');  
axis([0 1 0 1])                                                    
