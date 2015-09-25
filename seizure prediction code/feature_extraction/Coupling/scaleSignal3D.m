function x_norm = scaleSignal3D(x)
% x -- channel,band,time
D_x = size(x);
%To normalize a matrix such that all values fall in the range [0, 1] 
for i = 1:D_x(1)
    for j = 1:D_x(2)
        temp = squeeze(x(i,j,:));
        x_norm(i,j,:) = (temp - min(temp))/(max(temp) - min(temp));
    end
end