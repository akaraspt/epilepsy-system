function [z] = zScore(x)
	% x is a vector or a matrix
	% if it is a matrix, the z scores are calculated *by column*
	z = bsxfun(@rdivide, (x - repmat(mean(x, 1), size(x,1), 1)), std(x));
end 