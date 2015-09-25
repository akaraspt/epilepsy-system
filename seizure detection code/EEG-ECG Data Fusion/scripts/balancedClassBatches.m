function [N] = balancedClassBatches(y, n)
	N = zeros(n,2*numel(y(y==2)));
	for k=1:n
		% Balance classes
		normalIndexes = find(y==1);
		seizureIndexes = find(y==2);
		normalIndexes = normalIndexes(randsample(1:numel(normalIndexes),...
			numel(seizureIndexes) ));
		newIndexes = [normalIndexes seizureIndexes];
		N(k,:) = newIndexes;
	end
end