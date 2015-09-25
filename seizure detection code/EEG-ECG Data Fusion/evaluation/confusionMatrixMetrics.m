function [D, precision, recall, fScores, specificity, accuracy, accuracies, Bs] =...
	confusionMatrixMetrics(C)
	% C: Confusion matrix
	%
	% D: Normalised confusion matrix (each row adds up to 100%)

	% C = confusionmat(yTest, pred);
	numClasses = size(C,1);
	D = bsxfun(@rdivide, C, sum(C,2)) .* 100;

	falseAccuracy = computeAccuracy(C);
	accuracy = computeAccuracy(D);
	accuracies = zeros(1,numClasses);
	precision = zeros(1,numClasses);
	recall = zeros(1,numClasses);
	specificity = zeros(1,numClasses);
	fScores = zeros(1,numClasses);

	indexes = 1:numClasses;
	Bs = cell(numClasses,1);
	for k=indexes
		% n is a vector of the other indexes
		n = indexes(find(indexes ~= k));
		% B is a binary version of D
		% with respect to class k
		B = [D(k,k) sum(D(k,n));
			sum(D(n,k))/(numClasses-1) sum(sum(D(n,n)))/(numClasses-1)];
		Bs{k} = B;
		% TP: B(1,1)
		% FN: B(1,2)
		% FP: B(2,1)
		% TN: B(2,2)
		accuracies(k) = computeAccuracy(B);
		precision(k) = B(1,1) / (B(1,1) + B(2,1));
		recall(k) = B(1,1) / (B(1,1) + B(1,2));
		specificity(k) = B(2,2) / (B(2,2) + B(2,1));
		fScores(k) = (2*precision(k)*recall(k)) / (precision(k) + recall(k));
	end
end

function [accuracy] = computeAccuracy(C)
	% C is a confusion matrix
	accuracy = sum(diag(C)) / sum(sum(C));
end