function [falsePositives, onsets, C] = falsePositivesAndOnsets(yTest, pred, seizures)
	% Confusion matrix
	C = confusionmat(yTest, pred);
	% False positives
	falsePositives = C(1,2);
	% Onsets
	onsets = ones(numel(seizures),1) .* inf;
	for s=1:numel(seizures)
		count = 0;
		for i=seizures{s}
			if pred(i) == 2
				onsets(s) = count;
				break;
			end
			count = count + 1;
		end
	end
end

