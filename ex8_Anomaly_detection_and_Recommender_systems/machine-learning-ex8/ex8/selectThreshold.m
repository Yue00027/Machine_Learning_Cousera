function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;
F1_vec = size(pval);

stepsize = (max(pval) - min(pval)) / 1000;
% stepsize = (max(pval) - min(pval)) / 5;
for epsilon = min(pval):stepsize:max(pval)
%for epsilon = min(pval):max(pval)

    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions


%	printf('Epsilon %f\n ',epsilon);
	cvPredictions = (pval <= epsilon);
%	printf('Predictions %d \n ',cvPredictions(1:10));


	tp = sum((cvPredictions == 1) & (yval == 1));
	fp = sum((cvPredictions == 1) & (yval == 0));
	fn = sum((cvPredictions == 0) & (yval == 1));
%	printf('tp, fp, fn %f\n ',tp,fp,fn);


	prec = tp/(tp+fp);
	rec = tp/(tp+fn);

	F1 = 2*prec*rec/(prec+rec);
	
	F1_vec(end+1) = F1;

    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

% figure;
% plot([1:1:length(F1_vec)],F1_vec,'rx');
% axis([0 1000 0 2]);
% xlabel('Epsilon test step number');
% ylabel('F1');
% print _dpng 'F1.png'
% close;
end
