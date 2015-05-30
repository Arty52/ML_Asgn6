function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = dataset3Params(X, y, Xval, yval) returns your choice of C  
%   and sigma. You should complete this function to return the optimal C  
%   and sigma based on a cross-validation set.
%

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% Step sizes for both C and sigma
steps = [0.01 0.03 0.1 0.3 1 3 10 30]';

% Predictions based on chosen C and sigma for each iteration
predictions = zeros(length(steps));


for i = 1:length(steps)       % C
    
    for j = 1:length(steps)   % Sigma
        
       model = svmTrain(X, y, steps(i), @(x1, x2) gaussianKernel(x1, x2, steps(j)));
       prediction = svmPredict(model, Xval);
       predictions(i, j) = mean(double(prediction ~= yval));
       
    end
    
end

% find index of smallest element in the 'predictions' matrix
[error, row_index] = min(predictions);% Returns minimum value and row index
[~, index] = min(error);

C = steps(row_index(index));
sigma = steps(index);

% =========================================================================

end
