function [ V, L, Mu ] = compute_pca( X )
%   In this function, the student should implement the Principal Component 
%   Algorithm following Eq.1, 2 and 3 of Assignment 1.
%
%   input -----------------------------------------------------------------
%   
%       o X      : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%
%   output ----------------------------------------------------------------
%
%       o V      : (M x M), Eigenvectors of Covariance Matrix.
%       o L      : (M x M), Eigenvalues of Covariance Matrix
%       o Mu     : (N x 1), Mean Vector of Dataset

% Auxiliary variables
[N, M] = size(X);

% Output variables
V  = zeros(M,M);
L  = zeros(M,M);
Mu = zeros(N,1);

%% ====================== Implement Eq. 1 Here ====================== 
%Centering the dataset
Mu=mean(X')';
X=X-Mu;
%% ====================== Implement Eq.2 Here ======================
%calculating the covaiance matrix
C=(1/(M-1))*X*X';
%% ====================== Implement Eq.3 Here ======================
[V,L]=eig(C);
%% =================== Sort Eigenvectors wrt. EigenValues ==========
eigenvalues=diag(L);
[vals,inds]=sort(eigenvalues,'descend');
V=V(:,inds);
L=diag(eigenvalues(inds));
end