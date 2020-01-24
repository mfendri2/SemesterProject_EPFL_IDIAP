function [X, iter] = apply_fista(Y, D, Xinit, lambda, opts)
% This function solve the following convex optimazation problem :
% argmin (0.5) ||x(t)-Dh(t)||_2^2 + lambda*||h(t)||_1

% Copyright (c) 2019 Idiap Research Institute, http://idiap.ch/
% Written by Hedi Fendri
% Supervised by Sylvain Calinon, http://calinon.ch/
% Created : 23/09/2019 
% Last modified: 30/09/2019
%    
    %% Initial
    if numel(Xinit) == 0
            Xinit = zeros(size(D,2), size(Y,2));
    end
  
    %% gradient
    DtD = D'*D;
    DtY = D'*Y;
    function res = grad(X) 
        res = DtD*X - DtY;
    end 
    %% co
    L = max(eig(DtD));
    [X, iter] = fista(@grad, Xinit, L, lambda, opts);
end
function [X, iter] = fista(grad, Xinit, L, lambda, opts)   
% * A Fast Iterative Shrinkage-Thresholding Algorithm for 
% Linear Inverse Problems.

%   - INPUT:
%     - `grad`: a _function_ calculating gradient of `f(X)` given `X`.
%     - `Xinit`: initial guess.
%     - `L`: the Lipschitz constant of the gradient of `f(X)`.
%     - `lambda`: a regularization parameter
%     - `opts`: a _structure_ variable describing the algorithm.
%       * `opts.max_iter`: maximum iterations of the algorithm. 
%           Default `300`.
%       * `opts.tol`: a tolerance, the algorithm will stop if difference 
%           between two successive `X` is smaller than this value. 
%           Default `1e-8`.
%    
% -------------------------------------
% Copyright (c) 2019 Idiap Research Institute, http://idiap.ch/
% Written by Hedi Fendri Adapted from Tiep Vu (http://www.personal.psu.edu/thv102/)
% Please read  the following article that describes the Fast iterative
% shrinkage agorithm (FISTA) for more details: 
% A Fast Iterative Shrinkage-Thresholding Algorithm 
% for Linear Inverse Problems by Amir Beck  and Marc Teboulle
% Supervised by Sylvain Calinon, http://calinon.ch/
% Created : 23/09/2019 
% Last modified: 30/09/2019
% -------------------------------------    
    x_old = Xinit;
    y_old = Xinit;
    t_old = 1;
    iter = 0;
    %% MAIN LOOP
    while  iter < opts.max_iter
        iter = iter + 1;
        x_new = shrink(y_old - (1/L)*feval(grad, y_old),  lambda/L);
        t_new = 0.5*(1 + sqrt(1 + 4*t_old^2));
        y_new = x_new + (t_old - 1)/t_new * (x_new - x_old);
        %% check stop criteria
        e = norm(x_new - x_old,1)/numel(x_new);
        if e < opts.tol
            break;
        end
        %% update
        x_old = x_new;
        t_old = t_new;
        y_old = y_new;
    end
    X = x_new;
end 
function X = shrink(f, lambda)
% Copyright (c) 2019 Idiap Research Institute, http://idiap.ch/
% Written by Hedi Fendri
% Supervised by Sylvain Calinon, http://calinon.ch/
% Created : 23/09/2019 
% Last modified: 30/09/2019
% 
	X = max(0, f - lambda) + min(0, f+ lambda);
end