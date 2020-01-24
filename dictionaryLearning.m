function [D,h,iter]= dictionaryLearning(X,lambda,k,optsD,optsH)

%% Defining the initial dictionaray and sparse filter 
rand_= randi(size(X,2),1,k); 
D = normc(X(:,rand_)); %intializing with k random columns from X
h = zeros(size(D,2), size(X,2));
iter=0;

%% Dictionary learning
while(iter<optsD.max_iter)
    iter=iter+1;
    %% sparse coding update
    h=apply_fista(X,D,h,lambda,optsH);
    %h=normc(h);
    %% dictionary learning update
    A=h*h';
    B=X*h'; 
    Dold=D;
    sizeD = numel(D);
    for i = 1: size(D,2)
        if(A(i,i) ~= 0)
            a = 1.0/A(i,i) * (B(:,i) - D*A(:, i)) + D(:,i);
            D(:,i) = a/(max( norm(a,2),1));			
        end
    end 
    if (norm(D - Dold, 'fro')/sizeD < optsD.tol)
        break;
    end
    Dold = D;
    	  
end
