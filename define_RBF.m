function [Phi]=define_RBF(nbStates,width,offset, T, coeff)
% This function generates a radial basis function dictionary. 
% Copyright (c) 2019 Idiap Research Institute, http://idiap.ch/
% Written by Hedi Fendri
% Supervised by Sylvain Calinon, http://calinon.ch/
% Created : 23/09/2019 
% Last modified: 30/09/2019
if nargin==2 
    offset=5;
    T=100;
    coeff=250;
end
    tList =linspace(1,T,T);

    Mu = linspace(tList(1)-offset, tList(end)+offset, nbStates);
    Sigma  = reshape(repmat(width, 1, nbStates),[1, 1, nbStates]);
    Phi = zeros(T, nbStates);
    nb_states_range=linspace(1,nbStates,nbStates);
    for i=nb_states_range
    Phi(:,i) = coeff*normpdf(tList,Mu(i), Sigma(1,1,i));
    end
end