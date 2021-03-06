% This is the main function that apply a dictionary learning on Humaoid
% robot TALOS
% (Sparse coding + dictionary learning)

% Copyright (c) 2019 Idiap Research Institute, http://idiap.ch/
% Written by Hedi Fendri
% Supervised by Sylvain Calinon, http://calinon.ch/
% Created : 23/09/2019 
% Last modified: 20/12/2019
% 

clc 
close all 
clear all 
addpath('./m_fcts/');


%% Parameters

nbData = 100; %Length of each trajectory

%% Generate TALOS data
%--------------------------------------------------------------------------

%right_foot_data_2d.reshape(39,int(39000/39),100).transpose(1,2,0)

state="position";

X=load("data/two_feet_2d.txt");

X=X';
%[X,mu,sigma] = zscore(X,0,'all');
%%plot(joint_d(10,:))
% let's plot the joints 
%% Online dictionary learning 
% defining Dataset parameters 
[N,M]=size(X);
k=6; %number of atoms 
lambda=0.0; % LASSO regularization
%% Defining maximum iteration and tolerance to stop
optsH.max_iter=500;
optsD.max_iter=500;
optsH.tol=1e-6;
optsD.tol=1e-6;
%% Dictionary Learning 

[D,h,iter]= dictionaryLearning(X,lambda,k,optsD,optsH);

%% Plotting the difference between reconstructed and the real trajectories

Reconstructed=D*h; 
LINEWIDTH=2;    

%% Performance evaluation based on reconstruction error

% Plotting dictionary 
close all 
for i=1:k
   figure();
   title("Dictionary iteration "+i)

plot(D(:,i),'LineWidth',LINEWIDTH);
xlabel('Time step','Interpreter','latex','FontSize',14)
ylabel('trajectory','Interpreter','latex','FontSize',14)
 
set(gca,  'fontsize', 12);
pbaspect([2 1 1]) 

end

%% 
Delta_x=0.5;
close all 
for k=1:10
        index=k+randi(1000-20);
        figure(); 
    for i=1:39
        rank=1000*(i-1)+index;
         subplot(5,8,i); 
         plot(X(:,rank),'r','LineWidth',LINEWIDTH);
         X_mean=mean(X(:,rank),'all');
         ylim([X_mean-Delta_x,X_mean+Delta_x])
        hold on
        plot(Reconstructed(:,rank),'g','LineWidth',LINEWIDTH);
        hold on 
    end
end

%% elbow_method to determine the best number of atoms 
 Nbexperiments=20;
 k_range=linspace(1,Nbexperiments,Nbexperiments); 
 nb_runs=5;
[y]=apply_elbow(X,lambda,optsD,optsH,k_range,nb_runs); 
%% Plotting elbow
figure()
AIC=zeros(length(k_range),1);
BIC=zeros(length(k_range),1);
for k=1:Nbexperiments
    B=k*N+k*M;
    AIC(k)=1000*M*y(k)+B;
    BIC(k)=M*y(k)+B*log(M);
end
RSS=M*y;
plot(k_range,RSS,'LineWidth',3)
hold on 
%plot(k_range(1:15),AIC(1:15),'LineWidth',3);
hold on 
%plot(k_range,BIC,'LineWidth',3);
grid on 
[min_aic , min_idx]=min(AIC);

%plot(min_idx,min_aic,'o','LineWidth',3);

xlabel('nb of atoms','Interpreter','latex','FontSize',14)
ylabel('Performance metric','Interpreter','latex','FontSize',14)
legend(["RSS"])
%% Let's compare our implementation with RBF 
nbStates=6; 
width_range=linspace(1,40,40);
l=0;
%errors_rbf=zeros(1,length(width_range));
rbf_basis=define_RBF(nbStates,12);
 figure();
   title("RBF Dictionaries ")

plot(rbf_basis,'LineWidth',LINEWIDTH);
xlabel('Time step','Interpreter','latex','FontSize',14)
ylabel('trajectory','Interpreter','latex','FontSize',14)
 
set(gca,  'fontsize', 12);
pbaspect([2 1 1]) 


%%
for width=width_range
    l=l+1;
    % Loading rbf basis function generated by python notebook rbf.ipynb
    rbf_basis=define_RBF(nbStates,width);
    % let s plot the rbf basis functions 

    h_rbf=mldivide(rbf_basis,X);
    Reconstructed_rbf=rbf_basis * h_rbf;
    errors_rbf(l)=immse(X,Reconstructed_rbf);
end

%% training best rbf 

[~,best_width_rbf]=min(errors_rbf);
rbf_basis=define_RBF(nbStates,width_range(best_width_rbf));
h_rbf=mldivide(rbf_basis,X);
Reconstructed_rbf=rbf_basis * h_rbf;
close all 
plot(errors_rbf,'LineWidth',LINEWIDTH)
hold on 
plot(best_width_rbf,errors_rbf(best_width_rbf),'o','LineWidth',LINEWIDTH);
set(gca, 'YScale', 'log')
xlabel("width")
ylabel("MSE log scale")
%% Rbf reconstruction  
close all 
for k=1:10
        index=k+randi(1000-20);
        figure(); 
    for i=1:39
        rank=1000*(i-1)+index;
         subplot(5,8,i); 
         plot(X(:,rank),'r','LineWidth',LINEWIDTH);
        hold on
        plot(Reconstructed_rbf(:,rank),'b','LineWidth',LINEWIDTH);

    end
end


%% assessing the improvement on training and testing set

lambda=0;
nbruns=100;
k=8;
errors_dl_train = zeros(1,nbruns);
errors_rbf_train = zeros(1,nbruns);
errors_dl_test = zeros(1,nbruns);
errors_rbf_test = zeros(1,nbruns);
rbf_basis=define_RBF(nbStates,width_range(best_width_rbf));
for run=1:nbruns 
    % always radnomize 
    cv = cvpartition(size(X,2),'HoldOut',0.3);
    idx = cv.test;
    X_train = X(:,~idx);
    X_test  = X(:,idx);

    % Training 
    [D_train,h_train,~]= dictionaryLearning(X_train,lambda,k,optsD,optsH);
    h_rbf_train=mldivide(rbf_basis,X_train);
    % training error 
    X_dl_hat_train= D_train * h_train;
    X_rbf_hat_train= rbf_basis * h_rbf_train; 
    errors_dl_train(run)=immse(X_train,X_dl_hat_train);
    errors_rbf_train(run)=immse(X_train,X_rbf_hat_train);

    % Testing 
    h_test=mldivide(D_train,X_test);

    X_dl_hat_test= D_train * h_test;
    h_rbf_test=mldivide(rbf_basis,X_test); 
    X_rbf_hat_test= rbf_basis * h_rbf_test; 
    
    errors_dl_test(run)=immse(X_test,X_dl_hat_test);
    errors_rbf_test(run)=immse(X_test,X_rbf_hat_test);  
end
%% plotting box plots 
close all
figure()
group = [  1;2]; 
subplot(1,2,1)
boxplot(errors_dl_test)
hold on 
 plot(mean(errors_dl_test), 'dg')
 ylim([0.4e-5,1.2e-4])

 hold off 
text(50,50,"me,an = " )
ylabel("MSE")
title("Dictionary learning errors boxplot")
subplot(1,2,2) 
boxplot(errors_rbf_test)
 ylim([0.4e-5,1.2e-4])
title("Radial basis function error boxplot")
ylabel("MSE")


%% Let's try PCA
Reconstructed_pca = load("data/Reconstructed_pca.txt");
close all 
for k=1:10
        index=k+randi(1000-20);
        figure(); 
    for i=1:39
        rank=1000*(i-1)+index;
         subplot(5,4,i); 
         plot(X(:,rank),'r','LineWidth',LINEWIDTH);
        hold on
        
        plot(Reconstructed_pca(:,rank),'g','LineWidth',LINEWIDTH);
        hold on 
        plot(Reconstructed(:,rank),'-b','LineWidth',LINEWIDTH);

    end
    
end
 


