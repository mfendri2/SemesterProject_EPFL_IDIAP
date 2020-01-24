% This is the main function that apply a dictionary learning on handwritten
% letters trajectories in 2D (Sparse coding + dictionary learning)

% Copyright (c) 2019 Idiap Research Institute, http://idiap.ch/
% Written by Hedi Fendri
% Supervised by Sylvain Calinon, http://calinon.ch/
% Created : 23/09/2019 
% Last modified: 30/09/2019
% 
% This file is part of PbDlib, http://www.idiap.ch/software/pbdlib/
% 
% PbDlib is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License version 3 as
% published by the Free Software Foundation.
% 
% PbDlib is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with PbDlib. If not, see <http://www.gnu.org/licenses/>.

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

X=load("data/two_feet_joint_features.txt");
X=X';
%%plot(joint_d(10,:))
% let's plot the joints 
%% Online dictionary learning 
% defining Dataset parameters 
[N,M]=size(X);
k=25; %number of atoms 
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
close all 
% title("Difference Reconstructed Vs Real trajectories")
% Here we plot 20 random reconstructions 
for i=1:50 
    index=i+randi(size(X,2)-50);
    figure()
    grid on 
    if state=="speed"
        vx=X(1:size(X,1)/2,index);
        vy=X((size(X,1)/2)+1:end,index);
        speed=sqrt(vx.^2+vy.^2);
        plot(speed,'r','LineWidth',LINEWIDTH);
     
        vx_tilde=Reconstructed(1:size(X,1)/2,index);
        vy_tilde=Reconstructed((size(X,1)/2)+1:end,index);
        speed_tilde=sqrt(vx_tilde.^2+vy_tilde.^2);
        hold on
        plot(speed_tilde,'--r','LineWidth',LINEWIDTH);
        
        legend('Real speed','Reconstructed speed')
        xlabel('Samples','Interpreter','latex','FontSize',14)
        ylabel('Speed','Interpreter','latex','FontSize',14)
    elseif state=="position"
        plot(X(:,index),'r','LineWidth',LINEWIDTH);

        hold on
        plot(Reconstructed(:,index),'g','LineWidth',LINEWIDTH);
        hold on 
        legend('Real trajectory','Reconstructed trajectory')
        xlabel('Samples','Interpreter','latex','FontSize',14)
        ylabel('trajectory','Interpreter','latex','FontSize',14)
         set(gca,  'fontsize', 12);
          pbaspect([2 1 1]) 
    else
        fprintf("INVALID STATE")
    end

end
%% Plotting dictionary 
close all 
for i=1:k
   figure();
   title("Dictionary iteration "+i)

plot(D(:,i),'LineWidth',LINEWIDTH);
xlabel('samples','Interpreter','latex','FontSize',14)
ylabel('trajectory','Interpreter','latex','FontSize',14)
 
set(gca,  'fontsize', 12);
pbaspect([2 1 1]) 

end
%% elbow_method to determine the best number of atoms 
 Nbexperiments=15;
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
%% 
close all 
X_to3d=X';
Reconstruct_to3d=Reconstructed'; 
X_to3d=reshape(X_to3d,[nbData,size(X_to3d,1)/nbData,size(X_to3d,2)]);
X_to3d=permute(X_to3d,[2,1,3]);
Reconstruct_to3d=reshape(Reconstruct_to3d,[nbData,size(Reconstruct_to3d,1)/nbData,size(Reconstruct_to3d,2)]);
Reconstruct_to3d=permute(Reconstruct_to3d,[2,1,3]);


for k=1:10
        index=k+randi(1000-20);
        figure(); 
        rank=1;
    for i=1:39

         subplot(5,8,i); 
         plot(X_to3d(index,:,i),'r','LineWidth',LINEWIDTH);

        hold on
        plot(Reconstruct_to3d(index,:,i),'g','LineWidth',LINEWIDTH);
        hold on 
    end
%          legend('Real trajectory','Reconstructed trajectory')
%         xlabel('Samples','Interpreter','latex','FontSize',14)
%         ylabel('trajectory','Interpreter','latex','FontSize',14)
end