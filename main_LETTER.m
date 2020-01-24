% This is the main function that apply a dictionary learning on handwritten
% letters trajectories in 2D (Sparse coding + dictionary learning)

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

nbData = 200; %Length of each trajectory

%% Generate handwriting data
%--------------------------------------------------------------------------
state="position";
[X]=genrateLettersDataset(state);
%% Online dictionary learning 
% defining Dataset parameters 
[N,M]=size(X);
k=12; %number of atoms 
lambda=0.1; % LASSO regularization
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

%% Integration of reconstructed velocity
tv = linspace(0, size(Reconstructed(1:size(X)/2,:),1), size(Reconstructed(1:size(X)/2,:),1))';
Reconstructed_x=cumtrapz(tv,Reconstructed(1:size(X)/2,:),1);
Reconstructed_y=cumtrapz(tv,Reconstructed((size(X)/2)+1:end,:),1); 
Reconstructed_pos=[Reconstructed_x;Reconstructed_y]; 

x=cumtrapz(tv,X(1:size(X)/2,:),1);
y=cumtrapz(tv,X((size(X)/2)+1:end,:),1); 
original_pos=[x;y]; 

close all 
for i=1:5 
    index=i+randi(300);
    figure()
    grid on
    
    
    plot(Reconstructed_pos(1:size(Reconstructed_pos,1)/2,index),...
        Reconstructed_pos((size(Reconstructed_pos,1)/2)+1:end,index),'r','LineWidth',LINEWIDTH);
     xlabel('Time step','Interpreter','latex','FontSize',14)
    ylabel('trajectory','Interpreter','latex','FontSize',14)
    hold on 
    plot(original_pos(1:size(original_pos,1)/2,index),...
        original_pos((size(original_pos,1)/2)+1:end,index),'g','LineWidth',LINEWIDTH);
    legend(["Reconstructed trajectory","Original trajectory"])
    xlabel('Time step','Interpreter','latex','FontSize',14)
    ylabel('trajectory','Interpreter','latex','FontSize',14)
end

%% Performance evaluation based on reconstruction error
close all 
% title("Difference Reconstructed Vs Real trajectories")
% Here we plot 20 random reconstructions 

for i=1:10 
    index=i+randi(300);
    figure()
    grid on 
    if state=="speed"
        vx=X(1:size(X,1)/2,index);
        vy=X((size(X,1)/2)+1:end,index);
        speed=sqrt(vx.^2+vy.^2);
        plot(speed,'b','LineWidth',LINEWIDTH);
     
        vx_tilde=Reconstructed(1:size(X,1)/2,index);
        vy_tilde=Reconstructed((size(X,1)/2)+1:end,index);
        speed_tilde=sqrt(vx_tilde.^2+vy_tilde.^2);
        hold on
        plot(speed_tilde,'r','LineWidth',LINEWIDTH);
        
        legend('Real velocity','Reconstructed velocity')
        xlabel('Time step','Interpreter','latex','FontSize',14)
        ylabel('velocity','Interpreter','latex','FontSize',14)
        
        figure()
        
        plot(Reconstructed_pos(1:size(Reconstructed_pos,1)/2,index),...
        Reconstructed_pos((size(Reconstructed_pos,1)/2)+1:end,index),'r','LineWidth',LINEWIDTH);
        
        hold on 
        plot(original_pos(1:size(original_pos,1)/2,index),...
         original_pos((size(original_pos,1)/2)+1:end,index),'g','LineWidth',LINEWIDTH);
        legend(["Reconstructed trajectory","Original trajectory"])
        
    elseif state=="position"
        plot(X(1:size(X,1)/2,index),X((size(X,1)/2)+1:end,index),'g','LineWidth',LINEWIDTH);

        hold on
        plot(Reconstructed(1:size(X,1)/2,index),Reconstructed((size(X,1)/2)+1:end,index),'r','LineWidth',LINEWIDTH);
        hold on 
        legend('Real trajectory','Reconstructed trajectory')
    else
        fprintf("INVALID STATE")
    end

    set(gca,  'fontsize', 12);
    pbaspect([2 1 1]) 
end
%% Plotting dictionary 
close all 
for i=1:10
    
   figure();
   title("Dictionary iteration "+i)
plot(D(1:200,i),D(201:400,i));
xlabel('Time step','Interpreter','latex','FontSize',14)
ylabel('trajectory','Interpreter','latex','FontSize',14)
 
set(gca,  'fontsize', 12);
pbaspect([2 1 1]) 

end

%% elbow_method to determine the best number of atoms 
 NbLetters=26;
 k_range=linspace(1,NbLetters,NbLetters); 
 nb_runs=10;
%[y]=apply_elbow(X,lambda,optsD,optsH,k_range,nb_runs); 
%% Plotting elbow
figure()
load('data/elbow_odl.mat');
AIC=zeros(length(k_range),1);
BIC=zeros(length(k_range),1);
for k=1:NbLetters
    B=k*N;
    AIC(k)=M*y(k)+B;
    BIC(k)=M*y(k)+B*log(M);
end
RSS=M*y;
plot(k_range,RSS,'LineWidth',3)
hold on 
plot(k_range,AIC,'LineWidth',3);
hold on 
%plot(k_range,BIC,'LineWidth',3);
grid on 
[min_aic , min_idx]=min(AIC);

plot(min_idx,min_aic,'o','LineWidth',3);

xlabel('nb of atoms','Interpreter','latex','FontSize',14)
ylabel('Performance metric','Interpreter','latex','FontSize',14)
legend(["RSS","AIC","Optimal K"])
%% Noise effect 
% Let's try low noise
snr=100;
h_noised_low=awgn(h,snr,'measured','linear'); 
Reconstructed_low=D*h_noised_low;
errors_low=immse(X,Reconstructed_low);

% Let's try medium noise 
snr=50;
h_noised_medium=awgn(h,snr,'measured','linear'); 
Reconstructed_medium=D*h_noised_medium;
errors_medium=immse(X,Reconstructed_medium);

% Let's try high noise 

snr=10;
h_noised_high=awgn(h,snr,'measured','linear'); 
Reconstructed_high=D*h_noised_high;
errors_high=immse(X,Reconstructed_high);

% Let's try very high noise 
snr=5;
h_noised_very_high=awgn(h,snr,'measured','linear'); 
Reconstructed_very_high=D*h_noised_very_high;
errors_very_high=immse(X,Reconstructed_very_high);
%% 
i=0;
errors_snr=zeros(101,1);
for snr=1:100
     i=i+1;
     h_noised=awgn(h,snr,'measured','linear'); 
    Reconstructed_noise=D*h_noised;
    errors_snr(i,1)=immse(X,Reconstructed_noise); 
end

figure()
plot(linspace(1,100,101)',errors_snr,'LineWidth',3)
hold on  
%plot(k_range,BIC,'LineWidth',3);
grid on 

xlabel("signal to noise ratio",'Interpreter','latex','FontSize',14)
ylabel("mean square error",'Interpreter','latex','FontSize',14)

%% 
h_noised=awgn(h,10,'measured','linear'); 
close all 
Reconstructed_noise=D*h_noised;
for i=1:20 
index=i+randi(300);
figure()
plot(X(1:200,index),X(201:400,index),'LineWidth',LINEWIDTH);
hold on
plot(Reconstructed_noise(1:200,index),Reconstructed_noise(201:400,index),'LineWidth',LINEWIDTH);
legend('Real','Reconstructed')
xlabel('Time step','Interpreter','latex','FontSize',14)
ylabel('trajectory','Interpreter','latex','FontSize',14)
 
set(gca,  'fontsize', 12);
pbaspect([2 1 1]) 
end