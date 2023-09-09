%% Motion Planning
clear all;
clc;
close all;
print_file = 0;     % whether to save the fig and print it to pdf. 
line_wd = 1.5;    
w_max = 1;          % amplitude of disturbance
figId = 0;
rccm_sim_data = 'sim_rccm_pos_NN__lam_0.5_w_dist_1_w_obs.mat'; % RCCM-P 
rccm_pos_sim_data = 'sim_rccm_pos_lam_0.5_w_dist_1_w_obs.mat'; % RCCM-SOS-P
rccm_nomTraj_data = 'nomTraj_w_obs_NN_rccm_0.5_wmax_1_plim_0.33pi_pos_rccm_0.01_w_input_penalty.mat';% RCCM-P 
rccm_pos_nomTraj_data = 'nomTraj_w_obs_rccm_0.5_wmax_1_plim_0.33pi_pos_rccm_0.01_w_input_penalty.mat';% RCCM-SOS-P
   
% For PVTOL 
n = 6;
nu = 2; 

% RCCM-P
load(rccm_sim_data);
times_rccm = times;
xTraj_rccm = xTraj;
uTraj_rccm = uTraj;
energyTraj_rccm = energyTraj; 
controller_rccm = controller;
controller_rccm.tube_xz = controller_rccm.tube_gain_xz*w_max;
controller_rccm.tube_x = controller_rccm.tube_xz;
controller_rccm.tube_z = controller_rccm.tube_xz; 
load(rccm_nomTraj_data);
nomTrajSoln = soln; 
simuLen = length(times);
xnomTraj_rccm = zeros(n,simuLen);
unomTraj_rccm = zeros(nu,simuLen);
for t =1:simuLen
    xnomTraj_rccm(:,t) = nomTrajSoln.interp.state(times(t));
    unomTraj_rccm(:,t) = nomTrajSoln.interp.control(times(t));
end
times_rccm = times;

fig_name_com = '_w_obs';
obs = sim_config.trajGen_config.obs;

% RCCM-P-SOS
load(rccm_pos_sim_data);
times_rccm_pos = times;
xTraj_rccm_pos = xTraj;
uTraj_rccm_pos = uTraj;
energyTraj_rccm_pos = energyTraj;
controller_rccm_pos = controller;
controller_rccm_pos.tube_xz = controller_rccm_pos.tube_gain_xz*w_max;
controller_rccm_pos.tube_x = controller_rccm_pos.tube_xz;
controller_rccm_pos.tube_z = controller_rccm_pos.tube_xz;
load(rccm_pos_nomTraj_data);
nomTrajSoln = soln; 
simuLen = length(times);
xnomTraj_rccm_pos = zeros(n,simuLen);
unomTraj_rccm_pos = zeros(nu,simuLen);
for t =1:simuLen
    xnomTraj_rccm_pos(:,t) = nomTrajSoln.interp.state(times(t));
    unomTraj_rccm_pos(:,t) = nomTrajSoln.interp.control(times(t));
end
times_rccm_pos = times;

%% ----------- Plot the trajectory -----------------
% Nominal trajectory
close all;
load('nomTraj.mat');
xF = trajGen_config.xF;
x0 = trajGen_config.x0;
x_nom_fcn = soln.interp.state;
u_nom_fcn = soln.interp.control;
simuLen = length(times);
xnomTraj = zeros(n,simuLen);
unomTraj = zeros(nu,simuLen);
for t =1:simuLen
    xnomTraj(:,t) = x_nom_fcn(times(t));
    unomTraj(:,t) = u_nom_fcn(times(t));
end

% color testing
color = {'k','b',[0 0.5 0],'r',[0.8 0.9 0.9741],[0.8 0.98 0.9],[1 0.8 0.8]};
linestyle = {':','--','-.','-'};
close all;
%% show the trajectories in X-Z plane
figId = figId + 1;
figure(figId);clf;
hold on;
scale = 20;
% RCCM-P tube
Len = length(times_rccm);
for i=1:Len/scale
    center = [xnomTraj_rccm(1,i*scale-1),xnomTraj_rccm(2,i*scale-1)];
   h3_tube= ellipse(controller_rccm.tube_xz,controller_rccm.tube_xz,0,center(1),center(2),color{6},[],1);
end

% RCCM-P-SOS tube
Len = length(times_rccm_pos);
for i=1:Len/scale
    center = [xnomTraj_rccm_pos(1,i*scale-1),xnomTraj_rccm_pos(2,i*scale-1)];
    h4_tube = ellipse(controller_rccm_pos.tube_xz,controller_rccm_pos.tube_xz,0,center(1),center(2),color{7},[],1);
end

h4 = plot(xTraj_rccm_pos(1,:),xTraj_rccm_pos(2,:),[color{4} linestyle{4}],'Linewidth',line_wd);
h3 = plot(xTraj_rccm(1,:),xTraj_rccm(2,:),'-','color',color{3},'Linewidth',line_wd);
h1 = plot(xnomTraj_rccm_pos(1,:),xnomTraj_rccm_pos(2,:),'k--','Linewidth',line_wd);   
plot(xnomTraj_rccm(1,:),xnomTraj_rccm(2,:),'k--','Linewidth',line_wd);    
plot(xnomTraj_rccm_pos(1,:),xnomTraj_rccm_pos(2,:),'k--','Linewidth',line_wd); 

visualize_obs(obs); 
arrow = annotation('arrow',[0.15 0.25],[0.55 0.55],'Linewidth',2);
txt_wind = text(-1.2,5.9,'Wind','Fontsize',13,'color','k');
xlabel('$p_x$ (m)','interpreter','latex');
ylabel('$p_z$ (m)','interpreter','latex')
    

legend([h1,h3,h4,h3_tube,h4_tube],{'Nominal','RCCM-P','RCCM-P-SOS'},'Location','northwest');
legend('boxon')
goodplot([6 5]);
fig_name = ['traj' fig_name_com];

if print_file == 1
    savefig([fig_name '.fig']);
    print([fig_name '.pdf'], '-painters', '-dpdf', '-r150');
end