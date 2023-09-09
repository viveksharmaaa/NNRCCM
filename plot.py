from matplotlib import pyplot as plt
import numpy as np
import torch
from np2pth import get_system_wrapper, get_controller_wrapper
from mpl_toolkits.mplot3d import Axes3D
import importlib
from utils import EulerIntegrate
import time
from torchmetrics.functional import auc

import os
import sys
sys.path.append('systems')
sys.path.append('configs')
sys.path.append('models')
import argparse

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 15
HUGE_SIZE = 25


plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
plt.rc('legend', fontsize=15)    # legend fontsize
plt.rc('figure', titlesize=10)  # fontsize of the figure title
plt.rc('axes', axisbelow=True)

left = 0.14  # the left side of the subplots of the figure
right = 0.98   # the right side of the subplots of the figure
bottom = 0.17  # the bottom of the subplots of the figure
top = 0.925     # the top of the subplots of the figure

color_CCM = (128 / 255, 150 / 255, 244 / 255)
color_RCCM = (247 / 255, 112 / 255, 136 / 255)

parser = argparse.ArgumentParser(description="")
parser.add_argument('--task',type=str,default='PVTOL')
parser.add_argument('--pretrained_CCM', type=str)
parser.add_argument('--pretrained_RCCM', type=str)
parser.add_argument('--plot_type', type=str, default='error')
parser.add_argument('--plot_dims_pos', nargs='+', type=int, default=[0,1])
parser.add_argument('--nTraj', type=int, default=15)
parser.add_argument('--seed', type=int, default=4)
parser.add_argument('--sigma', type=float, default=1.0)
args = parser.parse_args()

np.random.seed(args.seed)

system = importlib.import_module('system_'+args.task)
f, B, Bw, _, num_dim_x, num_dim_control = get_system_wrapper(system)
controller_CCM = get_controller_wrapper(args.pretrained_CCM)
controller_RCCM = get_controller_wrapper(args.pretrained_RCCM)


CCM = torch.load(args.pretrained_CCM.split("/")[0] + '/' + args.pretrained_CCM.split("/")[1].replace('controller','model'), map_location=torch.device('cpu'))
CCM_tube = np.sqrt(CCM['args'].w_ub / CCM['args'].w_lb) * args.sigma / CCM['args']._lambda

RCCM = torch.load(args.pretrained_RCCM.split("/")[0] + '/' + args.pretrained_RCCM.split("/")[1].replace('controller','model'), map_location=torch.device('cpu'))

if RCCM.get('alpha') != None:
    alpha = RCCM['alpha']
else:
    alpha = RCCM['alpha_ref']

def w_distb(t_max = 10, dt = 0.05):
    t = np.arange(0, t_max, dt)
    w_dis = np.zeros((len(t), system.num_dim_distb,1))
    num = t[0] + np.random.rand()
    distb = np.random.uniform(0,args.sigma)
    w_distb = np.random.randn(system.num_dim_distb,1)
    if np.linalg.norm(w_distb) > distb:
        w_distb =  args.sigma * distb * w_distb/np.linalg.norm(w_distb)
    w_max = np.linalg.norm(w_distb)

    for i in range(len(t)):
        if (t[i] <= num):
            w = w_distb
        else:
            num = t[i] + np.random.rand()
            distb = np.random.uniform(0, args.sigma)
            w_distb = np.random.randn(system.num_dim_distb, 1)
            if np.linalg.norm(w_distb) > distb:
                w_distb =  args.sigma * distb * w_distb / np.linalg.norm(w_distb)
            w = w_distb
            w_max = np.linalg.norm(w) if np.linalg.norm(w) > w_max else w_max
        w_dis[i,:,:] = w
    return w_dis

if __name__ == '__main__':
    config = importlib.import_module('config_'+args.task)
    t = config.t
    time_bound = config.time_bound
    time_step = config.time_step
    XE_INIT_MIN = config.XE_INIT_MIN
    XE_INIT_MAX = config.XE_INIT_MAX


    fig = plt.figure(figsize=(8.0, 5.0))
    if args.plot_type=='3D':
        ax = fig.add_subplot(projection='3d')
    else:
        ax = fig.gca()

    if args.plot_type == 'time':
        cmap = plt.get_cmap('plasma')
        colors = [cmap(i) for i in np.linspace(0, 1, len(args.plot_dims_pos))]

    x_closed_CCM = []
    controls_CCM = []
    x_closed_RCCM = []
    x_closed_RCCMP = []
    controls_RCCMP = []
    controls_RCCM = []
    errors_CCM = []
    errors_CCMd = []
    errors_RCCM = []
    errors_RCCMd = []
    xinits = []
    errors_CCMdp = []
    errors_RCCMdp = []
    errors_RCCMp = []
    errors_CCMc = []
    errors_RCCMc = []
    xstar0 = []
    u_star = []
    x_star =[]
    wmax_CCM = []
    wmax_RCCM = []

    auc_CCMp = []
    auc_RCCMp = []
    for _ in range(args.nTraj):
        # Sampling different initial states xstar_0 and reference control trajectories uref
        x_0, xstar_0, ustar = config.system_reset(np.random.rand())
        ustar = [u.reshape(-1, 1) for u in ustar]
        xstar_0 = xstar_0.reshape(-1, 1)
        w = w_distb(time_bound, time_step)
        xstar, _= EulerIntegrate(None, f, B, Bw, None, ustar, xstar_0, time_bound, time_step, with_tracking=False,distb = w)
        x_star.append(xstar)
        xstar0.append(xstar_0)
        u_star.append(ustar)

        xinit = xstar_0 # Same initial conditions for nominal and actual trajectory
        xinits.append(xinit)
        x, u= EulerIntegrate(controller_CCM, f, B, Bw, xstar,ustar,xinit,time_bound,time_step,with_tracking=True,distb = w)
        x_closed_CCM.append(x)
        controls_CCM.append(u)
        start_time = time.time() #
        x_RCCM,u_RCCM = EulerIntegrate(controller_RCCM, f, B, Bw, xstar,ustar,xinit,time_bound,time_step,with_tracking=True,distb = w)
        #print("--- %s seconds ---" % (time.time() - start_time))
        x_closed_RCCM.append(x_RCCM)
        controls_RCCM.append(u_RCCM)

    for n_traj in range(args.nTraj):
        initial_dist_CCM = np.sqrt(((x_closed_CCM[n_traj][0] - x_star[n_traj][0])**2).sum())
        errors_CCMd.append([np.sqrt(((x - xs) ** 2).sum()) for x, xs in zip(x_closed_CCM[n_traj][:-1], x_star[n_traj][:-1])])
        errors_CCMdp.append([np.sqrt(((x[args.plot_dims_pos[0]:args.plot_dims_pos[-1] + 1] - xs[args.plot_dims_pos[0]:args.plot_dims_pos[-1] + 1]) ** 2).sum()) for x, xs in zip(x_closed_CCM[n_traj][:-1], x_star[n_traj][:-1])])
        errors_RCCMd.append([np.sqrt(((x - xs) ** 2).sum()) for x, xs in zip(x_closed_RCCM[n_traj][:-1], x_star[n_traj][:-1])])
        errors_RCCMdp.append([np.sqrt(((x[args.plot_dims_pos[0]:args.plot_dims_pos[-1] + 1] - xs[args.plot_dims_pos[0]:args.plot_dims_pos[-1] + 1]) ** 2).sum()) for x, xs in zip(x_closed_RCCM[n_traj][:-1], x_star[n_traj][:-1])])
        errors_CCMc.append([np.sqrt(((u - us) ** 2).sum()) for u, us in zip(controls_CCM[n_traj], u_star[n_traj])])
        errors_RCCMc.append([np.sqrt(((u - us) ** 2).sum()) for u, us in zip(controls_RCCM[n_traj], u_star[n_traj])])
        auc_CCMp.append(auc(torch.tensor(t),torch.as_tensor(errors_CCMdp)[n_traj,:]).item())
        auc_RCCMp.append(auc(torch.tensor(t), torch.as_tensor(errors_RCCMdp)[n_traj, :]).item())

        if initial_dist_CCM !=0:
           errors_CCM.append([np.sqrt(((x-xs)**2).sum()) / initial_dist_CCM for x, xs in zip(x_closed_CCM[n_traj][:-1],x_star[n_traj][:-1])])
        initial_dist_RCCM = np.sqrt(((x_closed_RCCM[n_traj][0] - x_star[n_traj][0])** 2).sum())
        if initial_dist_RCCM != 0:
            errors_RCCM.append([np.sqrt(((x - xs) ** 2).sum()) / initial_dist_RCCM for x, xs in zip(x_closed_RCCM[n_traj][:-1],x_star[n_traj][:-1])])
        if args.plot_type=='2D':
            plt.plot([x[args.plot_dims_pos[0],0] for x in x_closed_CCM[n_traj]], [x[args.plot_dims_pos[1],0] for x in x_closed_CCM[n_traj]], 'g', label='NN-CCM' if n_traj==0 else None)
            plt.plot([x[args.plot_dims_pos[0], 0] for x in x_closed_RCCM[n_traj]], [x[args.plot_dims_pos[1], 0] for x in x_closed_RCCM[n_traj]], 'k--', label='NN-RCCM' if n_traj == 0 else None)
            plt.title(f"{args.task} ($\sigma=$ {args.sigma})")
        elif args.plot_type=='3D':
            plt.plot([x[args.plot_dims_pos[0],0] for x in x_closed_CCM[n_traj]], [x[args.plot_dims_pos[1],0] for x in x_closed_CCM[n_traj]], [x[args.plot_dims_pos[2],0] for x in x_closed_CCM[n_traj]], 'g', label='NN-CCM' if n_traj==0 else None)
            plt.plot([x[args.plot_dims_pos[0], 0] for x in x_closed_RCCM[n_traj]], [x[args.plot_dims_pos[1], 0] for x in x_closed_RCCM[n_traj]], [x[args.plot_dims_pos[2], 0] for x in x_closed_RCCM[n_traj]], 'k--', label='NN-RCCM' if n_traj == 0 else None)
            plt.title(f"{args.task} ($\sigma=$ {args.sigma})")
        elif args.plot_type=='time':
            for i, plot_dim in enumerate(args.plot_dims_pos):
                plt.plot(t, [x[plot_dim,0] for x in x_closed_CCM[n_traj]][:-1], color=colors[i])
                plt.plot(t, [x[plot_dim, 0] for x in x_closed_RCCM[n_traj]][:-1], color=colors[i])
        elif args.plot_type=='error':
                plt.plot(t, [np.sqrt(((x - xs) ** 2).sum()) for x, xs in zip(x_closed_CCM[n_traj][:-1],x_star[n_traj][:-1])], 'g', label='NN-CCM' if n_traj == 0 else None)
                plt.plot(t, [np.sqrt(((x - xs) ** 2).sum())  for x, xs in zip(x_closed_RCCM[n_traj][:-1],x_star[n_traj][:-1])], 'k', label='NN-RCCM' if n_traj == 0 else None)
                plt.plot(t, np.repeat(args.sigma * alpha, len(t)), 'r-.', label='NN-RCCM Tube' if n_traj == 0 else None)
                plt.title(f"{args.task} ($\sigma=$ {args.sigma})")
                plt.yscale('log')

        elif args.plot_type=='error_pos':
                plt.plot(t, [np.sqrt(((x[args.plot_dims_pos[0]:args.plot_dims_pos[-1]+1] - xs[args.plot_dims_pos[0]:args.plot_dims_pos[-1]+1]) ** 2).sum())  for x, xs in zip(x_closed_CCM[n_traj][:-1], x_star[n_traj][:-1])], 'g', label='NN-CCM' if n_traj == 0 else None)
                plt.plot(t, [np.sqrt(((x[args.plot_dims_pos[0]:args.plot_dims_pos[-1]+1] - xs[args.plot_dims_pos[0]:args.plot_dims_pos[-1]+1]) ** 2).sum())  for x, xs in zip(x_closed_RCCM[n_traj][:-1], x_star[n_traj][:-1])], 'k--', label='NN-RCCM-P' if n_traj == 0 else None)
                if n_traj == 0:
                    plt.plot(t, np.repeat(args.sigma * alpha, len(t)), 'r-.',label='NN-RCCM-P Tube')
                plt.title(f"{args.task} ($\sigma=$ {args.sigma})")
                plt.yscale('log')

        elif args.plot_type=='control_error':
                plt.plot(t, [np.sqrt(((u - us) ** 2).sum())  for u, us in zip(controls_CCM[n_traj], u_star[n_traj])], 'g', label='NN-CCM' if n_traj == 0 else None)
                plt.plot(t, [np.sqrt(((u - us) ** 2).sum())  for u, us in zip(controls_RCCM[n_traj], u_star[n_traj])], 'k--', label='NN-RCCM' if n_traj == 0 else None)
                if n_traj ==0:
                    plt.plot(t, np.repeat(alpha, len(t)), 'r-.', label='NN-RCCM Tube')
                plt.title(f"{args.task} ($\sigma=$ {args.sigma})")
                plt.yscale('log')
        elif args.plot_type=='augmented':
            plt.plot(t, [np.sqrt(((np.concatenate((x,u),axis=0) - np.concatenate((xs,us),axis=0)) ** 2).sum()) for x, xs, u, us in zip(x_closed_CCM[n_traj][:-1],x_star[n_traj][:-1],controls_CCM[n_traj], u_star[n_traj])], 'g', label='CCM' if n_traj == 0 else None)
            plt.plot(t, [np.sqrt(((np.concatenate((x,u),axis=0) - np.concatenate((xs,us),axis=0)) ** 2).sum()) for x, xs, u, us in zip(x_closed_RCCM[n_traj][:-1],x_star[n_traj][:-1],controls_RCCM[n_traj], u_star[n_traj])], 'k--', label='RCCM' if n_traj == 0 else None)
            plt.title(f"{args.task} ($\sigma=$ {args.sigma})")
            plt.yscale('log')


    if args.plot_type == "mean_std_plot":
        plt.fill_between(t, torch.as_tensor(errors_CCMd).mean(dim=0) - (1.0 * torch.as_tensor(errors_CCMd).std(dim=0)), torch.as_tensor(errors_CCMd).mean(dim=0) + (1.0 * torch.tensor(errors_CCMd).std(dim=0)),facecolor=(*color_CCM, 0.15),
        edgecolor=(0, 0, 0, 0))
        plt.plot(t, torch.as_tensor(errors_CCMd).mean(dim=0),color=color_CCM, linewidth=2.5,label='NN-CCM')
        plt.fill_between(t, torch.as_tensor(errors_RCCMd).mean(dim=0) - (1.0 * torch.as_tensor(errors_RCCMd).std(dim=0)), torch.as_tensor(errors_RCCMd).mean(dim=0) + (1.0 * torch.as_tensor(errors_RCCMd).std(dim=0)), facecolor=(*color_RCCM, 0.15))
        plt.plot(t, torch.as_tensor(errors_RCCMd).mean(dim=0), color=color_RCCM, linewidth=2.5, label='NN-RCCM')
        plt.plot(t, np.repeat(args.sigma * alpha, len(t)), 'k--', linewidth=2.5, label='NN-RCCM Tube')
        plt.yscale('log')
        plt.title(f"{args.task} ($\sigma=$ {args.sigma})")

    elif args.plot_type == "mean_std_plot_control":
        plt.fill_between(t, torch.as_tensor(errors_CCMc).mean(dim=0) - (1* torch.as_tensor(errors_CCMc).std(dim=0)), torch.as_tensor(errors_CCMc).mean(dim=0)  + (1 * torch.tensor(errors_CCMc).std(dim=0)),facecolor=(*color_CCM, 0.15),
        edgecolor=(0, 0, 0, 0))
        plt.plot(t, torch.as_tensor(errors_CCMc).mean(dim=0), color=color_CCM, linewidth=2.5,label='NN-CCM')
        plt.fill_between(t, torch.as_tensor(errors_RCCMc).mean(dim=0) - (1* torch.as_tensor(errors_RCCMc).std(dim=0)), torch.as_tensor(errors_RCCMc).mean(dim=0) + (1* torch.as_tensor(errors_RCCMc).std(dim=0)), facecolor=(*color_RCCM, 0.15))
        plt.plot(t, torch.as_tensor(errors_RCCMc).mean(dim=0), color=color_RCCM, linewidth=2.5, label='NN-RCCM')
        plt.yscale('log')
        plt.plot(t, np.repeat(alpha, len(t)), 'k--', label='NN-RCCM Tube')
        plt.title(f"{args.task} ($\sigma=$ {args.sigma})")

    elif args.plot_type == "mean_std_plot_pos":
        plt.fill_between(t, torch.as_tensor(errors_CCMdp).mean(dim=0) - (1 * torch.as_tensor(errors_CCMdp).std(dim=0)), torch.as_tensor(errors_CCMdp).mean(dim=0)  + (1 * torch.tensor(errors_CCMdp).std(dim=0)), facecolor=(*color_CCM, 0.15), edgecolor=(0, 0, 0, 0))
        plt.plot(t, torch.as_tensor(errors_CCMdp).mean(dim=0),color=color_CCM, linewidth=2.5,label='NN-CCM')
        plt.fill_between(t, torch.as_tensor(errors_RCCMdp).mean(dim=0) - (1 * torch.as_tensor(errors_RCCMdp).std(dim=0)), torch.as_tensor(errors_RCCMdp).mean(dim=0) + (1 * torch.as_tensor(errors_RCCMdp).std(dim=0)), facecolor=(*color_RCCM, 0.15),edgecolor=(0, 0, 0, 0))
        plt.plot(t, torch.as_tensor(errors_RCCMdp).mean(dim=0), color=color_RCCM, linewidth=2.5,label='NN-RCCM-P')
        plt.plot(t, np.repeat(args.sigma * alpha, len(t)), 'k--', linewidth=2.5, label='NN-RCCM-P Tube')
        plt.yscale('log')
        plt.title(f"{args.task} ($\sigma=$ {args.sigma})")
        print("CCM AUC MEAN" + u"\u00B1" + "STD dev :", torch.as_tensor(auc_CCMp).mean().item(), torch.as_tensor(auc_CCMp).std().item())
        print("RCCM AUC MEAN" + u"\u00B1" + "STD dev :", torch.as_tensor(auc_RCCMp).mean().item(), torch.as_tensor(auc_RCCMp).std().item())

    if args.plot_type=='2D':
        for n_traj in range(args.nTraj):
            plt.plot([x[args.plot_dims_pos[0], 0] for x in x_star[n_traj]], [x[args.plot_dims_pos[1], 0] for x in x_star[n_traj]], 'r--', label='Reference' if n_traj == 0 else None) #xstar
            plt.plot(x_star[n_traj][0][args.plot_dims_pos[0]], x_star[n_traj][0][args.plot_dims_pos[1]], 'bo', markersize=3.)
            plt.xlabel("x")
            plt.ylabel("y")

    elif args.plot_type=='3D':
        for n_traj in range(args.nTraj):
            plt.plot([x[args.plot_dims_pos[0], 0] for x in x_star[n_traj]],
                     [x[args.plot_dims_pos[1], 0] for x in x_star[n_traj]],
                     [x[args.plot_dims_pos[2], 0] for x in x_star[n_traj]], 'r--', label='Reference'if n_traj == 0 else None)
            plt.plot(x_star[n_traj][0][args.plot_dims_pos[0]], x_star[n_traj][0][args.plot_dims_pos[1]],
                     x_star[n_traj][0][args.plot_dims_pos[2]], 'bo', markersize=3.)
            ax.set_ylabel('y')
            ax.set_zlabel('z')

    elif args.plot_type=='time':
        for plot_dim in args.plot_dims_pos:
            plt.plot(t, [x[plot_dim,0] for x in xstar][:-1], 'k')
        plt.xlabel("time(s)")
        plt.ylabel("x")
    elif args.plot_type=='error':
          plt.xlabel("time(s)")
          plt.ylabel('${||x_t - x_t^{*}||}_{2}$')
    elif args.plot_type=='error_pos':
          plt.xlabel("time(s)")
          plt.ylabel('${||p_t - p_t^{*}||}_{2}$')
    elif args.plot_type=='control_error':
        plt.xlabel("time(s)")
        plt.ylabel('${||u_t - u_t^{*}||}_{2}$')
    elif args.plot_type == "mean_std_plot":
        plt.xlabel("time(s)")
        plt.ylabel('${||x_t - x_t^{*}||}_{2}$')
    elif args.plot_type=='mean_std_plot_pos':
          plt.xlabel("time(s)")
          plt.ylabel('${||p_t - p_t^{*}||}_{2}/T$')
    elif args.plot_type == "mean_std_plot_control":
        plt.xlabel("time(s)")
        plt.ylabel('${||u_t - u_t^{*}||}_{2}$')
    elif args.plot_type == "augmented":
        plt.xlabel("time(s)")
        plt.ylabel('${||[x_t,u_t]- [x_t^{*},u_t^{*}]||}_{2}$')
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    plt.legend(frameon=True, loc='lower right',fancybox=False, edgecolor='black', borderaxespad=0.1, handlelength=1.25)
    #plt.savefig(args.task + '.pdf', bbox_inches='tight')
    plt.show()





