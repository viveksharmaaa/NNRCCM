# Learning Tube Certified Control Using Robust Contraction Metric


## Install package dependencies
Dependencies include ```torch```, ```tqdm```, ```numpy```, and ```matplotlib```. You can install them using the following command.
```bash
pip install -r requirements.txt
```

## Usage
The script ```main.py``` can be used for learning the controller. Usage of this script is as follows
```
usage: main.py [-h] [--task TASK] [--bs BS]
               [--num_train NUM_TRAIN] [--num_test NUM_TEST]
               [--lr LEARNING_RATE] [--epochs EPOCHS] [--lr_step LR_STEP]
               [--lambda _LAMBDA] [--w_lb W_LB] [--log LOG]
               [--sigma _SIGMA]

optional arguments:
  -h, --help            show this help message and exit
  --task TASK           Name of the model.
  --bs BS               Batch size.
  --num_train NUM_TRAIN
                        Number of samples for training.
  --num_test NUM_TEST   Number of samples for testing.
  --lr LEARNING_RATE    Base learning rate.
  --epochs EPOCHS       Number of training epochs.
  --lr_step LR_STEP
  --lambda _LAMBDA      Convergence rate: lambda
  --w_lb W_LB           Lower bound of the eigenvalue of the dual metric.
  --log LOG             Path to a directory for storing the log.
  --sigma _SIGMA        Disturbance bound
```


## Jointly learn the controller and RCCM

Run the following command to learn a robust control contraction metric (RCCM), controller while minimizing gain for PVTOL model.
```
mkdir log_PVTOL
python main.py --log log_PVTOL --task PVTOL --sigma 1
python main.py --log log_QUADROTOR_9D --task QUADROTOR_9D  --sigma 1
python main.py --log log_NL --task NL --sigma 1
python main.py --log log_TLPRA --task TLPRA --sigma 1 --lambda 2
```
Please note that for generating tube size for variables of your choice for the system PVTOL, you may modify the matrices ```C,D,C_ref,D_ref``` in``` systems\system_PVTOL.py```
## Comparisons 
Run the following command to reproduce plot comparison results between NN-RCCM-P (ours) and NN-CCM.
```
python plot.py --pretrained_CCM log_PVTOL/controller_best_CCM.pth.tar --pretrained_RCCM log_PVTOL/controller_best_ref.pth.tar --task PVTOL --plot_type mean_std_plot_pos --sigma 1 --plot_dims_pos 0 1 --nTraj 100
python plot.py --pretrained_CCM log_QUADROTOR_9D/controller_best_CCM.pth.tar --pretrained_RCCM log_QUADROTOR_9D/controller_best_ref.pth.tar --task QUADROTOR_9D --plot_type mean_std_plot_pos --sigma 1 --plot_dims_pos 0 1 2 --nTraj 100
python plot.py --pretrained_CCM log_NL/controller_best_CCM.pth.tar --pretrained_RCCM log_NL/controller_best_ref.pth.tar --task NL --plot_type mean_std_plot_pos --sigma 1 --plot_dims_pos 0 1 2 --nTraj 100
python plot.py --pretrained_CCM log_TLPRA/controller_best_CCM.pth.tar --pretrained_RCCM log_TLPRA/controller_best_ref.pth.tar --task TLPRA --plot_type mean_std_plot_pos --sigma 1 --plot_dims_pos 0 1 --nTraj 100
```
For demonstration of tube-based safe motion planning from Section 4.2.2 in the paper, run the following file from the self-contained folder ```motion-planning``` in MATLAB 2022b

````
plot_compare_trajs.m
````

The motion planner uses trajectory-optimization toolbox taken from [OptimTraj](https://github.com/MatthewPeterKelly/OptimTraj)
Part of the code taken from [C3M](https://github.com/sundw2014/C3M) 

