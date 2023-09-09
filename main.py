import torch
from torch.autograd import grad

import importlib
import numpy as np
import time
from tqdm import tqdm
import argparse


import sys
sys.path.append('systems')
sys.path.append('configs')
sys.path.append('models')

np.random.seed(1024)

parser = argparse.ArgumentParser(description="")
parser.add_argument('--task', type=str,default='CAR', help='Name of the model.')
parser.add_argument('--bs', type=int, default=1024, help='Batch size.')
parser.add_argument('--num_train', type=int, default=131072, help='Number of samples for training.')
parser.add_argument('--num_test', type=int, default=32768, help='Number of samples for testing.')
parser.add_argument('--lr', dest='learning_rate', type=float, default=0.001, help='Base learning rate.')
parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs.')
parser.add_argument('--lr_step', type=int, default=5, help='')
parser.add_argument('--lambda', type=float, dest='_lambda', default=0.5, help='Convergence rate: lambda')
parser.add_argument('--w_lb', type=float, default=0.1, help='Lower bound of the eigenvalue of the dual metric.')
parser.add_argument('--log', type=str, help='Path to a directory for storing the log.')
parser.add_argument('--sigma', type=float, default=1, help='Upper bound of the disturbance $w(t)$.')

args = parser.parse_args()
epsilon = args._lambda * 0.1

# Randomly initialize alpha and mu
r1 = torch.rand([],requires_grad=True)
r2 = torch.rand([],requires_grad=True)
if r1 > r2:
    alpha = r2
    mu = r1
else :
    alpha = r1
    mu = r2

config = importlib.import_module('config_'+args.task)
X_MIN = config.X_MIN
X_MAX = config.X_MAX
U_MIN = config.UREF_MIN
U_MAX = config.UREF_MAX
XE_MIN = config.XE_MIN
XE_MAX = config.XE_MAX

system = importlib.import_module('system_'+args.task)
f_func = system.f_func
B_func = system.B_func
Bw_func = system.Bw_func
C,D,C_ref,D_ref = system.DgDxu()
num_dim_x = system.num_dim_x
num_dim_control = system.num_dim_control
num_dim_distb = system.num_dim_distb
if hasattr(system, 'Bbot_func'):
    Bbot_func = system.Bbot_func

model = importlib.import_module('model_'+args.task)
get_model = model.get_model

model_W, model_Wbot, model_u_w1, model_u_w2, W_func, u_func = get_model(num_dim_x, num_dim_control, w_lb=args.w_lb)

# constructing datasets
def sample_xef():
    return (X_MAX-X_MIN) * np.random.rand(num_dim_x, 1) + X_MIN

def sample_x(xref):
    xe = (XE_MAX-XE_MIN) * np.random.rand(num_dim_x, 1) + XE_MIN
    x = xref + xe
    x[x>X_MAX] = X_MAX[x>X_MAX]
    x[x<X_MIN] = X_MIN[x<X_MIN]
    return x

def sample_uref():
    return (U_MAX-U_MIN) * np.random.rand(num_dim_control, 1) + U_MIN

def sample_full():
    xref = sample_xef()
    uref = sample_uref()
    x = sample_x(xref)
    return (x, xref, uref)

X_tr = [sample_full() for _ in range(args.num_train)]
X_te = [sample_full() for _ in range(args.num_test)]

if 'Bbot_func' not in locals():
    def Bbot_func(x): # columns of Bbot forms a basis of the null space of B^T
        bs = x.shape[0]
        Bbot = torch.cat((torch.eye(num_dim_x-num_dim_control, num_dim_x-num_dim_control),
            torch.zeros(num_dim_control, num_dim_x-num_dim_control)), dim=0)
        Bbot.unsqueeze(0)
        return Bbot.repeat(bs, 1, 1)

def Jacobian_Matrix(M, x):
    # NOTE that this function assume that data are independent of each other
    # along the batch dimension.
    # M: B x m x m
    # x: B x n x 1
    # ret: B x m x m x n
    bs = x.shape[0]
    m = M.size(-1)
    n = x.size(1)
    J = torch.zeros(bs, m, m, n).type(x.type())
    for i in range(m):
        for j in range(m):
            J[:, i, j, :] = grad(M[:, i, j].sum(), x, create_graph=True)[0].squeeze(-1)
    return J

def Jacobian(f, x):
    # NOTE that this function assume that data are independent of each other
    f = f + 0. * x.sum() # to avoid the case that f is independent of x
    # f: B x m x 1
    # x: B x n x 1
    # ret: B x m x n
    bs = x.shape[0]
    m = f.size(1)
    n = x.size(1)
    J = torch.zeros(bs, m, n).type(x.type())
    for i in range(m):
        J[:, i, :] = grad(f[:, i, 0].sum(), x, create_graph=True)[0].squeeze(-1)
    return J

def weighted_gradients(W, v, x, detach=False):
    # v, x: bs x n x 1
    # DWDx: bs x n x n x n
    assert v.size() == x.size()
    bs = x.shape[0]
    if detach:
        return (Jacobian_Matrix(W, x).detach() * v.view(bs, 1, 1, -1)).sum(dim=3)
    else:
        return (Jacobian_Matrix(W, x) * v.view(bs, 1, 1, -1)).sum(dim=3)

def loss_neg(a):
    if a >=0:
        return 0.0
    else:
        return -10*a

K = 1024
def loss_pos_matrix_random_sampling(A):
    # A: bs x d x d
    # z: K x d
    z = torch.randn(K, A.size(-1))
    z = z / z.norm(dim=1, keepdim=True)
    zTAz = (z.matmul(A) * z.view(1,K,-1)).sum(dim=2).view(-1)
    negative_index = zTAz.detach().cpu().numpy() < 0
    if negative_index.sum()>0:
        negative_zTAz = zTAz[negative_index]
        return -1.0 * (negative_zTAz.mean())
    else:
        return torch.tensor(0.).type(z.type()).requires_grad_()

def loss_pos_matrix_eigen_values(A):
    # A: bs x d x d
    eigv = torch.symeig(A, eigenvectors=True)[0].view(-1)
    negative_index = eigv.detach().cpu().numpy() < 0
    negative_eigv = eigv[negative_index]
    return negative_eigv.norm()

def forward(x, xref, uref, _lambda, verbose=False, acc=False, detach=False, refine = False):
    # x: bs x n x 1
    bs = x.shape[0]
    W = W_func(x)
    M = torch.inverse(W)
    f = f_func(x)
    B = B_func(x)
    Bw = Bw_func(x)
    DfDx = Jacobian(f, x)
    DBDx = torch.zeros(bs, num_dim_x, num_dim_x, num_dim_control).type(x.type())
    for i in range(num_dim_control):
        DBDx[:,:,:,i] = Jacobian(B[:,:,i].unsqueeze(-1), x)

    DBwDx = torch.zeros(bs, num_dim_x, num_dim_x, num_dim_distb).type(x.type())
    for i in range(num_dim_distb):
        DBwDx[:, :, :, i] = Jacobian(Bw[:, :, i].unsqueeze(-1), x)

    _Bbot = Bbot_func(x)
    u = u_func(x, x - xref, uref) # u: bs x m x 1
    K = Jacobian(u, x)
    w = torch.randn(bs, num_dim_distb)
    pred = (w.norm(dim=1, keepdim=True) > 1)
    w[pred.squeeze(-1), :] = w[pred.squeeze(-1), :] / w[pred.squeeze(-1), :].norm(dim=1, keepdim=True)
    w = args.sigma * w.unsqueeze(-1)
    A = DfDx + sum([u[:, i, 0].unsqueeze(-1).unsqueeze(-1) * DBDx[:, :, :, i] for i in range(num_dim_control)]) + sum([w[:,i,0].unsqueeze(-1).unsqueeze(-1) * DBwDx[:, :, :, i] for i in range(num_dim_distb)])

    if refine:
        si = C_ref.repeat(bs, 1, 1) + D_ref.repeat(bs, 1, 1).matmul(K)  # added
    else:
        si = C.repeat(bs, 1, 1) + D.repeat(bs, 1, 1).matmul(K)

    dot_x = f + B.matmul(u) + Bw.matmul(w)
    dot_M = weighted_gradients(M, dot_x, x, detach=detach) # DMDt
    dot_W = weighted_gradients(W, dot_x, x, detach=detach) # DWDt
    if detach:
        Contraction = dot_M + (A + B.matmul(K)).transpose(1,2).matmul(M.detach()) + M.detach().matmul(A + B.matmul(K)) + _lambda * M.detach()
        # Schur complements of Condition 1 (Eq.10) and Condition 2 (Eq.10) from the paper
        Cond1 = Contraction + 1/mu * M.detach().matmul(Bw).matmul(Bw.transpose(1, 2).matmul(M.detach()))
        Cond2 = _lambda * M.detach() - 1/alpha.detach() * si.transpose(1, 2).matmul(si)
    else:
        Contraction = dot_M + (A + B.matmul(K)).transpose(1,2).matmul(M) + M.matmul(A + B.matmul(K)) + _lambda * M
        # Schur complements of Condition 1 (Eq.10) and Condition 2 (Eq.10) from the Lemma
        Cond1 = Contraction + 1/mu* M.matmul(Bw).matmul(Bw.transpose(1, 2).matmul(M))
        Cond2 = _lambda * M - 1/alpha * si.transpose(1, 2).matmul(si)

    # C1
    C1_inner = - weighted_gradients(W, f, x) + DfDx.matmul(W) + W.matmul(DfDx.transpose(1,2)) + _lambda * W
    C1_LHS_1 = _Bbot.transpose(1,2).matmul(C1_inner).matmul(_Bbot) # this has to be a negative definite matrix

    # C2
    C2_inners = []
    C2s = []
    for j in range(num_dim_control):
        C2_inner = weighted_gradients(W, B[:,:,j].unsqueeze(-1), x) - (DBDx[:,:,:,j].matmul(W) + W.matmul(DBDx[:,:,:,j].transpose(1,2)))
        C2 = _Bbot.transpose(1,2).matmul(C2_inner).matmul(_Bbot)
        C2_inners.append(C2_inner)
        C2s.append(C2)

    loss = 0
    loss += loss_pos_matrix_random_sampling(-Cond1)
    loss += loss_pos_matrix_random_sampling(Cond2)
    if detach:
        loss += alpha.detach()
    else:
        loss += alpha

    loss += loss_pos_matrix_random_sampling((alpha - mu)* torch.eye(num_dim_distb))

    loss += loss_pos_matrix_random_sampling(-C1_LHS_1)

    loss += 1. * sum([1.*(C2**2).reshape(bs,-1).sum(dim=1).mean() for C2 in C2s])
    if verbose:
        print(torch.linalg.eigh(Contraction,UPLO = 'U')[0].min(dim=1)[0].mean(), torch.linalg.eigh(Contraction,UPLO = 'U')[0].max(dim=1)[0].mean(), torch.linalg.eigh(Contraction,UPLO = 'U')[0].mean())
    if acc:
        return loss, loss_pos_matrix_random_sampling(-Contraction).item(), loss_pos_matrix_random_sampling(-C1_LHS_1).item(),sum([1.*(C2**2).reshape(bs,-1).sum(dim=1).mean() for C2 in C2s]).item(),loss_pos_matrix_random_sampling(-Cond1).item(),loss_pos_matrix_random_sampling(Cond2).item()
    else:
        return loss, None, None, None, None, None

optimizer = torch.optim.Adam(list(model_W.parameters()) + list(model_Wbot.parameters()) + list(model_u_w1.parameters()) + list(model_u_w2.parameters()) + [alpha] + [mu], lr=args.learning_rate)
optimizer1 = torch.optim.SGD([mu] + [alpha] , lr=0.01)

def trainval(X, bs=args.bs, train=True, _lambda=args._lambda, acc=False, detach=False, refine=False): # trainval a set of x

    if train:
        indices = np.random.permutation(len(X))
    else:
        indices = np.array(list(range(len(X))))

    total_loss = 0
    total_p1 = 0
    total_p2 = 0
    total_l3 = 0
    total_p3 = 0
    total_p4 = 0


    if train:
        _iter = tqdm(range(len(X) // bs))
    else:
        _iter = range(len(X) // bs)
    for b in _iter:
        start = time.time()
        x = []; xref = []; uref = [];
        for id in indices[b*bs:(b+1)*bs]:
                x.append(torch.from_numpy(X[id][0]).float())
                xref.append(torch.from_numpy(X[id][1]).float())
                uref.append(torch.from_numpy(X[id][2]).float())

        x, xref, uref = (torch.stack(d).detach() for d in (x, xref, uref))
        x = x.requires_grad_()

        start = time.time()

        loss, p1, p2, l3, p3, p4 = forward(x, xref, uref, _lambda=_lambda, verbose=False if not train else False, acc=acc, detach=detach, refine = refine)

        start = time.time()
        if train and not refine:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            alpha.clamp(min=0)
            mu.clamp(min=0)
        elif train and refine:
            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()
            alpha.clamp(min=0)
            mu.clamp(min=0)

        total_loss += loss.item() * x.shape[0]

        if acc:
            total_p1 += p1 * x.shape[0]
            total_p2 += p2 * x.shape[0]
            total_p3 += p3 * x.shape[0]
            total_p4 += p4 * x.shape[0]
            total_l3 += l3 * x.shape[0]
    return total_loss / len(X), total_p1/ len(X) , total_p2/ len(X) , total_l3/ len(X), total_p3/ len(X),total_p4/ len(X)

best_acc = 0

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by every args.lr_step epochs"""
    lr = args.learning_rate * (0.1 ** (epoch // args.lr_step))
    for param_group in optimizer.param_groups:
       param_group['lr'] = lr

for epoch in range(args.epochs):
    loss, _, _, _, _,_= trainval(X_tr, train=True, _lambda=args._lambda, acc=False, detach= True if epoch < args.lr_step else False, refine = False)
    print("Training loss: ", loss)
    print("alpha/mu",alpha.item(),mu.item())
    loss, p1, p2, l3, p3, p4 = trainval(X_te, train=False, _lambda=0., acc=True, detach=False,refine = False)
    print("Epoch %d: Testing loss/Contraction/Lw1/KV/Cond1/Cond2: "%epoch, loss, p1, p2, l3, p3, p4)
    if p2+p3+p4 >= best_acc:
        filename = args.log+'/model_best.pth.tar'
        filename_controller = args.log+'/controller_best.pth.tar'
        torch.save({'args':args, 'precs':(loss, p1, p2), 'model_W': model_W.state_dict(), 'model_Wbot': model_Wbot.state_dict(), 'model_u_w1': model_u_w1.state_dict(), 'model_u_w2': model_u_w2.state_dict(),'alpha':alpha.item(),'mu':mu.item()}, filename)
        torch.save(u_func, filename_controller)
    if epoch == args.epochs - 1:
        print("Tube size (alpha) is:", alpha.item())

for epoch in range(10):
    print("REFININING TUBE SIZE FOR SELECTED VARIABLES")
    loss, _, _, _, _,_= trainval(X_tr, train=True, _lambda=args._lambda, acc=False, detach= False,refine = True)
    print("Epoch %d Refined alpha/mu:"%epoch, alpha.item(),mu.item())
    loss, p1, p2, l3, p3, p4 = trainval(X_te, train=False, _lambda=0., acc=True, detach=False,refine = True)
    if p2+p3+p4 >= best_acc:
        filename = args.log+'/model_best_ref.pth.tar'
        filename_controller = args.log+'/controller_best_ref.pth.tar'
        torch.save({'args':args, 'precs':(loss, p1, p2), 'model_W': model_W.state_dict(), 'model_Wbot': model_Wbot.state_dict(), 'model_u_w1': model_u_w1.state_dict(), 'model_u_w2': model_u_w2.state_dict(),'alpha_ref':alpha.item(),'mu_ref':mu.item()}, filename)
        torch.save(u_func, filename_controller)
    if epoch == args.epochs - 1:
        print("Refined tube size (alpha) for selected variable is:", alpha.item())