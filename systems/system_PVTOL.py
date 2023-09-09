import torch

num_dim_x = 6
num_dim_control = 2
num_dim_distb = 1

m = 0.486;
J = 0.00383;
g = 9.81;
l = 0.25;

def f_func(x):
    # x: bs x n x 1
    # f: bs x n x 1
    bs = x.shape[0]

    p_x, p_z, phi, v_x, v_z, dot_phi = [x[:,i,0] for i in range(num_dim_x)]
    f = torch.zeros(bs, num_dim_x, 1).type(x.type())
    f[:, 0, 0] = v_x * torch.cos(phi) - v_z * torch.sin(phi)
    f[:, 1, 0] = v_x * torch.sin(phi) + v_z * torch.cos(phi)
    f[:, 2, 0] = dot_phi
    f[:, 3, 0] = v_z * dot_phi - g * torch.sin(phi)
    f[:, 4, 0] = - v_x * dot_phi - g * torch.cos(phi)
    f[:, 5, 0] = 0
    return f

def DfDx_func(x):
    # x: bs x n x 1
    # f: bs x n x 1
    # ret: bs x n x n
    bs = x.shape[0]

    p_x, p_z, phi, v_x, v_z, dot_phi = [x[:,i,0] for i in range(num_dim_x)]
    J = torch.zeros(bs, num_dim_x, num_dim_x).type(x.type())
    J[:, 0, 2] = - v_x * torch.sin(phi) - v_z * torch.cos(phi)
    J[:, 1, 2] = v_x * torch.cos(phi) - v_z * torch.sin(phi)
    J[:, 2, 2] = 0
    J[:, 3, 2] = - g * torch.cos(phi)
    J[:, 4, 2] = g * torch.sin(phi)
    J[:, 5, 2] = 0
    J[:, 0, 3] = torch.cos(phi)
    J[:, 1, 3] = torch.sin(phi)
    J[:, 2, 3] = 0
    J[:, 3, 3] = 0
    J[:, 4, 3] = - dot_phi
    J[:, 5, 3] = 0
    J[:, 0, 4] = - torch.sin(phi)
    J[:, 1, 4] = torch.cos(phi)
    J[:, 2, 4] = 0
    J[:, 3, 4] = dot_phi
    J[:, 4, 4] = 0
    J[:, 5, 4] = 0
    J[:, 0, 5] = 0
    J[:, 1, 5] = 0
    J[:, 2, 5] = 1
    J[:, 3, 5] = v_z
    J[:, 4, 5] = - v_x
    J[:, 5, 5] = 0

    return J

def B_func(x):
    bs = x.shape[0]
    B = torch.zeros(bs, num_dim_x, num_dim_control).type(x.type())

    B[:, 4, 0] = 1 / m
    B[:, 4, 1] = 1 / m
    B[:, 5, 0] = l / J
    B[:, 5, 1] = -l / J
    return B

def Bw_func(x):
    bs = x.shape[0]
    Bw = torch.zeros(bs, num_dim_x, num_dim_distb).type(x.type())

    p_x, p_z, phi, v_x, v_z, dot_phi = [x[:,i,0] for i in range(num_dim_x)]

    Bw[:, 3, 0] = torch.cos(phi)
    Bw[:, 4, 0] = - torch.sin(phi)

    return Bw

def DBDx_func(x):
    # B: bs x n x m
    # ret: bs x n x n x m
    bs = x.shape[0]
    return torch.zeros(bs, num_dim_x, num_dim_x, num_dim_control).type(x.type())

def DgDxu():
    #Pick C and D as per Section 3.1 in the manuscript
    #Pick C_ref and D_ref as per Section 3.2 in the manuscript

    # All States and Control Variables
    # C = torch.cat((torch.eye(num_dim_x),torch.zeros(num_dim_control,num_dim_x)))
    # D = torch.cat((torch.zeros(num_dim_x,num_dim_control),1 * torch.diag(torch.tensor([1,1]))))

    # Only Position State Variables
    C_ref = torch.cat((torch.eye(2),torch.zeros(2,num_dim_x-2)),1)
    D_ref = torch.zeros(2,num_dim_control)

    # All State variables
    C = torch.eye(num_dim_x)
    D = torch.zeros(num_dim_x,num_dim_control)

    # Position and Control Variables
    # C = torch.cat((torch.cat((torch.eye(num_dim_control),torch.zeros(num_dim_control,4)),dim=1),torch.zeros(2,num_dim_x)),dim=0)
    # D = torch.cat((torch.zeros(2,num_dim_control),torch.eye(num_dim_control)),dim=0)

    # Only Control variables
    # C_ref = torch.zeros(num_dim_control,num_dim_x)
    # D_ref = torch.eye(num_dim_control)

    return C,D,C_ref,D_ref
