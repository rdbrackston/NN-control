import torch
from copy import deepcopy

def euler_self(f,y0,tf):
    t = 0
    h =0.01
    y = deepcopy(y0)
    traj =[]
    while t <= tf:
        traj.append(deepcopy(y))
#         print(y)
        dy = f.forward(0, y)
        t += h
        y += h * dy
    return torch.stack(traj)


def euler_self_tlist(f,y0,tlist,step):
    t = 0
    h = step
    y = 1*y0
    traj =[]
    
    t_ind = 0
    while t < tlist[-1]:
        dy = f.forward(0, y)
        t += h
        y = y.clone() + h * dy
        
        if t >= tlist[t_ind]:
            traj.append(1*y)
            t_ind += 1
    return torch.stack(traj)