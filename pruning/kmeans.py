import torch
import numpy as np
import pdb

def pairwise_distance(data1, data2=None, device=-1):
    r'''
    using broadcast mechanism to calculate pairwise ecludian distance of data
    the input data is N*M matrix, where M is the dimension
    we first expand the N*M matrix into N*1*M matrix A and 1*N*M matrix B
    then a simple elementwise operation of A and B will handle the pairwise operation of points represented by data
    '''
    if data2 is None:
        data2 = data1 

    if device!=-1:
        data1, data2 = data1.cuda(device), data2.cuda(device)

    #N*1*M
    A = data1.unsqueeze(dim=1)

    #1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A-B)**2.0
    #return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1).squeeze()
    return dis

def group_pairwise(X, groups, device=0, fun=lambda r,c: pairwise_distance(r, c).cpu()):
    group_dict = {}
    for group_index_r, group_r in enumerate(groups):
        for group_index_c, group_c in enumerate(groups):
            R, C = X[group_r], X[group_c]
            if device!=-1:
                R = R.cuda(device)
                C = C.cuda(device)
            group_dict[(group_index_r, group_index_c)] = fun(R, C)
    return group_dict

def forgy(X, n_clusters):
    _len = len(X)
    indices = np.random.choice(_len, n_clusters)
    initial_state = X[indices]
    return initial_state

def lin_spaced(X, n_clusters):
    min_ = torch.min(X)
    max_ = torch.max(X)
    space = torch.tensor(np.linspace(min_.cpu(), max_.cpu(), num=n_clusters), dtype=X.dtype, device=X.device)
    return space.unsqueeze(dim=1)

def lloyd(X, n_clusters, device=0, tol=1e-4):
    initial_state = lin_spaced(X, n_clusters)

    while True:
        dis = pairwise_distance(X, initial_state)

        choice_cluster = torch.argmin(dis, dim=0)
        initial_state_pre = initial_state.clone()

        for index in range(n_clusters):
            selected = torch.nonzero(choice_cluster==index).squeeze()
            selected = torch.index_select(X, 0, selected)

            if selected.nelement() != 0:
                initial_state[index] = selected.mean(dim=0)

        center_shift = torch.sum(torch.sqrt(torch.sum((initial_state - initial_state_pre) ** 2, dim=1)))
        if center_shift ** 2 < tol:
            break

    return choice_cluster, initial_state