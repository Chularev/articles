import gc
import numpy as np
import torch


def test_me(rows,columns):
    #for i in range(100):
    M1 = np.random.rand(rows, columns)
    M2 = np.random.rand(columns, rows)
    M3 = M1 @ M2

    M_1 = torch.from_numpy(M1)
    M_2 = torch.from_numpy(M2)
    M_1.requires_grad = False
    M_2.requires_grad = False
    M_3 = M_1 @ M_2

    cuda = torch.device('cuda:0')
    M1_CUDA = M_1.to(cuda)
    M2_CUDA = M_2.to(cuda)

    M3_CUDA = M1_CUDA @ M2_CUDA
    cpu = torch.device('cpu')
    del M1_CUDA
    del M2_CUDA
    del M3_CUDA
    with torch.cuda.device('cuda:0'):
        torch.cuda.synchronize(device='cuda:0')
        torch.cuda.empty_cache()