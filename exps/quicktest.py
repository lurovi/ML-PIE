#from util.setting import *
import numpy as np
import torch
from torch import nn

from gp.tree import Primitive
from util.Sort import heapsort

if __name__ == "__main__":
    a = np.array([0.50, 0.0, 0.0, 0.20, 0.05])

    print(np.all(a == 0.0))

    print(a[np.array([0, 2, 3])])
    print(a[5:])

    print(np.array([]))

    s = "ytuu"+""+"gdfgdf"+"\n"+"hyht"+"\t\t"+"def"+"dsf\t\t"+"abc"+"\n"
    print(s.find("d"))

    t = type(256.24)
    print(t("234"))

    print(type(torch.tensor([2,3,4])))

    print("abc\n")
    print("abcc")

    a = torch.tensor([[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10]], dtype=torch.float32)
    print(a)
    print(a.shape[1]//2)
    print(a[:, :a.shape[1] // 2])
    print(a[:, a.shape[1] // 2: ])

    a = torch.tensor([1,2,3,4,5])
    b = torch.tensor([6,7,8,9,10])
    print((torch.cat((a, b), dim=0).float().reshape(1, -1)[:, 2]*torch.cat((a, b), dim=0).float().reshape(1, -1)[:, 3]).sum())
    print(torch.sum( torch.tensor([[3]]) * (torch.tensor([[5]])-torch.tensor([[2]])) ))
    print(torch.tensor([1,2]).tolist() + torch.tensor([3,4]).tolist())
    print(nn.Softmax(dim=0)(torch.tensor([23,45]).float()))
    print(np.concatenate((np.array([1,2,3]),np.array([4,5,6])), axis=None).reshape(1, -1))
    print("######################")

    arr = [torch.tensor([2,3,4]),
           torch.tensor([1,5,6]),
           torch.tensor([10,2,4]),
           torch.tensor([2,1,1]),
           torch.tensor([2,40,4]),
           torch.tensor([5,5,1])]
    arr0 = [12, 11, 13, 5, 6, 7]
    arr, ind = heapsort(arr, lambda x, y: x.sum().item() < y.sum().item(), inplace=False, reverse=True)
    print(arr)
    print(ind)

