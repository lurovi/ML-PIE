#from util.setting import *
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from torch import nn

from exps.ExpsExecutor import ExpsExecutor

from util.Sort import Sort


def softmax_stable(x):
    return(np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum())


if __name__ == "__main__":
    l = [2, 2, 2, 2, 1, 1, 1]
    print(softmax_stable(l))
    exit(1)
    scaler = MinMaxScaler()
    a = np.random.uniform(-5, 5, (100, 5))
    scaler.fit(a)
    l = np.array([34, 23, 21, -33, -12]).reshape(1, -1)
    l = scaler.transform(l)[0]
    print(l)

    a = torch.tensor([3, 4, 51, 3, 2, 8]).repeat(3, 1).float()
    print(a)
    print(nn.Dropout(0.25)(a))
    print(nn.Dropout(0.25)(a))
    print(a)
    b = torch.tensor([[1,2,3,4,5,6], [3,2,1,4,5,6], [5,4,3,6,2,1]]).float()
    l = ((a - b)**2).sum(axis=1).tolist()
    print(l, np.max(l))
    print(torch.tensor(1))
    print(torch.tensor([0.24, 0.25, 0.26], dtype=torch.float32).reshape(-1, 1))
    exit(1)
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
    arr, ind = Sort.heapsort(arr, lambda x, y: x.sum().item() < y.sum().item(), inplace=False, reverse=True)
    print(arr)
    print(ind)
    print(torch.tensor([[1],[2],[3]]).flatten())
    print(torch.tensor(4).reshape(-1))

    print(torch.tensor([[23, 24, 17], [124, 53, 2], [78, 123, 45], [12, 32, 55]]).data.argmax(dim=1).float().tolist())
    print(confusion_matrix([1.0,1.0,-1.0,1,-1.0,1.0,-1.0],[-1.0,1,-1,1.0,-1,1.0,1]))
    print(ExpsExecutor.merge_dictionaries_of_list([{"a":[1],"b":[1]},{"a":[2],"b":[2]},{"a":[3],"b":[3]},{"a":[4],"b":[4]}]))
    #ExpsExecutor.plot_line({"a":[1,2,3]*4,"b":[34,12,36,23,37,45,23,15,46,12,23,45],"c":["a","a","b","b"]*3,"d":["a","b","a","b"]*3}, "a", "b", "c", "d")
