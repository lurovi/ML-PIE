#from config.setting import *
import numpy as np
import torch

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
    print(torch.cat((a, b), dim=0))
