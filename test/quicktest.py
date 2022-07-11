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
