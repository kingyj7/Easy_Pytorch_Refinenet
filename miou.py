#Reference: https://github.com/dilligencer-zrj/code_zoo/blob/master/compute_mIOU
#the result is the same with https://github.com/DrSleep/light-weight-refinenet/tree/master/src/miou.pyx

# *.-code: uft-8.*

import numpy as np

#设标签宽W，长H
def fast_hist(a, b, n):#a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的预测特征图，形状(H×W,)；n是类别数目
    #k = (a > 0) & (a <= n) #k是一个一维bool数组，形状(H×W,)；目的是找出标签中需要计算的类别（去掉了背景）,假设0是背景
    k = a <= n
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):#分别为每个类别（在这里是19类）计算mIoU，hist的形状(n, n)
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))#矩阵的对角线上的值组成的一维数组/矩阵的所有元素之和，返回值形状(n,)

def per_class_recall(hist):
    return np.diag(hist) / (hist.sum(0)) #tp/gt
