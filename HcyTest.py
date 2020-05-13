import numpy as np
import numpy.matlib as nm
import pandas as pd
from sklearn.decomposition import  PCA

def loaddata():
    b = np.loadtxt('f:\\python\\boston_house_prices.csv',skiprows=2,delimiter=',',unpack=False,dtype=float)
    #把标题行去掉
    #b = b[1:]
    
    #取后6列数据，作为新矩阵
    b = b[...,-6:]

    #计算每个维度（即每列）的均值
    r = np.mean(b,axis=0)

    #把样本每个维度的值减去该维度的均值，得到每个样本新矩阵，每个维度的均值为0
    c = np.subtract(b,r)

    #把新样本矩阵进行转置，再乘以新矩阵，除以N-1 得到协方差
    d = np.transpose(c)
    cov = np.dot(c,d)


    print(c)
    print(d)
    print(cov)


    #随机生成空矩阵
    #c = np.matlib.rand((2,3))
    #print(c)

def main():
    loaddata()
    MatrixA = np.random.random((3,5))
    print(MatrixA)
    print(MatrixA.min())

if __name__ == '__main__':
    main()
        