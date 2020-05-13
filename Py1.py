import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def inputdata():
    from sklearn.datasets import load_boston
    boston = load_boston()
    bostonX = boston.data    
    lastsix = bostonX[:,7:13]
    
    return lastsix

def pca_topvar(n_components, data):
    var = np.array([0,0])
    for i in range(0, 50):
        pca = PCA(n_components)
        pca.fit(data[i*10:i*10+10,:])
        varrat = pca.explained_variance_ratio_
        var = np.vstack((var, varrat))
    var = var[1:,:]
    return var


def main():
   
    print('五一快乐')
    bostondata = inputdata()
    var_ratio = pca_topvar(2,bostondata)
    time = np.arange(0,50)
    pca1 = var_ratio[:,0]
    pca12 = pca1 + var_ratio[:,1]
    plt.xlabel('time')
    plt.ylabel('var')
    plt.title('BostonPCA top 2')
    plt.plot(time, pca1 , 'b')
    plt.plot(time, pca12, 'r')
    plt.savefig("outfile-pca.png")
    plt.show()
    
   
if __name__ == '__main__':
    main()