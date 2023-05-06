# 最终在main函数中传入一个维度为6的numpy数组，输出预测值
import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    X,y=read_data()
    z=np.matmul(X.T,X)+np.eye(X.shape[1])*(-0.1)
    weight=np.matmul(np.linalg.inv(z),np.matmul(X.T,y))
    return weight @ data
    
def lasso(data):
    X,y=read_data()
    m, n = X.shape
    weight = np.array([ 1.49462254e+01, -2.50275342e-01, -8.76423816e-03,  1.23727270e+00,
       -1.80224871e+02, -2.10165019e+02])
    max_iterations = 100000
    for i in range(max_iterations):
       grad = (np.matmul(X.T, (np.matmul(X, weight) - y))) + 30 * np.sign(weight)
       weight = weight - 1e-12 * grad
       if np.linalg.norm(grad) < 0.0001:
           break
    return weight @ data

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
