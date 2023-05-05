# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as n
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as n

def ridge(data):
    X,y=read_data()
    z=n.matmul(X.T,X)+n.eye(X.shape[1])*(0.00000000000000001)
    weight=n.matmul(n.linalg.inv(z),n.matmul(X.T,y))
    return weight @ data
    
def lasso(data):
    X,y=read_data()
    m, k = X.shape
    weight=n.zeros(k)
    max_iterations = 100000
    for i in range(max_iterations):
       grad = (n.matmul(X.T, (n.matmul(X, weight) - y))) + 30 * n.sign(weight)
       weight = weight - 1e-12 * grad
       if n.linalg.norm(grad) < 0.0001:
           break
    return weight @ data

def read_data(path='./data/exp02/'):
    x = n.load(path + 'X_train.npy')
    y = n.load(path + 'y_train.npy')
    return x, y
