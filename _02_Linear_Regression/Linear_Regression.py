# 最终在main函数中传入一个维度为6的numpy数组，输出预测值
import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np
#最小二乘岭回归
def ridge(data):
    X,y=read_data()
    
    weight=np.matmul(np.linalg.inv(z),np.matmul(X.T,y))
    return weight @ data
    #梯度下降Lasso回归 
def lasso(data):
    X,y=read_data()
    m, k = X.shape
    weight=np.zeros(k)
    max_iterations = 100000
    for i in range(max_iterations):
       grad = (np.matmul(X.T, (np.matmul(X, weight) - y))) + 30 * np.sign(weight)
       weight = weight - 1e-11 * grad
        #学习率1e-11
       if np.linalg.norm(grad) < 0.0001:
           break
    return weight @ data

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
