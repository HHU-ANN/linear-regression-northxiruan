# 最终在main函数中传入一个维度为6的numpy数组，输出预测值
import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np
#最小二乘岭回归
def ridge(data):
    X,y=read_data() #读入数据
    z=np.matmul(X.T,X)+np.eye(X.shape[1])*(0.000000000000000000000000000001)
    weight=np.matmul(np.linalg.inv(z),np.matmul(X.T,y)) #最小二乘岭回归公式
    return weight @ data
#梯度下降Lasso回归 
def lasso(data):
    X,y=read_data()
    m, k = X.shape
    weight=np.zeros(k) #初始化
    max_iterations = 100000 #次数
    for i in range(max_iterations):
       grad = (np.matmul(X.T, (np.matmul(X, weight) - y))) + 30 * np.sign(weight) #梯度引入公式
       weight = weight - 1e-11 * grad
        #学习率1e-11
       if np.linalg.norm(grad) < 0.0001:#边界
           break
    return weight @ data

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
