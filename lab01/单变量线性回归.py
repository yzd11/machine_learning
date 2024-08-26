import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读入数据
data = pd.read_csv("D:\\BaiduNetdiskDownload\\ML_NG\\01-linear regression\\ex1data1.txt",names = ['population','profit'])

# 数据插入
data.insert(0,'ones',1)

# 数据切割
X = data.iloc[:,0:-1]
Y = data.iloc[:,-1]

# 转换X, Y类型
X = X.values
Y = Y.values
Y = Y.reshape(len(Y),1)

# 定义损失函数
def costFunction(X, Y, theta):
    inner = np.power(X@theta - Y,2)
    return np.sum(inner) / (2 * len(X))

theta = np.zeros((2,1))
# 得到初识代价函数值
cost_init = costFunction(X,Y,theta)

print(cost_init)

# 定义梯度下降函数
def gradientDescent(X, Y, theta,alpha, num_iter):
    costs = []
    for i in range(num_iter):
        theta = theta - (X.T @ (X@theta - Y)) * alpha / len(X)
        cost = costFunction(X, Y, theta)
        costs.append(cost)
        # 判断，当100的整数次时输出当前的代价函数值
        if i % 100 == 0:
            print(cost)

    # 返回参数值和代价函数值，便于进行可视化展示
    return theta, costs

# 初始化一个 alpha 值
alpha = 0.01
# 初始化迭代次数
num_iter = 2000

theta,costs = gradientDescent(X,Y,theta,alpha,num_iter)
print(theta)

# 可视化
fig,ax = plt.subplots(2,1)
ax1 = ax[0]
ax2 = ax[1]
ax1.plot(np.arange(num_iter),costs)
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Cost')
ax1.set_title('Learning Curve')

x = np.linspace(X.min(),X.max(),100)
y = theta[0,0] + theta[1,0] * x


ax2.scatter(X[:,1],Y,label='Training Data')
ax2.plot(x,y,'r',label='predict')
ax2.legend(loc='best')

plt.show()
