import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读入数据
data = pd.read_csv("ex1data2.txt",names = ['size','bedrooms','price'])

# 特征缩放
def normalize_feature(data):
    # 使用 Z-score normalization
    return (data - data.mean()) / data.std()

data = normalize_feature(data)

# 添加全为 1 的列
data.insert(0,'ones',1)

# 构造数据集
X = data.iloc[:,:-1].values
Y = data.iloc[:,-1].values
Y = Y.reshape(len(Y),1)

# 定义损失函数
def costFunction(X, Y, theta):
    inner = np.power(X@theta - Y,2)
    return np.sum(inner) / (2 * len(X))

theta = np.zeros((3,1))
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
alpha = [0.0001, 0.0003, 0.0005, 0.007]
# 初始化迭代次数
num_iter = 2000

# 可视化
fig,ax = plt.subplots()

# 不同学习率下的曲线情况
for alpha in alpha:
 theta, costs = gradientDescent(X, Y, theta, alpha, num_iter)
 ax.plot(np.arange(num_iter), costs,label='alpha='+str(alpha))
 ax.legend()

ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Learning Curve')

plt.show()