import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

# 定义正规方程
def normalEquation(X, Y):
    theta = np.linalg.inv(X.T@X)@X.T@Y
    return theta

theta = normalEquation(X, Y)

print(theta)

# 可视化
fig,ax = plt.subplots()

x = np.linspace(X.min(),X.max(),100)
y = theta[0,0] + theta[1,0] * x

ax.scatter(X[:,1],Y,label='Training Data')
ax.plot(x,y,'r',label='predict')
ax.legend(loc='best')

plt.show()