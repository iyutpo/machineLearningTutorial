import numpy as np
x = np.array([[68],
       [60],
       [51],
       [43],
       [31]])

y = np.array([[ 37.49],
       [ 36.46],
       [ 67.28],
       [ 93.75],
       [140.22]])

#m,b = np.polyfit(x[:, 0], y[:, 0], 1)
#print(m,b)

x = ((x - np.mean(x)) / (max(x) - min(x)))
y = ((y - np.mean(y)) / (max(y) - min(y)))

#X = np.concatenate((x, np.ones((x.shape[0], 1))), 1)
#print(X)
#
#theta = np.ones((2, 1))
#for i in range(500):
#    dif = (np.matmul(X, theta) - y)
#    temp = (theta - 0.01 * np.matmul(np.transpose(X), dif))
#    theta[0][0], theta[1][0] = temp[0][0], temp[1][0]
#
#import matplotlib.pyplot as plt
#
#plt.scatter(x, y)
#plt.scatter(x, np.matmul(X, theta))
#plt.show()
#

#theta = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(xx), xx)), np.transpose(xx)), y)

def costfunc(theta1, theta0):
    Z = (x[0][0] * theta1 + theta0 - y[0][0])**2.0 + (x[1][0] * theta1 + theta0 - y[1][0])**2.0 + (x[2][0] * theta1 + theta0 - y[2][0])**2.0 + (x[3][0] * theta1 + theta0 - y[3][0])**2.0 + (x[4][0] * theta1 + theta0 - y[4][0])**2.0
    return Z / 10
#    return Z / 10 + 2 * theta1**2
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
#ax = plt.axes(projection='3d')
plotx = np.linspace(-60, 60, 2000)
ploty = np.linspace(-60, 60, 2000)
theta1, theta0 = np.meshgrid(plotx, ploty)

Z = costfunc(theta1, theta0)
print(Z.shape)
plt.contour(theta1, theta0, Z, cmap='jet')
#ax.contour3D(theta1, theta0, Z, 100, cmap='jet')
#ax.set_xlabel('theta1')
#ax.set_ylabel('theta0')
#ax.set_zlabel('cost');
plt.show()


