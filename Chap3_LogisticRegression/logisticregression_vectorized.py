from numpy import random
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

def readdata(filename):
  f = open(filename)
  f1 = f.readlines()
  f.close()
  X = []
  y = []
  for i in range(1, len(f1)):
    X.append(list(map(float, f1[i].strip('\n').split()[1:3])))
    if 'passed' in f1[i]:
      y.append([1.0])
    elif 'failed' in f1[i]:
      y.append([0.0])
    else:
      print(f1[i], 'error')

  return np.array(X), np.array(y)

# example X, label y, parameter theta
X, y = readdata('score_logisticregression.txt')
m = X.shape[0]
n = X.shape[1]
X_origin = X.copy()

X = np.concatenate((np.ones((m, 1)), X), axis=1)

# feature scaling 
X[:, 1] = (X[:, 1] - np.mean(X[:, 1])) / (max(X[:, 1]) - min(X[:, 1]))
X[:, 2] = (X[:, 2] - np.mean(X[:, 2])) / (max(X[:, 2]) - min(X[:, 2]))

# the hypothesis 
def h(theta, X):
  theta = theta.reshape(n+1, 1)
  predict = (1 / (np.exp(-1 * np.matmul(X, theta)) + 1))
  return predict

# the cost function
def costfunc(theta, X, y):
  return ((np.matmul(y.T, np.log(h(theta, X))) + np.matmul((1-y).T, np.log(1-h(theta, X)))) / (-m))[0,0]


theta = np.array([0, 0, 0])
from scipy.optimize import minimize 
res = minimize(fun=costfunc,
                       x0=theta,
                       args=(X, y),
                       method='TNC')

theta_opt = res.x.copy()

## scale the parameters back 
theta_opt[0] = theta_opt[0] - theta_opt[1] * np.mean(X_origin[:, 0]) / (max(X_origin[:, 0]) - min(X_origin[:, 0])) - theta_opt[2] * np.mean(X_origin[:, 1]) / (max(X_origin[:, 1]) - min(X_origin[:, 1]))
theta_opt[1] = theta_opt[1] / (max(X_origin[:, 0]) - min(X_origin[:, 0]))
theta_opt[2] = theta_opt[2] / (max(X_origin[:, 1]) - min(X_origin[:, 1]))

################################
# visualize the training result
################################
def boundary(x1, theta_opt):
  return (-theta_opt[0] - theta_opt[1] * x1) / theta_opt[2]

for i in range(m):
  if y[i, 0] == 1.0:
    plt.scatter(X_origin[i, 0], X_origin[i, 1], color='red')
  elif y[i, 0] == 0.0:
    plt.scatter(X_origin[i, 0], X_origin[i, 1], color='black')


linex = [min(X_origin[:, 0]), max(X_origin[:, 0])]
liney = [boundary(min(X_origin[:, 0]), theta_opt), boundary(max(X_origin[:, 0]), theta_opt)]
plt.plot(linex, liney)
plt.show()






#######################################
# generate data for training the model#
#######################################
#m = 100
#X = random.randint(100, size=(m, 2))
#y = np.zeros((m, 1))
#for i in range(m):
#  if sqrt(np.matmul(X[i], X[i].transpose())) > 65:
#    y[i, 0] = 1
#
#f = open('score.txt', 'w')
#f.write('Subject#    Score1    Score2    Status\n')
#for i in range(m):
#  if y[i,0] == 1:
#    f.write('    {0:3d}    {1:6d}    {2:6d}     passed\n'.format(i+1, X[i,0], X[i,1]))
#  elif y[i,0] == 0:
#    f.write('    {0:3d}    {1:6d}    {2:6d}     failed\n'.format(i+1, X[i,0], X[i,1]))
#

