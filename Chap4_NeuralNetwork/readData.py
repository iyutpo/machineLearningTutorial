import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

f = open('data.txt')
f1 = f.readlines()
f.close()

X = []
y = []
for i in range(len(f1)):
  X.append([float(j) for j in f1[i].strip('\n').split()[:-1]])
  y.append(float(f1[i].strip('\n').split()[-1]))
#  temp = np.array([float(j) for j in f1[i].strip('\n').split()[:-1]]).reshape((20, 20))
#  plt.imshow(temp)
#  plt.show()

X = np.array(X)
y = np.array(y).reshape((-1, 1))

print(X.shape)
print(y.shape)

