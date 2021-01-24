## data 
## Pt	BP	Age	Weight	BSA	Dur	Pulse	Stress
## 1	105	47	85.4	1.75	5.1	63	33
## 2	115	49	94.2	2.10	3.8	70	14
## 3	116	49	95.3	1.98	8.2	72	10
## 4	117	50	94.7	2.01	5.8	73	99
## 5	112	51	89.4	1.89	7.0	72	95
## 6	121	48	99.5	2.25	9.3	71	10
## 7	121	49	99.8	2.25	2.5	69	42
## 8	110	47	90.9	1.90	6.2	66	8
## 9	110	49	89.2	1.83	7.1	69	62
## 10	114	48	92.7	2.07	5.6	64	35
## 11	114	47	94.4	2.07	5.3	74	90
## 12	115	49	94.1	1.98	5.6	71	21
## 13	114	50	91.6	2.05	10.2	68	47
## 14	106	45	87.1	1.92	5.6	67	80
## 15	125	52	101.3	2.19	10.0	76	98
## 16	114	46	94.5	1.98	7.4	69	95
## 17	106	46	87.0	1.87	3.6	62	18
## 18	113	46	94.5	1.90	4.3	70	12
## 19	110	48	90.5	1.88	9.0	71	99
## 20	122	56	95.7	2.09	7.0	75	99

import numpy as np

## X: matrix of examples. In this example, there are 20 examples (20 patients). 
## y: matrix of labels. In this example, there are 20 labels.
## theta: matrix of parameters. In this example, there are 6 parameters (Age, Weight, BSA, Dur, Pluse, Stress)
## plus theta_0 which is unrehularized, in total, there are 7 parameters.

## In this example, X is a 20 by 7 matrix and theta is a 7 by 1 matrix and y is a 20 by 1 matrix.


X_origin = np.array([
	[47, 85.4,  1.75, 5.1,  63, 33],
	[49, 94.2,  2.10, 3.8,  70, 14], 
	[49, 95.3,  1.98, 8.2,  72, 10],
	[50, 94.7,  2.01, 5.8,  73, 99],
	[51, 89.4,  1.89, 7.0,  72, 95],
	[48, 99.5,  2.25, 9.3,  71, 10],
	[49, 99.8,  2.25, 2.5,  69, 42],
	[47, 90.9,  1.90, 6.2,  66, 8 ],
	[49, 89.2,  1.83, 7.1,  69, 62],
	[48, 92.7,  2.07, 5.6,  64, 35],
	[47, 94.4,  2.07, 5.3,  74, 90],
	[49, 94.1,  1.98, 5.6,  71, 21],
	[50, 91.6,  2.05, 10.2, 68, 47],
	[45, 87.1,  1.92, 5.6,  67, 80],
	[52, 101.3, 2.19, 10.0, 76, 98],
	[46, 94.5,  1.98, 7.4,  69, 95],
	[46, 87.0,  1.87, 3.6,  62, 18],
	[46, 94.5,  1.90, 4.3,  70, 12],
	[48, 90.5,  1.88, 9.0,  71, 99],
	[56, 95.7,  2.09, 7.0,  75, 99]
])

y = np.array([	
[105],
[115],
[116],
[117],
[112],
[121],
[121],
[110],
[110],
[114],
[114],
[115],
[114],
[106],
[125],
[114],
[106],
[113],
[110],
[122]
])



m = X_origin.shape[0]
n = X_origin.shape[1]

columnof1 = np.ones((m, 1))

X = np.concatenate((columnof1, X_origin), 1)
theta = np.ones((n+1, 1))


X_scaled = columnof1.copy()

for i in range(1, n+1):
  mean = sum(X[:, i]) / m
  std = np.std(X[:, i])
  X_scaled = np.concatenate((X_scaled, ((X[:, i] - mean) / std).reshape(m, 1)), 1 )


def h(theta, X):
  return np.matmul(X, theta)

def costfunc(theta, X, y, factor): ## factor is the regularization factor lambda
  dif = h(theta, X) - y
  J = np.matmul(dif.transpose(), dif)
  J = J + factor * np.matmul(theta[1:, :].transpose(), theta[1:, :])
  J = J / (2*m)

  return J

def graddescent(theta, X, y, factor, alpha): ## alpha is the learning rate
  regu = theta * alpha * factor / m
  regu[0,0] = 0
  newtheta = theta - np.matmul(X.transpose(), (h(theta, X) - y)) *alpha / m - regu
  return newtheta
#  newtheta = theta - alpha * gradcheck(theta, X, y, factor)

#def gradcheck(theta, X, y, factor): 
#  print('grad descent')
#  regu = theta * factor / m
#  regu[0,0] = 0
#  print(np.matmul(X.transpose(), (h(theta, X) - y)) / m + regu)
#  print('num dif')
#  numgrad = np.zeros((n+1, 1))
#  epsilon = np.zeros((n+1, 1))
#  for i in range(n+1):
#    epsilon[i][0] = 0.002
#    numgrad[i][0] = ((costfunc(theta+epsilon, X, y, factor) - costfunc(theta-epsilon, X, y, factor)) / (0.002 * 2))[0,0]
#  return numgrad


factor = 5
alpha = 0.05
iternum = 1000
#yy = []
for i in range(iternum):
#  print('iteration:', i+1)
  new = graddescent(theta, X_scaled, y, factor, alpha)
  theta = new.copy()
#  yy.append(costfunc(theta, X_scaled, y, factor)[0,0])

#import matplotlib.pyplot as plt
#plt.plot(range(iternum), yy)
#plt.show()

print(theta)




### benchmark: normal equation ##
R = np.identity(n+1)
R[0,0] = 0
result = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_scaled.transpose(), X_scaled) + factor * R), X_scaled.transpose()),y)
print(result)




