import numpy as np
import matplotlib.pyplot as plt

thresh = 0.6225

num = 20
r = 9
m = 120


g = open('data.txt', 'w')
for i in range(m):
  a = np.zeros((num, num))
  centerx =  np.random.randint(num / 4, 3*num/4)
  centery =  np.random.randint(num / 4, 3*num/4)
#  centerx =  num / 2
#  centery =  num / 2
  for i in range(a.shape[0]):
    for j in range(a.shape[1]):
      if (i-centerx)**2 + (j-centery)**2 < r**2:
        a[i,j] = 1
 
 
  
  for i in range(a.shape[0]):
    for j in range(a.shape[1]):
      g.write('{0:d} '.format(int(a[i,j])))

#  plt.imshow(a)
#  plt.show()

  percent = sum(sum(a)) / num**2
  if percent < thresh:
    g.write('0\n')
  elif percent == thresh:
    g.write('1\n')
  else:
    print('error', percent)

  
g.close()


#for i in range(100):
#  a = np.zeros()
