import numpy as np
from math import e

def softmax(x):
  denominator = np.sum(np.exp(x))
  return np.exp(x) / denominator

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def rnn_cell_forward(xt, a_prev, parameters):
  Wax = parameters['Wax']
  Waa = parameters["Waa"]
  Wya = parameters["Wya"]
  ba = parameters["ba"]
  by = parameters["by"]

  a_next = np.tanh(Waa.dot(a_prev) + Wax.dot(xt) + ba)
  yt_pred = softmax(Wya.dot(a_next) + by)
#  a_next = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, xt) + ba)
#  yt_pred = softmax(np.dot(Wya, a_next) + by)
  cache = (a_next, a_prev, xt, parameters)
  return a_next, yt_pred, cache

def rnn_forward(x, a0, parameters):
  caches = []
  n_x, m, T_x = x.shape
  n_y, n_a = parameters['Wya'].shape
  a = np.zeros([n_a, m, T_x])
  y_pred = np.zeros([n_y, m, T_x])
  a_next = a0
  for t in range(T_x):
    a_next, yt_pred, cache = rnn_cell_forward(x[:, :, t], a_next, parameters)
    a[:, :, t] = a_next
    y_pred[:, :, t] = yt_pred
    caches.append(cache)

  caches = (caches, x)
  return a, y_pred, caches


### LSTM ###
def lstm_cell_forward(xt, a_prev, c_prev, parameters):
  Wf = parameters["Wf"]
  bf = parameters["bf"]
  Wi = parameters["Wi"]
  bi = parameters["bi"]
  Wc = parameters["Wc"]
  bc = parameters["bc"]
  Wo = parameters["Wo"]
  bo = parameters["bo"]
  Wy = parameters["Wy"]
  by = parameters["by"]

  n_x, m = xt.shape
  n_y, n_a = Wy.shape

  concat = np.random.randn(n_a + n_x, m)
  concat[: n_a, :] = a_prev
  concat[n_a: , :] = xt

  ft = sigmoid(Wf.dot(concat)+bf)
  it = sigmoid(Wi.dot(concat)+bi)
  cct = np.tanh(Wc.dot(concat) + bc)
  c_next = ft * c_prev + it*cct
  ot = sigmoid(Wo.dot(concat) + bo)
  a_next = ot * np.tanh(c_next)
  
  yt_pred = softmax(Wy.dot(a_next) + by)
  cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

  return a_next, c_next, yt_pred, cache


def lstm_forward(x, a0, parameters):
  caches = []
  n_x, m, T_x = x.shape
  Wy = parameters["Wy"]
  n_y, n_a = Wy.shape
  a = np.zeros((n_a, m, T_x))
  c = np.zeros((n_a, m, T_x))
  y = np.zeros((n_y, m, T_x))
  a_next = a0
  c_next = np.zeros((n_a, m))
  for t in range(T_x):
    a_next, c_next, yt, cache = lstm_cell_forward(x[:, :, t], a_next, c_next, parameters)
    a[:, :, t] = a_next
    y[:, :, t] = yt
    c[:, :, t] = c_next
    caches.append(cache)

  caches = (caches, x)

  return a, y, c, caches

def rnn_cell_backward(da_next, cache):
  (a_next, a_prev, xt, parameters) = cache
  Wax = parameters["Wax"]
  Waa = parameters["Waa"]
  Wya = parameters["Wya"]
  ba = parameters["ba"]
  by = parameters["by"]
  dtanh = (1 - np.square(a_next)) * da_next
  dxt = (Wax.T).dot(dtanh)
  dWax = dtanh.dot(xt.T)
  da_prev = (Waa.T).dot(dtanh)
  dWaa = dtanh.dot(a_prev.T)
  dba = np.sum(dtanh, axis=1, keepdims=True)
  gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba}

  return gradients


def rnn_backward(da, caches):
  (caches, x) = caches
  (a1, a0, x1, parameters) = caches[1]
  n_a, m, T_x = da.shape
  n_x, m = x1.shape
  
  dx = np.zeros((n_x, m, T_x), dtype=np.float)
  dWax = np.zeros((n_a, n_x), dtype=np.float)
  dWaa = np.zeros((n_a, n_a), dtype=np.float)
  dba = np.zeros((n_a, 1), dtype=np.float)
  da0 = np.zeros((n_a, m), dtype=np.float)
  da_prevt = np.zeros((n_a, m), dtype=np.float)

  for t in reversed(range(T_x)):
    gradients = rnn_cell_backward(da[:, :, t] + da_prevt, caches[t])
    dxt, da_prevt, dWaxt, dWaat, dbat = gradients['dxt'], gradients['da_prev'], gradients['dWax'], gradients['dWaa'], gradients['dba']
    dx[:, :, t] = dxt
    dWax += dWaxt
    dWaa += dWaat
    dba += dbat
  
  da0 = da_prevt
  gradients = {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa, "dba": dba}
  return gradients

def lstm_cell_backward(da_next, dc_next, cache):
    (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache
    
    n_x, m = xt.shape 
    n_a, m = a_next.shape 
    
    dit = (da_next * ot * (1 - np.tanh(c_next) ** 2) + dc_next) * cct * (1 - it) * it
    dft = (da_next * ot * (1 - np.tanh(c_next) ** 2) + dc_next) * c_prev * ft * (1 - ft)
    dot = da_next * np.tanh(c_next) * ot * (1 - ot)
    dcct = (da_next * ot * (1 - np.tanh(c_next) ** 2) + dc_next) * it * (1 - cct ** 2)

    dWf = np.dot(dft,np.concatenate((a_prev, xt), axis=0).T) # or use np.dot(dft, np.hstack([a_prev.T, xt.T]))
    dWi = np.dot(dit,np.concatenate((a_prev, xt), axis=0).T)
    dWc = np.dot(dcct,np.concatenate((a_prev, xt), axis=0).T)
    dWo = np.dot(dot,np.concatenate((a_prev, xt), axis=0).T)
    dbf = np.sum(dft,axis=1,keepdims=True)
    dbi = np.sum(dit,axis=1,keepdims=True) 
    dbc = np.sum(dcct,axis=1,keepdims=True) 
    dbo = np.sum(dot,axis=1,keepdims=True)  

    da_prev = np.dot(parameters['Wf'][:,:n_a].T,dft)+np.dot(parameters['Wi'][:,:n_a].T,dit)+np.dot(parameters['Wc'][:,:n_a].T,dcct)+np.dot(parameters['Wo'][:,:n_a].T,dot) 
    dc_prev = dc_next*ft+ot*(1-np.square(np.tanh(c_next)))*ft*da_next 
    dxt = np.dot(parameters['Wf'][:,n_a:].T,dft)+np.dot(parameters['Wi'][:,n_a:].T,dit)+np.dot(parameters['Wc'][:,n_a:].T,dcct)+np.dot(parameters['Wo'][:,n_a:].T,dot) 
    
    gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
                "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}

    return gradients

#  (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache
#  n_x, m = xt.shape
#  n_a, m = a_next.shape
#  concat = np.vstack((a_prev, xt))
#
#  dit = (da_next * ot * (1 - np.tanh(c_next) ** 2) + dc_next) * cct * (1 - it) * it
#  dft = (da_next * ot * (1 - np.tanh(c_next) ** 2) + dc_next) * c_prev * ft * (1 - ft)
#  dot = da_next * np.tanh(c_next) * ot * (1 - ot)
#  dcct = (da_next * ot * (1 - np.tanh(c_next) ** 2) + dc_next) * it * (1 - cct ** 2)
#
#
#  dWf = np.dot(dft, concat.T)
#  dWi = np.dot(dit, concat.T)
#  dWc = np.dot(dcct, concat.T)
#  dWo = np.dot(dot, concat.T)
#  dbf = np.sum(dft, axis=1, keepdims = True)
#  dbi = np.sum(dit, axis=1, keepdims = True)
#  dbc = np.sum(dcct, axis=1, keepdims = True)
#  dbo = np.sum(dot, axis=1, keepdims = True)
#
#  da_prev = np.dot(parameters['Wf'][:, : n_a].T, dft) + np.dot(parameters['Wi'][:, : n_a].T, dit) + np.dot(parameters['Wc'][:, : n_a].T, dcct) + np.dot(parameters['Wo'][:, : n_a].T, dot)
#  dc_prev = (dc_next * ft) + (ot * (1 - np.square(np.tanh(c_next))) * ft * da_next)
#  dxt = np.dot(parameters['Wf'][:, n_a:, ].T, dft) + np.dot(parameters['Wi'][:, n_a:].T, dit) + np.dot(parameters['Wc'][:, n_a:].T, dcct) + np.dot(parameters['Wo'][:, n_a : ].T, dot)
#  gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
#                "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}
#  return gradients


def lstm_backward(da, caches):
  (caches, x) = caches
  (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = caches[0]
  n_a, m, T_x = da.shape
  n_x, m = x1.shape
    
  dx = np.zeros((n_x, m, T_x))
  da0 = np.zeros((n_a, m))

  da_prevt = np.zeros((n_a, m))
  dc_prevt = np.zeros((n_a, m))

  dWf = np.zeros((n_a, n_a + n_x))
  dWi = np.zeros((n_a, n_a + n_x))
  dWc = np.zeros((n_a, n_a + n_x))
  dWo = np.zeros((n_a, n_a + n_x))

  dbf = np.zeros((n_a, 1))
  dbi = np.zeros((n_a, 1))
  dbc = np.zeros((n_a, 1))
  dbo = np.zeros((n_a, 1))
    
  for t in reversed(range(T_x)):
    gradients = lstm_cell_backward(da[:,:,t] + da_prevt, dc_prevt, caches[t])
    dx[:,:,t] = gradients["dxt"]
    dWf += gradients["dWf"]
    dWi += gradients["dWi"]
    dWc += gradients["dWc"]
    dWo += gradients["dWo"]
    dbf += gradients["dbf"]
    dbi += gradients["dbi"]
    dbc += gradients["dbc"]
    dbo += gradients["dbo"]   


  da0 = gradients["da_prev"]
  gradients = {"dx": dx, "da0": da0, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
                "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}

  return gradients


#############

np.random.seed(1)
x_tmp = np.random.randn(3,10,7)
a0_tmp = np.random.randn(5,10)

parameters_tmp = {}
parameters_tmp['Wf'] = np.random.randn(5, 5+3)
parameters_tmp['bf'] = np.random.randn(5,1)
parameters_tmp['Wi'] = np.random.randn(5, 5+3)
parameters_tmp['bi'] = np.random.randn(5,1)
parameters_tmp['Wo'] = np.random.randn(5, 5+3)
parameters_tmp['bo'] = np.random.randn(5,1)
parameters_tmp['Wc'] = np.random.randn(5, 5+3)
parameters_tmp['bc'] = np.random.randn(5,1)
parameters_tmp['Wy'] = np.random.randn(2,5)
parameters_tmp['by'] = np.random.randn(2,1)

a_tmp, y_tmp, c_tmp, caches_tmp = lstm_forward(x_tmp, a0_tmp, parameters_tmp)

da_tmp = np.random.randn(5, 10, 4)
gradients_tmp = lstm_backward(da_tmp, caches_tmp)

print("gradients[\"dx\"][1][2] =", gradients_tmp["dx"][1][2])
print("gradients[\"dx\"].shape =", gradients_tmp["dx"].shape)
print("gradients[\"da0\"][2][3] =", gradients_tmp["da0"][2][3])
print("gradients[\"da0\"].shape =", gradients_tmp["da0"].shape)
print("gradients[\"dWf\"][3][1] =", gradients_tmp["dWf"][3][1])
print("gradients[\"dWf\"].shape =", gradients_tmp["dWf"].shape)
print("gradients[\"dWi\"][1][2] =", gradients_tmp["dWi"][1][2])
print("gradients[\"dWi\"].shape =", gradients_tmp["dWi"].shape)
print("gradients[\"dWc\"][3][1] =", gradients_tmp["dWc"][3][1])
print("gradients[\"dWc\"].shape =", gradients_tmp["dWc"].shape)
print("gradients[\"dWo\"][1][2] =", gradients_tmp["dWo"][1][2])
print("gradients[\"dWo\"].shape =", gradients_tmp["dWo"].shape)
print("gradients[\"dbf\"][4] =", gradients_tmp["dbf"][4])
print("gradients[\"dbf\"].shape =", gradients_tmp["dbf"].shape)
print("gradients[\"dbi\"][4] =", gradients_tmp["dbi"][4])
print("gradients[\"dbi\"].shape =", gradients_tmp["dbi"].shape)
print("gradients[\"dbc\"][4] =", gradients_tmp["dbc"][4])
print("gradients[\"dbc\"].shape =", gradients_tmp["dbc"].shape)
print("gradients[\"dbo\"][4] =", gradients_tmp["dbo"][4])
print("gradients[\"dbo\"].shape =", gradients_tmp["dbo"].shape)

