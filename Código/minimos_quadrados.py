import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la

dados_1 = np.loadtxt('../dados_01.dat')
dados_2 = np.loadtxt('../dados_02.dat')

x = dados_1[:, 0] # x do gráfico
y = dados_1[:,2] # dados para fazer a função de treinamento

y2 = dados_2[:,2]

Y = np.array([np.concatenate((np.zeros(2), y[0:-2])),
              np.concatenate((np.zeros(1), y[0:-1])),
              x,
              np.concatenate((np.zeros(1), x[0:-1]))]).T

w = la.pinv(Y)@y

y_est =w[0]*Y[:,0] + w[1]*Y[:,1] + w[2]*Y[:,2] + w[3]*Y[:,3]

EMQ = np.mean((y - y_est)**2)
print(f"O EMQ é: {EMQ}")

plt.figure(figsize=(14,6))
plt.plot(x, y)
plt.plot(x, y_est, '--r')
plt.show()

Y2 = np.array([np.concatenate((np.zeros(2), y2[0:-2])),
              np.concatenate((np.zeros(1), y2[0:-1])),
              x,
              np.concatenate((np.zeros(1), x[0:-1]))]).T

y_est_test = w[0]*Y2[:,0] + w[1]*Y2[:,1] + w[2]*Y2[:,2] + w[3]*Y2[:,3]

EMQ = np.mean((y2 - y_est_test)**2)
print(f"O EMQ é: {EMQ}")

plt.figure(figsize=(14,6))
plt.plot(x, y2)
plt.plot(x, y_est_test, '--r')
plt.show()