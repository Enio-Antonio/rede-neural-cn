import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la

dados_1 = np.loadtxt('../dados_01.dat')
dados_2 = np.loadtxt('../dados_02.dat')

x = dados_1[:, 0] # x do gráfico
u = dados_1[:,1] # dados pra testar depois que o y tiver pronto
y = dados_1[:,2] # dados para fazer a função de treinamento

x2 = dados_2[:, 0]
u2 = dados_2[:, 1]
y2 = dados_2[:, 2]

plt.figure(figsize=(14,4))
plt.plot(x, u, 'r')
plt.plot(x, y)
plt.show()

plt.figure(figsize=(12,6))
plt.plot(x2, y2)
plt.show()