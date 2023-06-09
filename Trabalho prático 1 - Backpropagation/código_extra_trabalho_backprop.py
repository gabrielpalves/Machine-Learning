# -*- coding: utf-8 -*-
"""Código extra - trabalho backprop

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XTtZGgpAefbiWejTrEjsnWzS_XXYdzff
"""

###############
##### Plotando fronteira de decisão não-linear
###############


import numpy as np
import matplotlib.pyplot as plt

# Plotando fronteira de decisão
x1s = np.linspace(-1,1.5,50)
x2s = np.linspace(-1,1.5,50)
z=np.zeros((len(x1s),len(x2s)))

#y = h(x) = 1/(1+exp(- z))
#z = theta.T * x

for i in range(len(x1s)):
    for j in range(len(x2s)):
        x = np.array([x1s[i], x2s[j]]).reshape(2,-1)
        z[i,j] = net_z_output( x )  # saida do modelo antes de aplicar a função sigmoide - substituir aqui teu código
plt.contour(x1s,x2s,z.T,0)
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend(loc=0)

###############
##### Classificação binária com modelo de rede neural - backprop / regressão logística
###############

import pandas as pd
df=pd.read_csv("data/classification2.txt", header=None)

X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
pos , neg = (y==1).reshape(118,1) , (y==0).reshape(118,1)
plt.scatter(X[pos[:,0],0],X[pos[:,0],1],c="r",marker="+")
plt.scatter(X[neg[:,0],0],X[neg[:,0],1],marker="o",s=10)
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend(["Accepted","Rejected"],loc=0)

###############
##### Classificação de dígitos com modelo de rede neural - backprop
###############

from scipy.io import loadmat
mat=loadmat("data/classification3.mat")
X=mat["X"]
y=mat["y"]

import matplotlib.image as mpimg
fig, axis = plt.subplots(10,10,figsize=(8,8))
for i in range(10):
    for j in range(10):
        axis[i,j].imshow(X[np.random.randint(0,5001),:].reshape(20,20,order="F"), cmap="hot") #reshape back to 20 pixel by 20 pixel
        axis[i,j].axis("off")

