# --------- Importing useful libraries ---------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# --------- Definition of useful functions ---------
def step(w, inputs):
    x1, x2 = inputs
    #f = w[0] + np.matmul(w[1], x1) + np.matmul(w[2], x2)
    f = w[0] + w[1]*x1 + w[2]*x2
    if f >= 0: return 1
    else: return 0

def perceptron_step(w, inputs, labels, eta):
    # function to implement the single step of the perceptron algorithm
    # compute outputs

    outputs = step(w, inputs)
    if outputs != labels:
        w = w + eta*inputs*(labels - outputs)



# --------- Point (1a) --------- #
# --------- random initialization of weights ---------

w0 = np.random.uniform(-1/4, 1/4)
w1 = np.random.uniform(-1,1)
w2 = np.random.uniform(-1,1)

# create the weights vector

w = np.array([w0, w1, w2]).reshape(3,1)

# --------- Point (1b) --------- #
# --------- generation of samples and labels---------

# samples

x0 = 1
x = np.random.uniform(-1,1, size = (100, 2))


# labels

labels = []
for i in range(x.shape[0]):
    y = step(w, x[i])
    labels.append(y)

# transform labels into an array

labels = np.array(labels)

# --------- Visualize Results ---------
x1_list = []
x2_list = []
colors = []
for i in range(x.shape[0]):
    x1 = x[i,0]
    x2 = x[i, 1]
    x1_list.append(x1)
    x2_list.append(x2)
    if labels[i] == 1: colors.append('red')
    else: colors.append('blue')

# create the line
x1_array = np.array(x1_list)
x2_array = -(w[0] + w[1] * x1_array) / w[2]

# create the scatterplot

plt.figure(figsize = (10, 10))
plt.scatter(x1_list, x2_list, c = colors)
plt.plot(x1_array, x2_array, linewidth=1.5, color='forestgreen', label='Decision Boundary')
plt.title('Scatterplot with randomly initialized weights')
plt.xlabel(r'$x_{1}$')
plt.ylabel(r'$x_{2}$')
plt.show()
#plt.savefig('initial_scatterplot.png') # save figure

# --------- Point (1c) ---------#
# --------- Find the normal vector to the line ---------

print(type(-w[2].item())) # to extract a float element from the array

# define a vector tangent to the line

v_tangent = np.array([-w[2].item(), w[1].item()])
v_normal = np.array([w[1].item(), w[2].item()])

prod = np.dot(v_normal, v_tangent)
print(prod) # verify the the scalar product is 0!

# --------- Compute the distance between the point and the line ---------

# define line parameters

a = -w[1].item() / w[2].item()
#a = -w[1]/w[2]
print(type(a))
b = -1
c = -w[0].item() / w[2].item()

# define the point of interest 

xp = 0
yp = 0

d = np.abs(a*xp + b*yp + c) / math.sqrt(math.pow(a,2) + math.pow(b, 2))

# --------- Point (2a) ---------#
# Implementation of the perceptron algorithm --------- #

w_init = np.array([1, 1, 1])