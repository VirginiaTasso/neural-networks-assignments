# --------- Importing useful libraries ---------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# --------- Definition of useful functions ---------
def step(w, inputs):
    x0, x1, x2 = inputs
    #f = w[0] + np.matmul(w[1], x1) + np.matmul(w[2], x2)
    f = w[0]*x0 + w[1]*x1 + w[2]*x2
    if f >= 0: return 1
    else: return 0

def perceptron_step(w, inputs, label, eta, errors):
    # function to implement the single step of the perceptron algorithm
    # compute outputs

    output = step(w, inputs)

    if output != label:
        w = w + eta * inputs.reshape(-1, 1) * (label - output)

        errors += 1 # increment the number of errors  
        return w, errors # return the updated weights
    
    # if there are no errors simply return the current weights
    else: 
        return w, errors


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
x0 = np.ones((100, 1))
X = np.hstack((x0, x)) 


# labels

labels = []
for i in range(X.shape[0]):
    y = step(w, X[i])
    labels.append(y)

# transform labels into an array

labels = np.array(labels)

# --------- Visualize Results ---------
x1_list = []
x2_list = []
colors = []
for i in range(X.shape[0]):
    x1 = X[i,1]
    x2 = X[i, 2]
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

w_init = np.array([1, 1, 1]).reshape(3,1)
flag = 1 # flag to stop the algorithm when there are no more errors
errors_tot = 0 # total error committed
errors_epoch = 0 # errors committed in the current epoch

eta = 1 # regularization param
epochs = 0
# add the constant term to all inputs x1, x2
print(f'Labels shape : {labels.shape}')

while(flag):
    errors_epoch = 0 # reset errors of the current epocj
    epochs += 1
    for i in range(X.shape[0]):
        print(f'current input shape: {X[i].shape}')
        w_update, errors_tot = perceptron_step(w_init, X[i], labels[i], errors_tot, eta)
        if not np.array_equal(w_init, w_update):
            # an error was committed
            errors_epoch += 1
            w_init = w_update
    if errors_epoch == 0:
        # no errors were committed in the entire epoch --> stop the algorithm
        flag = 0

print(f'Loop ended with a total of {errors_tot} errors, in {epochs} epochs ')
