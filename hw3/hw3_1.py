# --------- Importing useful libraries ---------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# --------- Definition of useful functions ---------
def step(w, inputs):
    x0, x1, x2 = inputs
    f = w[0]*x0 + w[1]*x1 + w[2]*x2
    if f >= 0: return 1
    else: return 0

def perceptron_step(w, inputs, label, eta):
    # function to implement the single step of the perceptron algorithm
    # compute outputs

    output = step(w, inputs)

    if output != label:
        w += eta * inputs.reshape(-1, 1) * (label - output)

    return w, output # return the updated weights

#def visualize_results()
    


# ========= Point (1a) ========= #
# --------- random initialization of weights ---------

w0 = np.random.uniform(-1/4, 1/4)
w1 = np.random.uniform(-1,1)
w2 = np.random.uniform(-1,1)

# create the weights vector

w_star = np.array([w0, w1, w2]).reshape(3,1)
print(f"True Weights: {w_star} ")
# ========= Point (1b) ========= #
# --------- generation of samples and labels---------

# samples

x = np.random.uniform(-1,1, size = (100, 2))
x0 = np.ones((100,1))
X = np.hstack((x0, x)) 

print(f'The Inputs shape is: {X.shape}')
# labels

labels = []
for i in range(X.shape[0]):
    y = step(w_star, X[i])
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

# create the decision boundary line with the initial weights

x1_array = np.array(x1_list)
x2_array_0 = -(w_star[0] + w_star[1] * x1_array) / w_star[2] 

# create the scatterplot

plt.figure(figsize = (10, 10))
plt.scatter(x1_list, x2_list, c = colors)
plt.plot(x1_array, x2_array_0, linewidth=1.5, color='forestgreen', label='Decision Boundary')
plt.title('Scatterplot with randomly initialized weights')
plt.xlabel(r'$x_{1}$')
plt.ylabel(r'$x_{2}$')
#plt.show()
#plt.savefig('initial_scatterplot.png') # save figure

# ========== Point (1c) ========= #
# --------- Find the normal vector to the line ---------

print(type(-w_star[2].item())) # to extract a float element from the array

# define a vector tangent to the line

v_tangent = np.array([-w_star[2].item(), w_star[1].item()])
v_normal = np.array([w_star[1].item(), w_star[2].item()])

prod = np.dot(v_normal, v_tangent)
print(prod) # verify the the scalar product is 0!

# --------- Compute the distance between the point and the line ---------

# define line parameters

a = -w_star[1].item() / w_star[2].item()
#a = -w[1]/w[2]
print(type(a))
b = -1
c = -w_star[0].item() / w_star[2].item()

# define the point of interest 

xp = 0
yp = 0

d = np.abs(a*xp + b*yp + c) / math.sqrt(math.pow(a,2) + math.pow(b, 2))

# ========= Point (2a) ========= #
# Implementation of the perceptron algorithm --------- #

outputs = []

w_init = np.array([1, 1, 1]).reshape(3,1).astype(np.float64)

flag = 1 # flag to stop the algorithm when there are no more errors
eta = 1
errors_epoch_list = []

epochs = 0

while(flag):
    errors_epoch = 0 # reset errors of the current epocj
    epochs += 1
    for i in range(X.shape[0]):
        #print(f'Analizing data {i}')
    
        w_update, output = perceptron_step(w_init, (X[i]), (labels[i]), eta)
        outputs.append(output)
        if output != labels[i]:
            # an error was committed
            errors_epoch += 1
            w_init = w_update

    # end of epoch
    if errors_epoch == 0:
        # no errors were committed in the entire epoch --> stop the algorithm
        flag = 0
print('==================== Point 2a =====================')
print(f'Loop of point ended in {epochs} epochs ')
print(f'The final weights are :\n{w_update.ravel()}')
print(f'The initial weights were :\n{w_star.ravel()}')

# --------- Visualize Results ---------

# create the decision boundary line with the computed weights

x1_array = np.array(x1_list)
x2_array_1 = -(w_update[0] + w_update[1] * x1_array) / w_update[2] 

# create the scatterplot

plt.figure(figsize = (10, 10))
plt.subplot(2,1,1)
plt.scatter(x1_list, x2_list, c = colors)
plt.plot(x1_array, x2_array_1, linewidth=1.5, color='forestgreen', label='Decision Boundary')
plt.title('Scatterplot with the computed weights')
plt.subplot(2,1,2)
plt.plot(x1_array, x2_array_0, linewidth=1.5, color='purple', label='Decision Boundary')
plt.scatter(x1_list, x2_list, c = colors)
plt.title('Scatterplot with randomly initialized weights')
plt.xlabel(r'$x_{1}$')
plt.ylabel(r'$x_{2}$')
#plt.show()

# ========= Point (2b) ========= #

etas = [1,  0.1, 10] # regularization param
errors_per_eta = {} # create a dictionary to store the different number of errors per each learning rate
outputs = []

for eta in etas:
    w_init = np.array([1, 1, 1]).reshape(3,1).astype(np.float64)
    flag = 1 # flag to stop the algorithm when there are no more errors

    errors_epoch_list = []
   
    epochs = 0

    while(flag):
        errors_epoch = 0 # reset errors of the current epocj
        epochs += 1
        for i in range(X.shape[0]):
            w_update, output = perceptron_step(w_init, X[i], labels[i], eta)
            outputs.append(output)
            if output != labels[i]:
                # an error was committed
                errors_epoch += 1
                w_init = w_update

        # end of epoch
        errors_epoch_list.append(errors_epoch)
        if errors_epoch == 0:
            # no errors were committed in the entire epoch --> stop the algorithm
            flag = 0
    errors_per_eta[eta] = errors_epoch_list

    print('==================== Point 2b =====================')
    print(f'Loop ended in {epochs} epochs with eta = {eta}')
    print(f'The final weights are :\n{w_update.ravel()}')
    print(f'The initial weights were :\n{w_star.ravel()}')





# --------- Create a plot for the different values of the learning rate ---------

plt.figure(figsize=(10,10))
for i, eta in enumerate(etas, 1):
    plt.subplot(3,1,i)
    plt.plot(errors_per_eta[eta], linewidth = 1.5, label = f'eta = {eta}')
    plt.xlabel('Epochs')
    plt.ylabel('Errors')
    plt.legend()

plt.suptitle('Errors per Epoch for Different Learning Rates', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the layout to fit the suptitle
plt.show()

# ========= Point (2c) ========= #
x = np.random.uniform(-1,1, size = (1000,2))
x0 = np.ones((1000,1))
X = np.hstack((x0, x)) 

# labels

labels = []
for i in range(X.shape[0]):
    y = step(w_star, X[i])
    labels.append(y) # these are the true labels

# transform labels into an array

labels = np.array(labels)

# initialize weights

w_init = np.array([1,1,1]).reshape(3,1).astype(np.float64) 
flag = 1
eta = 1
errors_epoch = 0
errors_per_epoch_list = []
epochs = 0
while flag:
    errors_epoch = 0 # initialize errors for this epoch
    for i in range(X.shape[0]):
        #print(f'Analizing data {i}\n')
        w_update, output = perceptron_step(w_init, X[i], labels[i], eta)
        if output != labels[i]:
            errors_epoch += 1
    # end of epoch
    epochs += 1
    errors_epoch_list.append(errors_epoch)
    if errors_epoch == 0:
        flag = 0
print('==================== Point 2c =====================')
print(f"The true weights are: {w_star.ravel()}")
print(f"The updated weights are {w_update.ravel()} and were obtained after {epochs} epochs")

# ========= Point (2d) ======== #
etas = [1,  0.1, 10] # regularization param
errors_per_eta = {} # create a dictionary to store the different number of errors per each learning rate
outputs = []

for eta in etas:

    # for each eta initialize weights 100 times

    for k in range(100): 
        w_init = np.random.uniform(-1,1, size =(3,1)).astype(np.float64)
        flag = 1 # flag to stop the algorithm when there are no more errors

        errors_epoch_list = []
    
        epochs = 0

        while(flag):
            errors_epoch = 0 # reset errors of the current epocj
            epochs += 1
            for i in range(X.shape[0]):
                w_update, output = perceptron_step(w_init, X[i], labels[i], eta)
                outputs.append(output)
                if output != labels[i]:
                    # an error was committed
                    errors_epoch += 1
                    w_init = w_update

            # end of epoch
            errors_epoch_list.append(errors_epoch)
            if errors_epoch == 0:
                # no errors were committed in the entire epoch --> stop the algorithm
                flag = 0
        errors_per_eta[eta] = errors_epoch_list

        print(f'==================== Point 2d - round {k} =====================')
        print(f'Loop ended in {epochs} epochs with eta = {eta}')
        print(f'The final weights are :\n{w_update.ravel()}')
        print(f'The initial weights were :\n{w_star.ravel()}')
