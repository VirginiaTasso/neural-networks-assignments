# --------- Importing useful libraries ---------

import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
import math
import pandas as pd


# ========= Define Useful Functions =========

def reduce_dim(xrows,input_feature_dimension, d):
    '''
    Function to reduce dimensionality of the input data
    :param X: input data (dim: (num_data, input_feature_dimension) )
    :param input_feature_dimension: original feature dimensionality
    :param d: new dimensionality
    :returns: data with reduced dimensionality
    '''
    M = np.random.uniform(0, 1, size = (d, input_feature_dimension))
    M /= (d*255)

    # --------- Create data with new dimensionality --------

    xrows_reduced = []
    # iterate over rows

    for row in range(xrows.shape[0]):

        # select one row
        xrow = xrows[row, :]
        prod = np.matmul(M, np.transpose(xrow))
        xrows_reduced.append(prod)

    xrows_reduced = np.transpose(np.array(xrows_reduced))
    return xrows_reduced


def compute_mse(y, y_pred):
    """
    Compute the Mean Square Error

    :param y: ground truth label
    :param y_pred: predicted value

    :returns: the mean square error (:py:class:`~int`)
    """
     
    return np.mean((y - y_pred)**2)

def compute_n_mistakes(y, y_pred):
    '''
    Function to compute the number of errors when labels are one-hot encoded
    '''
    n_errors = 0
    for col in range(y.shape[1]):
        curr_true_val = y[:,col]
        predicted_val = y_pred[:,col]
        if(np.argmax(curr_true_val) != np.argmax(predicted_val)):
            n_errors += 1
    return n_errors

def compute_n_mistakes_optimized(y, y_pred):
    return np.sum(np.argmax(y, axis = 0) != np.argmax(y_pred, axis = 0))

# ========= Point a ========= #
# --------- Format correcly data and create images --------- #
#  Load the full Dataset 
mnist = fetch_openml('MNIST_784')

# Iterate over data to collect all pixels together to form each image
# store them in a python dictionary with the corresponding label

images = {}
j = 0

# define xrow and yrow vectors

xrows = []
yrows = []

for j in range(len(mnist.data)):
    img = []
    # select a single row of data
    xrow = mnist.data.loc[j]
    xrows.append(xrow)

    for i, element in enumerate(xrow):
        #print(element)
        img.append(element) # form the image

    # once finished, transform into array and reshape
    
    img = np.array(img).reshape(28,28)
    #print(img.shape) # (28, 28)

    # select the corresponding label
    label = mnist.target[j]
    yrows.append(label)

    print(f"==== Created image n° {j} ====")

    # update dictionary

    if label not in images.keys():
        # create the list
        images[label] = []

    images[label].append(img)

    # update j

    j += 1

xrows = np.array(xrows) # (70000, 784)
yrows = np.array(yrows) # (70000, )


# sort keys because they are in random order
myKeys = list(images.keys())
myKeys.sort()
sorted_images = {i: images[i] for i in myKeys}

# pick one example for digit
labels = []
img_examples = []

for key in sorted_images.keys():
    labels.append(key)
    img_examples.append(sorted_images[key][0])

print(f'Variable type: {type(labels[0])}')

# Plot images

_, axes = plt.subplots(nrows = 2, ncols = 5, figsize = (10,4))
for ax, img, label in zip(axes.ravel(), img_examples, labels):
    
    ax.imshow(img, cmap = 'gray', interpolation = 'nearest')
    ax.set_axis_off()
    ax.set_title(f"Digit {label}")

plt.show()

# ========= Point b ========= #
# --------- Define the random feature extractor ---------

#d = 50
d = 484
M = np.random.uniform(0, 1, size = (d, 784))
M /= (d*255)

# --------- Create X_matrix ---------- #

X_matrix = []
# iterate over rows

for row in range(xrows.shape[0]):

    # select one row
    xrow = xrows[row, :]
    prod = np.matmul(M, np.transpose(xrow))
    X_matrix.append(prod)

X_matrix = np.transpose(np.array(X_matrix))
print(f" ==== X_matrix shape: {X_matrix.shape} ====") # 70000, d

# --------- Create Y_Matrix --------- #
labels_enc = [[int(label)] for label in labels] 

# check what happened to images

images_reduced = []

for i in range(X_matrix.shape[1]):
    n1 = int(math.sqrt(d))
    while d % n1 != 0:
        n1 -= 1
    n2 = d // n1
    img = X_matrix[:, i].reshape(n1, n2) # select an entire row
    images_reduced.append(img)



_, axes = plt.subplots(nrows = 2, ncols = 5, figsize = (10,4))
for ax, img in zip(axes.ravel(), images_reduced[:10]):
    
    ax.imshow(img, cmap = 'gray', interpolation = 'nearest')
    ax.set_axis_off()
    ax.set_title(f"Digit {label} reduced")

plt.show()
 



enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(labels_enc)


encoded_labels = enc.transform(labels_enc).toarray()

Y_matrix = []
for row in range(len(yrows)): # 70000
    value = yrows[row]
    for i in range(len(labels_enc)):  
        #print('Value: ', value)
        #print('Label: ', labels_enc[i][0])     
        if int(value) == labels_enc[i][0]:
            lbl = encoded_labels[i]
            Y_matrix.append(lbl)

Y_matrix = np.transpose(np.array(Y_matrix))

print(f" ==== Y_matrix shape: {Y_matrix.shape} ====")

# probelm to solve W = YX'(XX')^-1
A = Y_matrix
B = lg.pinv(X_matrix) # should give as output X'(XX')^-1
W_vector = np.dot(A,B)

# ========= Point c ========= #
d_list = [10, 50, 100, 200, 500]
# create a dictionary to store the weight vectors  and mean squared errors for the different values of d
W_vectors = {}
mse = {}
n_errors = {}
# repeat previous process for all d's

for d in d_list:
    M = np.random.uniform(0, 1, size = (d, 784))
    M /= (d*255)

    # --------- Create X_matrix ---------- #

    X_matrix = []
    # iterate over rows

    for row in range(xrows.shape[0]):

        # select one row
        xrow = xrows[row, :]
        tmp = np.transpose(xrow)
        prod = np.matmul(M, np.transpose(xrow))
        X_matrix.append(prod)

    X_matrix = np.transpose(np.array(X_matrix))

    # probelm to solve W = YX'(XX')^-1
    A = Y_matrix
    B = lg.pinv(X_matrix) # should give as output X'(XX')^-1
    W_vector = np.dot(A,B)
    W_vectors[d] = W_vector

    # --------  Now make predictions --------- #

    y_pred = np.dot(W_vector, X_matrix)

    # MSE for the different predictors
    mse[d] = compute_mse(Y_matrix, y_pred)

    # number of errors for the possible predictors
    n_errors[d] = compute_n_mistakes(Y_matrix, y_pred)


print(" ==== MSE and number of errors for the different predictors ==== ")
for (key_mse, value_mse), (key_err, value_err) in zip(mse.items(), n_errors.items()):
    print('='*20)
    print(f"MSE with d = {key_mse}: {value_mse}")
    print(f"Number of errors with d = {key_err}: {value_err}")
    print('='*20)
    print('\n\n')

# --------- Compare the number of obtainer errors to the number of errors I would obtaine my randomly selecting the digit ---------#

n_random_errors = np.sum(np.argmax(Y_matrix, axis = 0) != np.random.randint(0,10,Y_matrix.shape[1]))


print('='*20)
print(f"Number of errors committed with random guessing of digits: {n_random_errors}")
print('='*20)

# ---------  Plot results --------- #

_, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 5))

axes[0].plot(d_list, [mse[d] for d in d_list], marker='o', color = 'b')
axes[0].set_title('MSE for different values of d', fontsize = 20)
axes[0].set_xlabel('d', fontsize = 18)
axes[0].set_ylabel('MSE', fontsize = 18)
axes[0].grid(True)

axes[1].plot(d_list, [n_errors[d] for d in d_list], marker = 'o', color = 'r')
axes[1].set_title("Number of errors for different values of d", fontsize = 20)
axes[1].set_xlabel('d', fontsize = 18)
axes[1].set_ylabel(r"$n_{errors}$", fontsize = 18)
axes[1].grid(True)

plt.tight_layout()
plt.savefig('point2c.png')
#plt.show()


# ========= Point d ========= #
# ---------  Implement the Widrow-Hoff LMS --------- #
d = 100
W = np.zeros((10, d))
lr = 0.001
n_epochs = 10
X_matrix = reduce_dim(xrows, 784, d)
print(f"X_matrix dimensions with function: {X_matrix.shape}")
mse_list = []
n_errors_list = []
for epoch in range(n_epochs):
    for col in range(X_matrix.shape[1]): # iterate thorugh each sample
        sample = X_matrix[:,col].reshape(-1,1) # select one sample; dim (d, 1)
        y_true = Y_matrix[:, col].reshape(-1,1) # reshape to obtain shape(10,1), otherwise it woudl be (10,)
        # make prediction with current weights
        y_pred_sample = np.dot(W, sample) # prediction on the current sample dim (10, 1)

        error = y_true - y_pred_sample

        # update weights
        W += lr * np.dot(error, np.transpose(sample))
    
    y_pred_total = np.dot(W, X_matrix)
    # MSE for each epoch
    mse = compute_mse(Y_matrix, y_pred_total)
    mse_list.append(mse)
    print('='*20)
    print(f"Epoch n° {epoch + 1}: MSE: {mse}\t Number of errors: {n_errors}")

    # number of errors for each epoch
    n_errors = compute_n_mistakes(Y_matrix, y_pred_total)
    n_errors_list.append(n_errors)

# -------- Plot final results --------- #

_, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 5))
axes[0].plot(range(1, n_epochs +1), mse_list, linewidth = 1.5, marker = 'o', color = 'b')
axes[0].set_title('MSE across epochs', fontsize = 20)
axes[0].set_xlabel('Epochs', fontsize = 18)
axes[0].set_ylabel('MSE', fontsize = 18)
axes[0].grid(True)

axes[1].plot(range(1, n_epochs +1), n_errors_list, linewidth = 1.5, marker = 'o', color = 'r')
axes[1].set_title(r"$N_{errors}$ across epochs", fontsize = 20)
axes[1].set_xlabel('Epochs', fontsize = 18)
axes[1].set_ylabel(r'$n_{errors}$', fontsize = 18)
axes[1].grid(True)

plt.tight_layout()
plt.savefig('pointd.png')


print('='*20)
print(f"Number of errors at the final epoch: {n_errors_list[-1]} ")
print('='*20)
print('\n\n')
print('='*20)
print(f"MSE at the last epoch; {mse_list[-1]}")
print('='*20)


# very bad result-> try smaller lr and more epoch

d = 100
W = np.zeros((10, d))
lr = 0.0001
n_epochs = 100
X_matrix = reduce_dim(xrows, 784, d)
mse_list = []
n_errors_list = []
for epoch in range(n_epochs):
    for col in range(X_matrix.shape[1]): # iterate thorugh each sample
        sample = X_matrix[:,col].reshape(-1,1) # select one sample
        y_true = Y_matrix[:, col].reshape(-1,1)
        # make prediction with current weights
        y_pred_sample = np.dot(W, sample) # prediction on the current sample

        error = y_true - y_pred_sample

        # update weights
        W += lr * np.dot(error, np.transpose(sample))
    
    y_pred_total = np.dot(W, X_matrix) # dim (10, 70000)
    # MSE for each epoch
    mse = compute_mse(Y_matrix, y_pred_total)
    mse_list.append(mse)
    print('='*20)
    print(f"Epoch n° {epoch + 1}: MSE: {mse}\t Number of errors: {n_errors}")
    print('='*20)
    print('\n')

    # number of errors for each epoch
    n_errors = compute_n_mistakes(Y_matrix, y_pred_total)
    n_errors_list.append(n_errors)

# -------- Plot final results --------- #

_, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 5))
axes[0].plot(mse_list, linewidth = 1.5, marker = 'o', color = 'b')
axes[0].set_title('MSE across epochs', fontsize = 20)
axes[0].set_xlabel('Epochs', fontsize = 18)
axes[0].set_ylabel('MSE', fontsize = 18)
axes[0].grid(True)

axes[1].plot(n_errors_list, linewidth = 1.5, marker = 'o', color = 'r')
axes[1].set_title(r"$N_{errors}$ across epochs", fontsize = 20)
axes[1].set_xlabel('Epochs', fontsize = 18)
axes[1].set_ylabel(r'$n_{errors}$', fontsize = 18)
axes[1].grid(True)

plt.tight_layout()
plt.savefig('pointd2.png')


print('='*20)
print(f"Number of errors at the final epoch: {n_errors_list[-1]} ")
print('='*20)
print('\n\n')
print('='*20)
print(f"MSE at the last epoch; {mse_list[-1]}")
print('='*20)


