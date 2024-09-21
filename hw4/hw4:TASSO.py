# --------- Importing useful libraries ---------

import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import time


# ========= Define Useful Functions =========
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

d = 50
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
print(f" ==== X_matrix shape: {X_matrix.shape} ====")

# --------- Create Y_Matrix --------- #
labels_enc = [[int(label)] for label in labels] 


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
W_vector = np.matmul(A,B)

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
    W_vector = np.matmul(A,B)
    W_vectors[d] = W_vector

    # --------  Now make predictions --------- #

    y_pred = np.matmul(W_vector, X_matrix)

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


# with increasing d the number of errors and the MSE reduce
# d represents the number of features. the X matrix has 784 features with d = 1, but the number
# can vary to dx784, depending on the value of d.
# By multiplying X with M, which is a random matrix with uniformously distributed values,
# the original images are projected in a new feature space.
# The greater d, the more we are able to keep info from the original data.
# A small value of d means that data are being compressed in a space with less features,
# so there is the risk of losing information.

# so it's quite reasonable that with a greater value of d, the number of errors diminishes,
# because more features are being preserved

# --------- Compare the number of obtainer errors to the number of errors I would obtaine my randomly selecting the digit ---------#

n_random_errors = np.sum(np.argmax(Y_matrix, axis = 0) != np.random.randint(0,10,Y_matrix.shape[1]))


print('='*20)
print(f"Number of errors committed with random guessing of digits: {n_random_errors}")
print('='*20)

# ---------  Plot results --------- #

_, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 3))

axes[0].plot(d_list, [mse[d] for d in d_list], marker='o', color = 'b')
axes[0].set_title('MSE for different values of d')
axes[0].set_xlabel('d')
axes[0].set_ylabel('MSE')
axes[0].grid(True)

axes[1].plot(d_list, [n_errors[d] for d in d_list], marker = 'o', color = 'r')
axes[1].set_title("Number of errors for different values of d")
axes[1].set_xlabel('d')
axes[1].set_ylabel(r"$n_{errors}$")
axes[1].grid(True)

plt.tight_layout()
plt.show()

# references
'''
https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html

https://h1ros.github.io/posts/loading-scikit-learns-mnist-dataset/

https://stackoverflow.com/questions/47324921/cant-load-mnist-original-dataset-using-sklearn

https://www.geeksforgeeks.org/accessing-elements-of-a-pandas-series/

https://www.geeksforgeeks.org/python-sort-python-dictionaries-by-key-or-value/


https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html#numpy.linalg.pinv

https://stackoverflow.com/questions/39064684/mean-squared-error-in-python

'''

