# --------- Importing useful libraries ---------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import time

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
print(M)

# --------- Create X_matrix ---------- #

X_matrix = []
# iterate over rows

for row in range(xrows.shape[0]):

    # select one row
    xrow = xrows[row, :]
    tmp = np.transpose(xrow)
    prod = np.matmul(M, np.transpose(xrow))
    X_matrix.append(prod)

X_matrix = np.array(X_matrix)
print("==== yrows ====")
print(yrows[0:5,])


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

Y_matrix = np.array(Y_matrix).reshape(10, 70000)






# references
'''
https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html

https://h1ros.github.io/posts/loading-scikit-learns-mnist-dataset/

https://stackoverflow.com/questions/47324921/cant-load-mnist-original-dataset-using-sklearn

https://www.geeksforgeeks.org/accessing-elements-of-a-pandas-series/

https://www.geeksforgeeks.org/python-sort-python-dictionaries-by-key-or-value/


https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

'''

