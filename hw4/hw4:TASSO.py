# --------- Importing useful libraries ---------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
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

for j in range(len(mnist.data)):
    img = []
    # select a single row of data
    xrow = mnist.data.loc[j]

    for i, element in enumerate(xrow):
        #print(element)
        img.append(element) # form the image

    # once finished, transform into array and reshape
    
    img = np.array(img).reshape(28,28)
    #print(img.shape) # (28, 28)

    # select the corresponding label
    label = mnist.target[j]

    print(f"==== Created image n° {j} ====")

    # update dictionary

    if label not in images.keys():
        # create the list
        images[label] = []

    images[label].append(img)

    # update j

    j += 1


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


# Plot images

_, axes = plt.subplots(nrows = 2, ncols = 5, figsize = (10,4))
for ax, img, label in zip(axes.ravel(), img_examples, labels):
    
    ax.imshow(img, cmap = 'gray', interpolation = 'nearest')
    ax.set_axis_off()
    ax.set_title(f"Digit {label}")

plt.show()



# references
'''
https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html

https://h1ros.github.io/posts/loading-scikit-learns-mnist-dataset/

https://stackoverflow.com/questions/47324921/cant-load-mnist-original-dataset-using-sklearn

https://www.geeksforgeeks.org/accessing-elements-of-a-pandas-series/

https://www.geeksforgeeks.org/python-sort-python-dictionaries-by-key-or-value/

'''

