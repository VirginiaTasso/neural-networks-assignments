# --------- Importing useful libraries ---------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import pandas as pd
import tqdm

# --------- Load the full Dataset ---------
mnist = fetch_openml('MNIST_784')
print(mnist.keys())
print(len(mnist))
# show some of the data
'''https://stackoverflow.com/questions/47324921/cant-load-mnist-original-dataset-using-sklearn
_, axes = plt.subplots(nrows = 1, ncols = 4, figsize = (10,3))
for ax, image, label in zip(axes, mnist.images, mnist.target):
    ax.set_axis_off()
    ax.imshow(image, cmap = 'gray', interpolation = 'nearest')
    ax.set_title(f"Training sample {label}")

plt.show()'''


# references
'''
https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html

https://h1ros.github.io/posts/loading-scikit-learns-mnist-dataset/

https://stackoverflow.com/questions/47324921/cant-load-mnist-original-dataset-using-sklearn

'''