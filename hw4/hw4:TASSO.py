# --------- Importing useful libraries ---------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import pandas as pd
import tqdm

# --------- Load the full Dataset ---------
mnist = fetch_openml('MNIST_784')
print(mnist.keys())


#print("printing shapes")
#print(f"Shape of the image dataset {mnist.data.shape}") # (70000, 784)
#print(f"Labels shape {mnist.target.shape}") # (70000, )

# reshape images
#print((mnist.data.head()))
images = {}
j = 0

for xrow in mnist.data.iterrows():
    #print(xrow)
    img = []
    # iterate over the row
    print(f"Tipo riga {type(xrow)} e pixel {xrow[1]}")
    print(f' ciao {len(xrow)}')
    print(f'VALORI TUPLA: {xrow[0]}, {xrow[1]}')
    print(f'TIPO TUPLA {type(xrow[0])}, {type(xrow[1])}') # the second element is Pandas.Series
    ser = xrow[1]
    print(f'Lenght of the series {len(ser)}') # check (it should be 784 elements, i.e 784 pixels)
    for i,element in enumerate(ser):
        img.append(element)
    
    # once finished, transform into array and reshape
    img = np.array(img).reshape(28,28)

    # update dictionary
    images[j] = img
    j += 1
    print(j)
    



'''_, axes = plt.subplots(nrows = 1, ncols = 4, figsize = (10,3))
for ax, image, label in zip(axes, mnist.data, mnist.target):
    ax.set_axis_off()
    ax.imshow(image, cmap = 'gray', interpolation = 'nearest')
    ax.set_title(f"Training sample {label}")

plt.show()
'''

# references
'''
https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html

https://h1ros.github.io/posts/loading-scikit-learns-mnist-dataset/

https://stackoverflow.com/questions/47324921/cant-load-mnist-original-dataset-using-sklearn



'''