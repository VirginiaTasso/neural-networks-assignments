import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

d = 50
M = np.random.uniform(0, 1, size = (d, 784))
M /= (d*255)
#print(M)

xrows = np.random.uniform(0, 255, size = (70000, 784))
yrows = np.random.randint(0,9, size = (70000,))
X_matrix = []
# iterate over rows
for row in range(xrows.shape[0]):

    # select one row
    xrow = xrows[row, :]
    tmp = np.transpose(xrow)
    #print(tmp.shape)
    prod = np.matmul(M, np.transpose(xrow))
    X_matrix.append(prod)

X_matrix = np.array(X_matrix)



labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
labels_enc = [[int(label)] for label in labels]  # Crea una lista di liste, ogni lista con una sola stringa


enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(labels_enc)


encoded_labels = enc.transform(labels_enc).toarray()

Y_matrix = []
for row in range(len(yrows)): # 70000
    value = yrows[row]
    for i in range(len(labels_enc)):       
        if value == labels_enc[i][0]:
            lbl = encoded_labels[i]
            Y_matrix.append(lbl)

Y_matrix = np.array(Y_matrix).reshape(10, 70000)
print(f"SHape of the Y_matrix: {Y_matrix.shape}")
