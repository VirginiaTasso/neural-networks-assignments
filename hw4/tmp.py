import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import numpy.linalg as lg

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
            print(lbl)
            Y_matrix.append(lbl)

Y_matrix = np.transpose(np.array(Y_matrix))

print(f"SHape of the Y_matrix: {Y_matrix.shape}")

# ========= Point c ========= #
d_list = [10, 50, 100, 200, 500]
# create a dictionary to store the weight vectors  and mean squared errors for the different values of d
W_vectors = {}
MSE = {}
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
    print(f" ==== X_matrix shape: {X_matrix.shape} ====")

    # probelm to solve W = YX'(XX')^-1
    A = Y_matrix
    B = lg.pinv(X_matrix) # should give as output X'(XX')^-1
    W_vector = np.matmul(A,B)
    W_vectors[d] = W_vector

    #print(f"==== The final Weight vector is : {W_vector.ravel()} ====")
    print(f"==== Final weight vector shape: {W_vector.shape} ====")

    # --------  Now make predictions --------- #

    y_pred = np.matmul(W_vector, X_matrix)
    print(y_pred.shape)
    mse = compute_mse(Y_matrix, y_pred)
    MSE[d] = mse

for key, value in MSE.items():
    print(f"MSE with d = {key}: {value}")

n_errors = compute_n_mistakes(Y_matrix, y_pred)
print(f"Number of errors: {n_errors}")


print('Generation of a random digit')
random_digit = np.random.randint(0, 9, 1)
print(random_digit[0])