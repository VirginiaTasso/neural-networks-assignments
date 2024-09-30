# =============================================== #
# ========= Importing Useful Libraries ========= #
# =============================================== #
import numpy as np
import matplotlib.pyplot as plt
import math

# =============================================== #
# ========= Definition of useful functions ========= #
# =============================================== #

def compute_distance(x1, y1, x2, y2):
    
    dist = math.sqrt((x1-x2)**2 + (y1 -y2)**2)
    return dist


def compute_gradient(w1, w2):
    '''
    Function to compute the gradient of R(w1, w2)
    '''
    dR_dw1 = 26 * w1 - 10 * w2 + 4
    dR_dw2 = -10 * w1 + 4 * w2 - 2
    return np.array([dR_dw1, dR_dw2])


def gradient_descend(weights, gradient, eta):
    '''
    Function to implement one step of gradient descend
    :param weights: np.ndarray
        The weights to update.
    :param gradients: np.ndarray
        The computed gradients.
    :param eta: float
        The learning rate.
        
    :return: np.ndarray
        The updated weights
    '''
    return weights - eta * gradient
    


def sigmoid(f, a = 5):
    '''
    Implementation of the sigmoid activation function
    :param f: np.ndarray or float
        The input value(s) to apply the sigmoid function on. Can be a single float or an array.
    :param a: float, optional, default=5
        The steepness parameter of the sigmoid function. Controls how steep or flat the curve is.
        Higher values of `a` make the curve steeper.
    :return: np.ndarray or float
        The sigmoid-transformed value(s) in the range (0, 1).
    '''
    return 1 / (1 + np.exp(-a*f))

def diff_sigmoid(f, a = 5):
    '''
    Derivative of the sigmoid function
    :param f: np.ndarray or float
        The input value(s) to apply the sigmoid derivative on. Can be a single float or an array.
    :param a: float, optional, default=5
        The steepness parameter of the sigmoid function. Controls the rate of change in the derivative.
    :return: np.ndarray or float
        The derivative of the sigmoid function with respect to the input `f`.
    '''
    return (- np.exp(-a*f)) / (1 + np.exp(-a*f))**2

def forward_pass(x, W, b, U, c):
    vz = np.dot(W, x.T) + b # (3, 1)
    z = sigmoid(vz) # (3, 1)
    vf = np.dot(U, z) + c.T # (1, 1)
    f = sigmoid(vf)  # (1, 1)

    '''print('Forward')
    print('='*20)
    print(f"W shape {W.shape}")
    print(f"Input shape: {x.shape}")
    print(f"bias shape: {b.shape}")
    print(f"Shape of vz (pre-activation hidden layer): {vz.shape}")
    print(f"Shape of z (activation hidden layer): {z.shape}")
    print(f"Shape of vf (pre-activation output layer): {vf.shape}")
    print(f"Shape of f (final output): {f.shape}")'''

    return vz, z, vf, f

def backward_pass(x,f, y, vf, z, vz, W, U):
    """
    :param x: current input
    :param f: output of the neural network
    :param y: target value (the one to which the output should be as close as possible)
    """
    # compute all the errors

    deltavf = diff_sigmoid(vf)*2*(f - y) # (1, 1) 

    deltaz = np.dot(deltavf.T, U) # (1,1) * (1, 3) = (1, 3)
    deltavz = diff_sigmoid(vz).T * deltaz  # (1, 3) 

    '''print('Backward')
    print('='*20)
    print(f"U shape {U.shape}")
    print(f"Shape of vz (pre-activation hidden layer): {vz.shape}")
    print(f"Shape of z (activation hidden layer): {z.shape}")
    print(f"Shape of vf (pre-activation output layer): {vf.shape}")
    print(f"Shape of f (final output): {f.shape}")'''
    
    # compute gradients
    deltaW = np.dot(deltavz.T, x) # (3, 1) * (1, 2) = (3, 2)
    deltaU = np.dot(deltavf, z.T)  # (1, 1) * (1, 3) = (1, 3)
    deltab = deltaz.T
    deltac = deltavf

    return deltaW, deltaU, deltab, deltac

def compute_mse(y, y_pred):
    """
    Compute the Mean Square Error

    :param y: ground truth label
    :param y_pred: predicted value

    :returns: the mean square error (:py:class:`~int`)
    """
     
    return np.mean((y - y_pred)**2)

# ================================ #
# ========= Point (1b) ========= #
# ================================ #

# ========= Implement Gradient Descend ========= #
# randomly initialize weights

# fix a seed for reproducibility

np.random.seed(11)
w1 = np.random.randint(1,2)
w2 = np.random.randint(1,2)
R = 13 * w1**2 - 10 * w1 * w2 + 4 * w1 + 2 * w2**2 - 2 * w2 + 1
dR = [26*w1 - 10 + 4, -10*w1 + 4*w2 -2]
dR = np.asarray(dR)


etas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.21]
iterations = 500
w1_true = 1
w2_true = 3
distances = {} # initialize dictionary to store all distances for each iteration for each eta

w1_upd = w1
w2_upd = w2
for eta in etas:
    distances[eta] = []
    for it in range(iterations):
        # update gradient
        dR = compute_gradient(w1_upd, w2_upd)

        # step of gradient descend
        weights = gradient_descend([w1_upd, w2_upd], dR, eta)
        w1_upd = weights[0]
        w2_upd = weights[1]
        
        dist = compute_distance(w1_true, w2_true, w1_upd, w2_upd)
        distances[eta].append(dist)  
        print('=' * 20)
        print(f'Iteration {it+1}, eta={eta}')
        print(f'Distance: {dist}')
        print(f'w1 and w2: {w1_upd}, {w2_upd}')
   

# ========= Plot Results ========= #
colors = ['r', 'hotpink', 'mediumblue', 'm', 'orange', 'forestgreen']
_, axes= plt.subplots(nrows = int(np.floor(len(etas) / 2)), ncols = int((len(etas) / 3)), figsize = (8, 10))
for ax, eta, color in zip(axes.ravel(), etas, colors):
    ax.plot([dist for dist in distances[eta]], linewidth = 1.5, color = color)
    ax.set_xlabel('N° iterations', fontsize = 16)
    ax.set_ylabel('Distance', fontsize = 16)
    ax.set_title(f'Distance for $\eta$ = {eta}', fontsize = 18)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 14)
    ax.grid(True)

if len(etas) < len(axes.ravel()):
    axes.ravel()[-1].axis('off') 

plt.tight_layout()
plt.show()

# ===================================== #
# ========= Point (2d) ========= #
# ===================================== #
# ========= Backpropagation ========= #

mu = 0
sigma = 0.1

x = np.random.uniform(-2, 2, size=(1000, 2))
W = np.random.normal(mu,sigma, size = (3,2))
U = np.random.normal(mu,sigma, size = (1,3))
b = np.random.normal(mu,sigma, size = (3,1))
c = np.random.normal(mu,sigma, size = (1,1 ))

z1 = (x[:, 0] - x[:, 1] + 1 >= 0)*1.
z2 = (-x[:, 0] - x[:, 1] + 1 >= 0)*1.
z3 = (-x[:, 1] -1 >= 0)*1.
y = np.heaviside(z1 + z2 -z3 -1.5, 1)
#y = np.expand_dims(y, axis = 1)

plt.figure(figsize = (6,8))
plt.scatter(x[:, 0][np.where(y == 1)], x[:, 1][np.where(y == 1)], c = 'red')
plt.scatter(x[:, 0][np.where(y == 0)], x[:, 1][np.where(y == 0)], c = 'blue')
#plt.show()

vz, z, vf, f = forward_pass(x, W,b, U, c)

# set hyperparameters
epochs = 100
eta = 0.01
mse_list = []

W_new = W
U_new = U
b_new = b
c_new = c
for epoch in range(epochs):
    mse_epoch = 0

    for i in range(x.shape[0]):
        inputs = np.expand_dims(x[i, :], axis= 0) # (1, 2)
        target = y[i]

        # Forward Pass

        vz, z, vf, f = forward_pass(inputs, W_new, b_new, U_new, c_new)


        # Backward Pass

        dW, dU, db, dc = backward_pass(inputs, f, target, vf, z, vz, W_new, U_new)

        '''print(f"Shape of dW (gradient for W): {dW.shape}")
        print(f"Shape of dU (gradient for U): {dU.shape}")
        print(f"Shape of db (gradient for b): {db.shape}")
        print(f"Shape of dc (gradient for c): {dc.shape}")'''

        # Gradient Descend

        W_new = gradient_descend(W, dW, eta)
        U_new = gradient_descend(U, dU, eta)
        b_new = gradient_descend(b, db, eta)
        c_new = gradient_descend(c, dc, eta)



        # 
        mse_epoch += compute_mse(target, f)
    
    # at the end of each epoch compute mse
    mse_epoch /= x.shape[0]
    mse_list.append(mse_epoch)
    print(f"Epoch {epoch+1}/{epochs}, MSE: {mse_epoch}")

eps = 1e-6
plt.figure(figsize = (10, 8))
plt.plot(range(epochs), mse_list, linewidth = 1.5, c = 'b')
plt.title('MSE over epochs', fontsize = 20)
plt.xlabel('Epochs', fontsize = 18)
plt.ylabel('MSE', fontsize = 18)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(True)
plt.show()

plt.figure(figsize = (10, 8))
plt.plot(range(5), mse_list[0:5], linewidth = 1.5, c = 'b')
plt.title('MSE over epochs', fontsize = 20)
plt.xlabel('Epochs', fontsize = 18)
plt.ylabel('MSE', fontsize = 18)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.xticks(range(5))
plt.grid(True)
plt.show()