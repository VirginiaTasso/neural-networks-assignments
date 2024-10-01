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
    """
    Function to compute the gradient of R(w1, w2)
    """
    dR_dw1 = 26 * w1 - 10 * w2 + 4
    dR_dw2 = -10 * w1 + 4 * w2 - 2
    return np.array([dR_dw1, dR_dw2])


def gradient_descend(weights, gradients, eta):
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
    return weights - eta * gradients
    


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

def diff_sigmoid(f, a):
    '''
    Derivative of the sigmoid function
    :param f: np.ndarray or float
        The input value(s) to apply the sigmoid derivative on. Can be a single float or an array.
    :param a: float, optional, default=5
        The steepness parameter of the sigmoid function. Controls the rate of change in the derivative.
    :return: np.ndarray or float
        The derivative of the sigmoid function with respect to the input `f`.
    '''
    return a * sigmoid(f, a) * (1 - sigmoid(f, a))

def compute_vz(x, W, b):
    """
    Compute the local field of the first neuron
    """
    return np.dot(W, x.T) + b

def compute_vf(z, U, c):
    """
    Compute the local field of the second neuron
    :param z: np.ndarray
        The output of the hidden layer
    :param U: np.ndarray
        weight matrix of the ouput layer
    :param c: np.ndarray
        bias of the ouptut layer
    """
    return np.dot(U, z) + c


def forward_pass(x, W, b, U, c, a = 5):
    """
    Implements the forward pass of the learning algorithm
    :param x: np.ndarray
        input data
    :param W: np.ndarray
        weight matrix of the input layer
    :param b: np.ndarray
        biases of the input layer
    :param U: np.ndarray
        weight matrix of the hidden layer
    :param c: np.ndarray
        bias of the ouptut layer
    :param a: int
        regularization paramter of the sigmoid function
    :return: np.ndarrays
        returns outputs and hidden variables useful for the backward pass
        
    """
    vz = np.dot(W, x.T) + b  # (3, 2) * (2, 1) = 3, 1

    z = sigmoid(vz, a)  # (3, 1)

    vf = np.dot(U, z) + c   # (1, 3) (3, 1) = (1, 1)

    f = sigmoid(vf, a)  # (1, 1)

    return vz, z, vf, f

def backward_pass(x,f, y, vf, z, vz, W, U, a = 5):
    """Implements the backward pass of the learning algorithm
    All the deltas and useful elements for the gradients are computed
    """
    # Backward Pass
    deltaf = 2 * (f - y) # (1, 1)

    deltavf = deltaf * diff_sigmoid(vf, a)  # (1,1) * (1, 1)
  
    deltaz = np.dot(deltavf, U) # (1, 1) * (1, 3) = (1, 3)

    deltavz = deltaz * diff_sigmoid(vz, a).T # (3, 1) (3, 1) 

    deltax = np.dot(deltavz, W)  # (3, 1)' * (3, 2) = (1, 3) * (3, 2) = (1, 2)
  
    deltaW = np.dot(deltavz.T, x)  # (3, 1) * (1, 2)
  
    deltaU = np.dot(deltavf, z.T) # (1, 1)  * (1, 3)
  
    deltab = deltavz.T

    deltac = deltaf 

    return  deltaW, deltaU, deltab, deltac


def compute_mse(y, y_pred):
    """
    Compute the Mean Square Error

    :param y: ground truth label
    :param y_pred: predicted value

    :returns: the mean square error (:py:class:`~int`)
    """
     
    return np.mean((y - y_pred)**2)

def draw_decision_boundary(inputs, pred_vect):
    """
    Draw the decision boundary with the computex weights
    """

    # Prepare the data for the 3D scatter plot
    predictions = pred_vect.flatten()  # Convert to a 1D array

    # Creating the 3D scatter plot
    fig = plt.figure(figsize=(7, 10))
    ax = plt.axes(projection="3d")
    ax.scatter3D(inputs[:, 0], inputs[:, 1], predictions, color="green")

    # Setting the labels
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    ax.zaxis._axinfo['juggled'] = (2, 2, 1)

    # Show the plot
    plt.show()

# ================================ #
# ========= Point (1b) ========= #
# ================================ #

# ========= Implement Gradient Descend ========= #
# randomly initialize weights

# fix a seed for reproducibility

np.random.seed(2)
npoints = 1000

# ============================================================== #

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

x = np.random.uniform(-2, 2, size=(npoints, 2))
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
f_s= np.zeros((npoints, 1))


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
        f_s[i, 0] = f

        # Backward Pass

        dW, dU, db, dc = backward_pass(inputs, f, target, vf, z, vz, W_new, U_new)

        # Gradient Descend

        W_new = gradient_descend(W_new, dW, eta) 
        U_new = gradient_descend(U_new, dU, eta)
        b_new = gradient_descend(b_new, db, eta)
        c_new = gradient_descend(c_new, dc, eta)

        mse_epoch += compute_mse(target, f)
    
    # at the end of each epoch compute mse
    mse_epoch /= x.shape[0]
    mse_list.append(mse_epoch)
    print(f"Epoch {epoch+1}/{epochs}, MSE: {mse_epoch}")


plt.figure(figsize = (10, 8))
plt.plot(range(epochs), mse_list, linewidth = 1.5, c = 'b')
plt.title('MSE over epochs', fontsize = 20)
plt.xlabel('Epochs', fontsize = 18)
plt.ylabel('MSE', fontsize = 18)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(True)
plt.show()

# ================================ #
# ========= Point (2d) ========= #
# ================================ #

# --------- Compute and Plot the decision boundary ---------

draw_decision_boundary(x, f_s)



# ============================== #
# ========= Point (2f) ========= #
# ============================== #

# ========= Try different strategies to improve the outcomr =========#

# decrease epochs, higher lr
# set hyperparameters
print("Test 1")
print('='*20)
epochs = 50
eta = 0.1
a = 5
mse_list = []
f_s= np.zeros((npoints, 1))


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

        vz, z, vf, f = forward_pass(inputs, W_new, b_new, U_new, c_new, a)
        f_s[i, 0] = f

        # Backward Pass

        dW, dU, db, dc = backward_pass(inputs, f, target, vf, z, vz, W_new, U_new, a)

        # Gradient Descend

        W_new = gradient_descend(W_new, dW, eta) 
        U_new = gradient_descend(U_new, dU, eta)
        b_new = gradient_descend(b_new, db, eta)
        c_new = gradient_descend(c_new, dc, eta)

        mse_epoch += compute_mse(target, f)
    
    # at the end of each epoch compute mse
    mse_epoch /= x.shape[0]
    mse_list.append(mse_epoch)
    print(f"Epoch {epoch+1}/{epochs}, MSE: {mse_epoch}")


plt.figure(figsize = (10, 8))
plt.plot(range(epochs), mse_list, linewidth = 1.5, c = 'b')
plt.title('MSE over epochs', fontsize = 20)
plt.xlabel('Epochs', fontsize = 18)
plt.ylabel('MSE', fontsize = 18)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(True)
plt.show()


# --------- Compute and Plot the decision boundary ---------

draw_decision_boundary(x, f_s)



# decrease epochs, higher lr
# set hyperparameters
print("Test 1")
print('='*20)
epochs = 500
eta = 0.001
a = 5
mse_list = []
f_s= np.zeros((npoints, 1))


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

        vz, z, vf, f = forward_pass(inputs, W_new, b_new, U_new, c_new, a)
        f_s[i, 0] = f

        # Backward Pass

        dW, dU, db, dc = backward_pass(inputs, f, target, vf, z, vz, W_new, U_new, a)

        # Gradient Descend

        W_new = gradient_descend(W_new, dW, eta) 
        U_new = gradient_descend(U_new, dU, eta)
        b_new = gradient_descend(b_new, db, eta)
        c_new = gradient_descend(c_new, dc, eta)

        mse_epoch += compute_mse(target, f)
    
    # at the end of each epoch compute mse
    mse_epoch /= x.shape[0]
    mse_list.append(mse_epoch)
    print(f"Epoch {epoch+1}/{epochs}, MSE: {mse_epoch}")


plt.figure(figsize = (10, 8))
plt.plot(range(epochs), mse_list, linewidth = 1.5, c = 'b')
plt.title('MSE over epochs', fontsize = 20)
plt.xlabel('Epochs', fontsize = 18)
plt.ylabel('MSE', fontsize = 18)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(True)
plt.show()


# --------- Compute and Plot the decision boundary ---------

draw_decision_boundary(x, f_s)


# decrease epochs, higher lr, introduce patient
# set hyperparameters
print("Test 3")
print('='*20)
epochs = 50
eta = 0.1
a = 5
mse_list = []
f_s= np.zeros((npoints, 1))
T = 5
count = 0

W_new = W
U_new = U
b_new = b
c_new = c

mse_old = float('inf') # initialize error
for epoch in range(epochs):
    mse_epoch = 0

    for i in range(x.shape[0]):
        inputs = np.expand_dims(x[i, :], axis= 0) # (1, 2)
        target = y[i]

        # Forward Pass

        vz, z, vf, f = forward_pass(inputs, W_new, b_new, U_new, c_new, a)
        f_s[i, 0] = f

        # Backward Pass

        dW, dU, db, dc = backward_pass(inputs, f, target, vf, z, vz, W_new, U_new, a)

        # Gradient Descend

        W_new = gradient_descend(W_new, dW, eta) 
        U_new = gradient_descend(U_new, dU, eta)
        b_new = gradient_descend(b_new, db, eta)
        c_new = gradient_descend(c_new, dc, eta)

        mse_epoch += compute_mse(target, f)
    
    # at the end of each epoch compute mse
    mse_epoch /= x.shape[0]
    mse_list.append(mse_epoch)
    if mse_epoch > mse_old:
        count += 1
    mse_old = mse_epoch

    if count >= T:
        print(f"Not improving: reducing learning rate to {eta*0.9}")
        eta *= 0.9
    print(f"Epoch {epoch+1}/{epochs}, MSE: {mse_epoch}")


plt.figure(figsize = (10, 8))
plt.plot(range(epochs), mse_list, linewidth = 1.5, c = 'b')
plt.title('MSE over epochs', fontsize = 20)
plt.xlabel('Epochs', fontsize = 18)
plt.ylabel('MSE', fontsize = 18)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(True)
plt.show()


# --------- Compute and Plot the decision boundary ---------

draw_decision_boundary(x, f_s)


# more epochs, lower lr, introduce patient, change a to steeper value
# set hyperparameters

print("Test 4")
print('='*20)
epochs = 500
eta = 0.001
a = 7
mse_list = []
f_s= np.zeros((npoints, 1))
T = 5
count = 0

W_new = W
U_new = U
b_new = b
c_new = c

mse_old = float('inf') # initialize error
for epoch in range(epochs):
    mse_epoch = 0

    for i in range(x.shape[0]):
        inputs = np.expand_dims(x[i, :], axis= 0) # (1, 2)
        target = y[i]

        # Forward Pass

        vz, z, vf, f = forward_pass(inputs, W_new, b_new, U_new, c_new, a)
        f_s[i, 0] = f

        # Backward Pass

        dW, dU, db, dc = backward_pass(inputs, f, target, vf, z, vz, W_new, U_new, a)

        # Gradient Descend

        W_new = gradient_descend(W_new, dW, eta) 
        U_new = gradient_descend(U_new, dU, eta)
        b_new = gradient_descend(b_new, db, eta)
        c_new = gradient_descend(c_new, dc, eta)

        mse_epoch += compute_mse(target, f)
    
    # at the end of each epoch compute mse
    mse_epoch /= x.shape[0]
    mse_list.append(mse_epoch)
    if mse_epoch > mse_old:
        count += 1
    mse_old = mse_epoch

    if count >= T:
        print(f"Not improving: reducing learning rate to {eta*0.9}")
        eta *= 0.9
    print(f"Epoch {epoch+1}/{epochs}, MSE: {mse_epoch}")


plt.figure(figsize = (10, 8))
plt.plot(range(epochs), mse_list, linewidth = 1.5, c = 'b')
plt.title('MSE over epochs', fontsize = 20)
plt.xlabel('Epochs', fontsize = 18)
plt.ylabel('MSE', fontsize = 18)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(True)
plt.show()


# --------- Compute and Plot the decision boundary ---------

draw_decision_boundary(x, f_s)
