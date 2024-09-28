# =============================================== #
# ========= Importing Useful Libraries ========= #
# =============================================== #
import numpy as np
import matplotlib.pyplot as plt
import math

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


def gradient_descend(weights, f,df, eta):
    '''
    Function to implement one step of gradient descend
    :param weights: list of weights
    :param f: function of the weights
    '''
    # initialize list containing the updated weights
    updated_weights = []
    for i in range(len(df)):
        updated_w = weights[i] - eta * df[i]
        updated_weights.append(updated_w)
    
    
    return updated_weights



# ================================ #
# ========= Point (1b) ========= #
# ================================ #

# ========= Implement Gradient Descend ========= #
# randomly initialize weights

w1 = np.random.randint(1,2)
w2 = np.random.randint(1,2)
R = 13 * w1**2 - 10 * w1 * w2 + 4 * w1 + 2 * w2**2 - 2 * w2 + 1
dR = [26*w1 - 10 + 4, -10*w1 + 4*w2 -2]
dR = np.asarray(dR)


etas = [0.02, 0.05, 0.1]
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
        weights = gradient_descend([w1_upd, w2_upd], R, dR, eta)
        w1_upd = weights[0]
        w2_upd = weights[1]
        
        dist = compute_distance(w1_true, w2_true, w1_upd, w2_upd)
        distances[eta].append(dist)  
        print('=' * 20)
        print(f'Iteration {it+1}, eta={eta}')
        print(f'Distance: {dist}')
        print(f'w1 and w2: {w1_upd}, {w2_upd}')
   

# ========= Plot Results ========= #
colors = ['r', 'b', 'm']
_, axes= plt.subplots(nrows = 1, ncols = len(etas), figsize = (6, 12))
for ax, eta, color in zip(axes.ravel(), etas, colors):
    ax.plot([dist for dist in distances[eta]], linewidth = 1.5, color = color)
    ax.set_xlabel('N° iterations', fontsize = 16)
    ax.set_ylabel('Distance', fontsize = 16)
    ax.set_title(f'Distance for $\eta$ = {eta}', fontsize = 18)
    ax.grid(True)

plt.tight_layout()
plt.show()

