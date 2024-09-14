# --------- Importing useful libraries -------
import numpy as np
import matplotlib.pyplot as plt

# --------- Define activation function and neural network layers
def step_fun(x):
    return np.where(x >= 0, 1, -1)
    
def f_x(inputs, weights_1, b1, weights_2, b2):
    x1, x2 = inputs
    inputs = np.array([x1, x2]).reshape(-1,1)
    first = np.add(np.matmul(weights_1, inputs),b1) 
    first = step_fun(first)
    second = np.add(np.matmul(weights_2, first), b2 )
    res = step_fun(second)

    return res

# create 1000 random points

x = np.random.uniform(-2, 2, size=(1000, 2))

# --------- define neural network parameters ---------

w = np.array([[1, -1], [-1, -1], [0, -1]]) # weights matrix
b = np.array([[1], [1], [-1]])
u = np.array([1, 1, -1])
c = -1.5

x1_list = []
x2_list = []
colors = []

# --------- For each input (x1, x2), compute the output

for tpl in x:
    x1, x2 = tpl
    res = f_x(tpl, w, b, u, c)
    print(f'Result: {res}')
    x1_list.append(x1)
    x2_list.append(x2)
    if res == 1:
        colors.append('red')
    else:
        colors.append('blue')

# --------- create plot --------
plt.figure(figsize = (8,8))
plt.scatter(x1_list, x2_list, c = colors)
plt.grid()
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Points Scatterplot')

plt.scatter([], [], c='red', label='Result = 1')  
plt.scatter([], [], c='blue', label='Result = 0')
plt.legend()  # Add legend

#plt.show()
plt.savefig('scatterplot.png')

# --------- Compute and Plot the decision boundary ---------

# Generate a grid of points over the range [-2, 2]

x1 = np.linspace(-2, 2, 200)
x2 = np.linspace(-2, 2, 200)
xx, yy = np.meshgrid(x1, x2)

# Evaluate  nn for each point in the grid
zz = np.zeros(xx.shape)
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        point = np.array([xx[i, j], yy[i, j]])
        zz[i, j] = f_x(point, w, b, u, c)
plt.contourf(xx, yy, zz, cmap='coolwarm', alpha=0.3, levels=np.arange(-1, 2, 1))
plt.title('Decision Boundary')


plt.savefig('scatterplot_with_boundary.png')
#plt.show()