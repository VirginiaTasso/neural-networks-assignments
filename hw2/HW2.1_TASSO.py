# --------- import useful libraries ---------

import itertools 
import numpy as np
import pandas as pd

# --------- neural network parameters -------

w = np.array([[1, 1, -1], [0, -1, 1]]) # weights matrix
b = np.array([-2.5, -1.5])
u = np.array([1, 1])
c = 0.5


# --------- define useful functions ---------
# function to implement logic gate

def logic_gate(inputs):
    x1, x2, x3 = inputs
    # need to manage the fact that inputs can be -1. The logic gate works with 1 and 0 --> conversion of the values
    x1, x2, x3 = (0 if x == -1 else x for x in inputs)
    return bool(x1 and x2 and not x3) or bool(not x2 and x3)


# function to implement the analitic expression

def f_x(inputs):
    x1, x2, x3 = inputs
    inputs = np.array([x1, x2, x3])
    first = np.add(np.matmul(w, inputs),b) 
    first = np.sign(first)
    second = np.add(np.matmul(u, first), c)
    res = int(np.sign(second))

    if res == 1: return True
    else: return False




perms = list(itertools.product([1,-1], repeat = 3))
perms = np.asarray(perms)


# --------- build the table resulting from the analitic expression and from the logic gate---------

print ("\t x1 | x2 | x3 | output\n---------------------------------")

data_1 = [] # data with the outputs of the analytic expression
data_2 = [] # data with the outputs of the logic gate
for perm in perms:
    x1, x2,x3 = perm
    y1 = f_x(np.array([x1, x2, x3]))  # analitic expression
    y2 = logic_gate(np.array([x1, x2, x3])) # logic gate
    print(f"\t{x1} | {x2} | {x3} | {y1} ")
    data_1.append([x1, x2, x3, y1])
    data_2.append([x1, x2, x3, y2])

df1 = pd.DataFrame(data_1, columns=['x1', 'x2', 'x3', 'Output'])
df2 = pd.DataFrame(data_2, columns=['x1', 'x2', 'x3', 'Output'])

# check if the two tables match

print(df1['Output'].equals(df2['Output']))


