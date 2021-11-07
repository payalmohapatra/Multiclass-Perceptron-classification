# import custom library
import autograd.numpy as np
import autograd.numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
from IPython.display import display, HTML
import copy
import time
import math
import random
from autograd import grad
from autograd import hessian
import pandas as pd# max_iters and alpha
### Get data set :: Modify this code ##########################
datapath = 'path/'
# load in dataset
csvname = datapath + '3class_data.txt'
data = np.loadtxt(csvname,delimiter = ',')

# get input/output pairs
x = data[:-1,:]
y = data[-1:,:] 
## to drop NaN from the dataset
x = pd.DataFrame(data[:-1,:]) 
x = x.dropna()
x = pd.DataFrame.to_numpy(x)
# Normalise data
for i in range(np.size(x,0)):    
    row_mean = np.mean(x[i,:])
    row_std = np.std(x[i,:])
    x[i,:] = x[i,:]-row_mean/row_std 

arr_ones = [1] * np.size(x,1)
x = np.append([arr_ones], x, axis =0)
print(np.shape(x))
print(np.shape(y))

###### ML Algorithm functions ######
# learn all C separators
W = []
num_classes = np.size(np.unique(y)) 
np.random.seed(10)
w = np.random.randn(np.shape(x)[0],num_classes) * 1.0
def gradient_descent(g,alpha,max_its,w):
    gradient = grad(g)

    # run the gradient descent loop
    weight_history = [w]           # container for weight history
    cost_history = [g(w)]          # container for corresponding cost function history
    for k in range(max_its):
        # evaluate the gradient, store current weights and cost function value
        grad_eval = gradient(w)
        grad_eval_norm = grad_eval/(np .linalg .norm(w[1: ,:] ,'fro')**2)

        # take gradient descent step
        #w = w - alpha*grad_eval
        w = w - alpha*grad_eval_norm
        
        # record weight and cost
        weight_history.append(w)
        cost_history.append(g(w))
    return weight_history,cost_history

# compute C linear combinations of input point, one per classifier
def model(x,w):
    #a = w[0] + np .dot(x .T, w[1:])
    a = np .dot(x .T, w)
    return a. T

# multiclass perceptron
lam = 10 **-5 # our regularization parameter
def multiclass_perceptron(w):
    # pre-compute predictions on all points
    all_evals = model(x, w)
    # compute maximum across data points
    a = np. max(all_evals,axis = 0)
    # compute cost in compact form using numpy broadcasting
    b = all_evals[y. astype(int). flatten(),np .arange(np.size(y))]
    cost = np. sum(a - b)
    
    # add regularizer FIXME :: See if you need regulariser
    cost = cost + lam*np .linalg .norm(w[1: ,:] ,'fro')**2
    
    # return average
    return cost/ float(np.size(y))

# run random search algorithm 
alpha = 0.1
max_its = 2000
weight_history, cost_history = gradient_descent(multiclass_perceptron,alpha,max_its,w)

print(np.shape(weight_history[max_its]))
weight_class1 = weight_history[max_its][:,0]
weight_class2 = weight_history[max_its][:,1]
weight_class3 = weight_history[max_its][:,2]
x1_arr = x[1,:]
y_temp_1 = (-weight_class1[1]/weight_class1[2])*x1_arr - (weight_class1[0]/weight_class1[2])
y_temp_2 = (-weight_class2[1]/weight_class2[2])*x1_arr - (weight_class2[0]/weight_class2[2])
y_temp_3 = (-weight_class3[1]/weight_class3[2])*x1_arr - (weight_class3[0]/weight_class3[2])

## Plotting the classifiers
plt.plot(x1_arr,y_temp_1)
plt.plot(x1_arr,y_temp_2)
plt.plot(x1_arr,y_temp_3)


plt.scatter(x[1,:], x[2,:],
           linewidths=1, alpha=.7,
           edgecolor='k',
           s = 200,
           c=y) 

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[1,:], x[2,:], y,
           linewidths=1, alpha=.7,
           edgecolor='k',
           s = 200,
           c=y)
plt.plot(x1_arr,y_temp_1)
plt.plot(x1_arr,y_temp_2)
plt.plot(x1_arr,y_temp_3)
plt.show()

check_class = []
for i in range(y.size):
    dist = np.dot(x[:,i].T, weight_history[max_its])
    #print('these are the distance', dist)
    index = np.argmax(dist)
    check_class.append(index)
    #print('this is the index', index)
check_class = np.array(check_class).reshape([1, np.array(check_class).size])
num_of_misclassification = np.count_nonzero(check_class - y)
print('Total number of misclassifications are:', num_of_misclassification)
print('y_pred,',check_class)
print('y_actual,',y)

plt.figure(1)
cost_history_arr = np.asarray(cost_history)
cost_arr = []
cost_arr = np.reshape(cost_history_arr,(max_its+1,1))
plt.plot(cost_arr)
plt.title('Cost Function')
plt.xlabel('iterations')
plt.ylabel('cost')
plt.show()
