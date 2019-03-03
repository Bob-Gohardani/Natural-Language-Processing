import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# c : vector of parameters for p0 and p1 in our linear function
# x : inputs
# y : output
def log_likelihood(c, x, y):
    p = 1 / (1 + np.exp(-(c[0]+c[1]*x)))
    # based on notes : Lj can be 1 or zero therefore Lj and 1-Lj are both one
    ll = np.sum(np.log(p[y==1])) + np.sum(np.log(1 - p[y==0]))
    return -ll
    
# sigmoid function
def sig(x, c):
    return 1 / (1 + np.exp(-(c[0]+c[1]*x)))

# training data
x_class_0 = np.random.normal(0, 1, 50)
x_class_1 = np.random.normal(0, 1, 50) + 1
x = np.concatenate([x_class_0, x_class_1])    

y_class_0 = np.zeros((50))
y_class_1 = np.ones((50))
y = np.concatenate([y_class_0, y_class_1])

yy = np.zeros((100))

colors = ['r', 'b']
f = lambda x: colors[int(x)] # based on value of x (0 x < 1 and 1 x> 1 map them to colors)
colors1 = list(map(f, y))

# maximum likelihood estimation:
# optimizing to find c0 and c1 parameters
start_params = np.array([1, 1])                                                 # when error is smaller than 1e-6 stop
res = minimize(log_likelihood, start_params, args=(x, y), method="BFGS", options={'gtol' : 1e-6 , 'disp' : False})

# test data and predictions:
x_test = np.random.normal(0, 1, 30) * 2
y_test = np.zeros((30))
                          # c0 and c1
predictions = sig(x_test, res.x)

f2 = lambda x: int(x > 0.5) # if x > 0.5 then turn it into 1 otherwise turn it into 0
predictions = list(map(f2, predictions))

c0 = res.x[0]
c1 = res.x[1]
# turn 0,1 into colors r, b 
colors2 = list(map(f, predictions))

print(colors2)