# version 3 - testing: choose smaller cpu time instead of smaller relative error
# testing: investigate H and xGrid with smaller data setting
# the plotting figure is kind of "ugly" 
# because the plotting points are too less to plot graph smothly
# !!! running result: H: 15, xGrid: 30, the minimum relative error: 0.01991266 !!!
import autograd.numpy as np
from autograd import grad 
import autograd.numpy.random as npr
from autograd.core import primitive
from matplotlib import pyplot as plt
import time
import os.path
from mpl_toolkits import mplot3d 
import matplotlib.pyplot as plt
from numba import jit

@jit(nopython=True)
def partOne(x):
    p1 = x + (1. + 3.*x**2) / (1. + x + x**3)
    return p1

@jit(nopython=True)
def partTwo(x):
    p2 = x**3 + 2.*x + x**2 * ((1. + 3.*x**2) / (1. + x + x**3))
    return p2

@jit()
def f(x, psy):
    fx = partTwo(x) - partOne(x) * psy
    return fx

@jit()
def analyticSol(x):
    sol = (np.exp((-x**2)/2.)) / (1. + x + x**3) + x**2
    return sol

@jit()
def sigmoid(x):
	sigFunc = 1 / (1 + np.exp(-x))
	return sigFunc

# first order derivative of sigmoid function
@jit()
def sigFirstDerivative(x):
    firstD = sigmoid(x) * (1 - sigmoid(x))
    return firstD

@jit()
def NN(W, x):
    z = np.dot(x, W[0]) + W[2]
    a1 = sigmoid(z)
    N = np.dot(a1, W[1])    # do we need bias for hidden layer to output layer? + b[1] 
    return N

# derivative of neural network; 1th order
@jit()
def NNDerivative(W, x):
    VW = np.dot(W[1].T, W[0].T)
    z = np.dot(x, W[0]) + W[2]
    NNDeriv = np.dot(VW, sigFirstDerivative(z))  # sigFirstDerivative(z); z = np.dot(x, W[0]) # do we need +b[1]?
    return NNDeriv

# cost function for a trial solution of the first derivative ODE
@jit()
def costFunction(W, x, ic):
    A = ic
    cost = 0.
    for xi in x:
        nnOut = NN(W, xi)[0][0]
        psy = A + xi * nnOut

        nnDerivOut = NNDerivative(W, xi)[0][0]
        psyDeriv = nnOut + xi * nnDerivOut

        func = f(xi, psy)       
        errSqr = (psyDeriv - func)**2

        cost += errSqr
    return cost

# function expression for result solution
@jit()
def resultSol(W, x):
    sol = 1 + x * NN(W, x)[0][0]
    return sol

# calculate the difference between result solution and analytic solution
@jit()
def funcDiff(W, x):
    yRes = resultSol(W, x)
    ySol = analyticSol(x)
    diff = yRes - ySol
    return diff

# calculate max norm for two functions within boundaries
@jit()
def maxNorm(funcDiff, a, b, numOfPoints, W):
    maxVal = 0
    X = np.linspace(a, b, numOfPoints)
    for x in X:
        maxVal = max(maxVal, np.abs(funcDiff(W, x)))
    return maxVal

# calculate max value for a function within boundaries 
@jit()   
def maxFunc(a, b, numOfPoints):
    maxVal = 0
    X = np.linspace(a, b, numOfPoints)
    for x in X:
        maxVal = max(maxVal, np.abs(analyticSol(x)))
    return maxVal

# calculate euclidean norm for two functions within boundaries
@jit()
def euclideanNorm(funcDiff, a, b, numOfPoints, W):
    culVal = 0
    X = np.linspace(a, b, numOfPoints)
    for x in X:
        diff = np.abs(funcDiff(W, x))
        culVal += diff**2
    eucliVal = np.sqrt(culVal)
    return eucliVal

# calculate euclidean value for a function within boundaries
@jit()
def euclideanFunc(a, b, numOfPoints):
    culVal = 0
    X = np.linspace(a, b, numOfPoints)
    for x in X:
        culVal += analyticSol(x)**2
    eucliVal = np.sqrt(culVal)
    return eucliVal

# relative error norm; k=1: max norm, otherwise: euclidean norm
@jit()
def errorEval(funcDiff, a, b, numOfPoints, W, k):
    if k == 1:
        maxN = maxNorm(funcDiff, a, b, numOfPoints, W)
        maxF = maxFunc(a, b, numOfPoints)
        relErrMax = maxN / maxF
        return relErrMax
    else:
        euclideanN = euclideanNorm(funcDiff, a, b, numOfPoints, W)
        euclideanF = euclideanFunc(a, b, numOfPoints)
        relErr2 = euclideanN / euclideanF
        return relErr2


# Investigating the effect of H and xGrid on the training
# for each initialization, train 20 times for W's random initialization and take the min rel error
lmb = 0.001 # learning rate
itr = 10000 # num of iterations - try 1000,000
tol = 0.05 # tolerance value - try 0.05

H = np.linspace(10, 30, 5, dtype = int) # Number of hidden nodes
xGrid = np.linspace(10, 30, 5, dtype = int) # number of discretization points

recordsArr = []

for i in range(len(H)):
    for j in range(len(xGrid)):
        records = []

        for k in range(20):
            X = np.linspace(0, 1, xGrid[j])
            W = [npr.randn(1, H[i]), npr.randn(H[i], 1), npr.randn(1, H[i])]  # bias: W[2]

            # start training NN
            idx = 0
            relErrMax = np.zeros(itr)
            relErrMax[idx] = errorEval(funcDiff, 0, 1, 100, W, 1)

            # start counting wall clock time for traning NN
            startWall = time.time()
            # start counting CPU clock time for traning NN
            startCPU = time.process_time()

            while not (relErrMax[idx] < tol) and idx < itr - 1:
                costGrad = grad(costFunction)(W, X, 1)
                W[0] = W[0] - lmb * costGrad[0]
                W[1] = W[1] - lmb * costGrad[1]
                W[2] = W[2] - lmb * costGrad[2]
                
                idx += 1
                relErrMax[idx] = errorEval(funcDiff, 0, 1, 100, W, 1)

            # end counting CPU clock time for training NN
            endCPU = time.process_time()
            cpuTime = endCPU - startCPU
            # end counting wall clock time for traning NN
            endWall = time.time()
            wallTime = endWall - startWall

            tmp = {'W': W, 'costValue': costGrad, 'relErrMax': relErrMax[idx], 'cpuTime': cpuTime, 'wallTime': wallTime}
            records.append(tmp)

        # take the min cpu time value for this time
        cpuTimeArr = [records[n]['cpuTime'] for n in range(len(records))]
        timeMin = min(cpuTimeArr)
        w = [records[n]['W'] for n in range(len(records)) if timeMin == records[n]['cpuTime']]
        cost = [records[n]['costValue'] for n in range(len(records)) if timeMin == records[n]['cpuTime']]
        relErrMax = [records[n]['relErrMax'] for n in range(len(records)) if timeMin == records[n]['cpuTime']]

        dictTmp = {'H': H[i], 'xGrid': xGrid[j], 'relErr': relErrMax[0], 'timeMin': timeMin, 'W': w[0], 'costValue': cost[0]}
        recordsArr.append(dictTmp)

print("records:", recordsArr)


# plot relative values change by each assigned H and xGrid
nPlot = np.linspace(1, 25, len(recordsArr))
nRes = [recordsArr[n]['relErr'] for n in range(len(recordsArr))]

plt.figure()
plt.plot(nPlot, nRes, color = 'green')
plt.show()

# plot 3d figure of each H and xGrid with corresponding relative error
# prepare arrays x: H, y: xGrid, z: relErr
x = np.outer(np.linspace(10, 30, 5, dtype = int), np.ones(5))
y = x.copy().T
relErrList = [recordsArr[n]['relErr'] for n in range(len(recordsArr))]
z = np.asarray(relErrList)
z.resize(5, 5)

print('x', x)
print('y', y)
print('z', z)

fig = plt.figure(figsize =(10, 6)) 
ax = plt.axes(projection ='3d')

surf = ax.plot_surface(x, y, z, cmap = plt.get_cmap('rainbow'), edgecolor ='none')  

ax.set_xlabel('H')
ax.set_ylabel('xGrid')
ax.set_zlabel('relative error')
  
# show plot 
plt.show()

# delete recordsArr array
del(recordsArr)