# parametric ODE2 Van Der Pol version 3
# Initial Value Problem for System - first order
# solve system of k=2: y' = v, v' = -y + Mu*(1 - y^2)*v   [Mu = 1]
# psy1 = a1j + (xi - x0)*nnOut1     [x0 = 0]
# psy2 = a2j + (xi - x0)*nnOut2     [x0 = 0]
# f_k(xi, psy1, psy2)   [k=2]
import autograd.numpy as np
from autograd import grad 
import autograd.numpy.random as npr
from autograd.core import primitive
from matplotlib import pyplot as plt
import time
import os.path

def leakyReLU(x, alpha=0.01):
    result = np.where(x > 0, x, x*alpha)
    return result

def leakyReLU_Derivative(x, alpha=0.01):
    dx = np.ones_like(x)
    dx[x < 0] = alpha
    return dx

def NN(W, x, a1, a2):
    XA = np.asarray([x, a1, a2])
    z = np.dot(XA, W[0]) + W[2]
    A1 = leakyReLU(z)
    N = np.dot(A1, W[1])    # do we need bias for hidden layer to output layer? + b[1] 
    return N

# derivative of neural network; 1th order
def NNDerivative(W, x, a1, a2):
    XA = np.asarray([x, a1, a2])
    VW = np.dot(W[1].T, W[0].T)
    z = np.dot(XA, W[0]) + W[2]
    NNDeriv = np.dot(VW.T, leakyReLU_Derivative(z)) 
    return NNDeriv

def f(x, psy1, psy2):
    Mu = 1
    f_x_psy1 = psy2
    f_x_psy2 = - psy1 + Mu*(1 - psy1**2)*psy2
    return f_x_psy1, f_x_psy2

# cost function for a trial solution of the first derivative ODE
def costFunction(W, X, A, x0):
    cost = 0.
    # error square sum of differential equations
    # k: systems of k
    for k in range(0, 2):
        for xi in X:
            aGrid = A[k].shape[0]
            # variable a1
            for m in range(0, aGrid):
                # variable a2
                for n in range(0, aGrid):
                    # k1: the current row index for A, either 1 or 0
                    k1 = k
                    # k2: the other row index for A, either 0 or 1
                    if k == 1:
                        k2 = 0
                        W1 = [W[3], W[4], W[5]]
                        W2 = [W[0], W[1], W[2]]
                    else:
                        k2 = 1
                        W1 = [W[0], W[1], W[2]]
                        W2 = [W[3], W[4], W[5]]
                    # NN for one of the ODE function (k th) in the system: y or dydt
                    nnOut1 = NN(W1, xi, A[k1][m], A[k2][n])[0][0]
                    psy1 = A[k1][m] + (xi - x0)*nnOut1
                    # NN for the other ODE function in the system: y or dydt
                    nnOut2 = NN(W2, xi, A[k1][m], A[k2][n])[0][0]
                    psy2 = A[k2][n] + (xi - x0)*nnOut2
                    # calculate NN derivative for the k th function
                    nnDerivOut = NNDerivative(W1, xi, A[k1][m], A[k2][n])[0][0]
                    psyDeriv = nnOut1 + (xi - x0)*nnDerivOut
                    # f_k(xi, psy1, psy2)
                    f_x_psy1, f_x_psy2 = f(xi, psy1, psy2)
                    # sum up the errors for each k th
                    if k == 0:
                        errSqr = (psyDeriv - f_x_psy1)**2
                    else:
                        errSqr = (psyDeriv - f_x_psy2)**2
                    # sum up errors 
                    cost += errSqr
    return cost

def numericalSol(xGrid, aGrid):
    # open records file of result y
    res_file = open("res_van_y_matlab.txt", "r")
    # read data line by line and record results in list
    y_list = []
    for line in res_file.readlines():
        y_tmp = float(line)
        y_list.append(y_tmp)
    # create dict to record data, key=1, value=y
    # each result(xGrid) for each assignment of a1(aGrid) and a2(aGrid)
    res_y_dict = {}
    for i in range(1, aGrid+1):
        # create temperary list to record one result
        y_tmp_list = []
        for j in range(1, xGrid+1):
            # keep tracking the index of y_list
            idx = (i - 1)*100 + j
            y_tmp_list.append(y_list[idx-1])
        # assign one result to dict with assignment of a1 and a2
        res_y_dict[i] = y_tmp_list
    # close file
    res_file.close()
    # print("numericalSol:", res_y_dict)
    return res_y_dict

# function expression for result solution
def resultSol(X, W, aGrid):
    W1 = [W[0], W[1], W[2]]
    W2 = [W[3], W[4], W[5]]
    # open records file of result a1 assignment
    res_file_a1 = open("res_van_a1_matlab.txt", "r")
    # create dict to record data, key=1, value=a1
    res_a1_dict = {}
    i = 1
    for line in res_file_a1.readlines():
        a1_tmp = float(line)
        res_a1_dict[i] = a1_tmp
        i += 1
    # close file
    res_file_a1.close()
    # open records file of result a1 assignment
    res_file_a2 = open("res_van_a2_matlab.txt", "r")
    # create dict to record data, key=1, value=a1
    res_a2_dict = {}
    i = 1
    for line in res_file_a2.readlines():
        a2_tmp = float(line)
        res_a2_dict[i] = a2_tmp
        i += 1
    # close file
    res_file_a2.close()
    # create dict to record data, key=1, value=y
    res_y_dict = {}
    print("W1:", W1)
    print("W2:", W2)
    # print("res_a1_dict",res_a1_dict)
    # print("res_a2_dict",res_a2_dict)
    # print("X",X)
    for j in range(1, aGrid+1):
        list_tmp = []
        for xi in X:
            val_tmp = res_a1_dict[j] + xi * NN(W1, xi, res_a1_dict[j], res_a2_dict[j])[0][0]
            list_tmp.append(val_tmp)
        res_y_dict[j] = list_tmp
    print("resultSol:", res_y_dict)
    return res_y_dict

# calculate the difference between result solution and analytic solution
def funcDiff(X, W, xGrid, aGrid):
    # result solution for NN
    yRes = resultSol(X, W, aGrid)
    # numerical solution from MATLAB
    ySol = numericalSol(xGrid, aGrid)
    # calculate solution difference and store it in dict
    diff = {}
    for i in range(1, aGrid+1):
        yRes_nparr = np.array(yRes[i])
        ySol_nparr = np.array(ySol[i])
        diff[i] = yRes_nparr - ySol_nparr
    print("diff:", diff)
    return diff

# calculate max norm for two functions within boundaries
def maxNorm(a, b, xGrid, aGrid, W):
    maxVal = 0
    X = np.linspace(a, b, xGrid)
    diff = funcDiff(X, W, xGrid, aGrid)
    # take max value for each assignment of a1 and a2
    for i in range(1, aGrid+1):
        maxVal = max(maxVal, max(np.abs(diff[i])))
    print("maxNorm:", maxVal)
    return maxVal

# calculate max value for a function within boundaries    
def maxFunc(xGrid, aGrid):
    maxVal = []
    res_y_dict = numericalSol(xGrid, aGrid)
    # take max value for each assignment of a1 and a2
    for i in range(1, aGrid+1):
        maxVal.append(max(np.abs(res_y_dict[i])))
    print("maxFunc:", max(maxVal))
    return max(maxVal)

# relative error norm; k=1: max norm, otherwise: euclidean norm
def errorEval(a, b, xGrid, m, n, aGrid, W, k):
    if k == 1:
        maxN = maxNorm(a, b, xGrid, aGrid, W)
        maxF = maxFunc(xGrid, aGrid)
        relErrMax = maxN / maxF
        return relErrMax
    else:
        euclideanN = euclideanNorm(a, b, xGrid, m, n, aGrid, W1, W2)
        euclideanF = euclideanFunc(a, b, xGrid, m, n, aGrid)
        relErr2 = euclideanN / euclideanF
        return relErr2

# train neural network
def trainNN(lmb, itr, tol, H, X, A, W, x0):
    alpha = 0.5
    # count iteration times
    idx = 0
    # record relative error for Max Norm at each iteration
    relErrMax = np.zeros(itr)
    relErrMax[idx] = errorEval(0, 20, 100, 0, 5, 100, W, 1)
    # relErrMax[idx] = errorEval(0, 20, 100, 0, 5, 100, W1, W2, 1)
    # print("W1:", W1)
    # print("W2:", W2)
    print(relErrMax[idx])
    # start counting wall clock time for traning NN
    startWall = time.time()
    # start counting CPU clock time for traning NN
    startCPU = time.process_time()
    # calculate partial derivative and update W
    while (not (relErrMax[idx] < tol)) and (idx < itr - 1):
        costGrad = grad(costFunction, 0)(W, X, A, x0)
        # update W1
        W[0] = W[0]*(1 - lmb*alpha) - lmb * costGrad[0]
        W[1] = W[1]*(1 - lmb*alpha) - lmb * costGrad[1]
        W[2] = W[2]*(1 - lmb*alpha) - lmb * costGrad[2]
        W[3] = W[3]*(1 - lmb*alpha) - lmb * costGrad[3]
        W[4] = W[4]*(1 - lmb*alpha) - lmb * costGrad[4]
        W[5] = W[5]*(1 - lmb*alpha) - lmb * costGrad[5]
        
        idx += 1
        # relErrMax[idx] = errorEval(0, 20, 100, 0, 5, 100, W1, W2, 1)
        relErrMax[idx] = errorEval(0, 20, 100, 0, 5, 100, W, 1)
        # print(costGrad1)
        print("costGrad",costGrad)
        print("relative error at final iteration",relErrMax[idx])
        # print("W1:", W1)
        # print("W2:", W2)

    # end counting CPU clock time for training NN
    endCPU = time.process_time()
    cpuTime = endCPU - startCPU
    # end counting wall clock time for traning NN
    endWall = time.time()
    wallTime = endWall - startWall

    return W, cpuTime, wallTime, relErrMax, idx


# discretization points for parameter x [0, 20]
xGrid = 10
X = np.linspace(0, 20, xGrid)
# discretization points for parameter A [0, 5]
aGrid = 10
A = [np.linspace(0, 5, aGrid), np.linspace(0, 5, aGrid)]
# number of hidden nodes
H = 10
# # initialize weight1 and bias: w1: W[0], v1: W[1], b1: W[2]
# W1 = [npr.randn(2, H), npr.randn(H, 1), npr.randn(1, H)]
# # initialize weight2 and bias: w2: W[0], v2: W[1], b2: W[2]
# W2 = [npr.randn(2, H), npr.randn(H, 1), npr.randn(1, H)]
# W1 and W2 
W = [npr.randn(3, H), npr.randn(H, 1), npr.randn(1, H), npr.randn(3, H), npr.randn(H, 1), npr.randn(1, H)]
# leaning rate lambda
lmb = 0.001
# iteratio n setting
itr = 1000
# tolerence rate setting
tol = 0.05

# train a neural network w.r.t. parameter X and A
W, cpuTime, wallTime, relErrMax, itrTimes = trainNN(lmb, itr, tol, H, X, A, W, 0)

print("weight1 from input unit to hidden unit:", W[0])
print("weight1 from hidden unit to the output:", W[1])
print("bias1:", W[2])
print("weight2 from input unit to hidden unit:", W[3])
print("weight2 from hidden unit to the output:", W[4])
print("bias2:", W[5])
print("cost value:", costFunction(W, X, A, 0))
print("wall time for training NN:", wallTime)
print("CPU time for training NN:", cpuTime)
print("relative error for max norm at the final iteration:", relErrMax[itrTimes])
print("iteration count:", itrTimes + 1)
