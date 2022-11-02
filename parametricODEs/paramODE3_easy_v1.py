# parametric ODE3 easy version 1
# Initial Value Problem for System - first order
# solve system of k=2: y' = v, v' = -y
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

# derivative of neural network; 1st order
def NNDerivative(W, x, a1, a2):
    XA = np.asarray([x, a1, a2])
    VW = np.dot(W[1].T, W[0].T)
    z = np.dot(XA, W[0]) + W[2]
    NNDeriv = np.dot(VW.T, leakyReLU_Derivative(z)) 
    # print("nnDerivOut", NNDeriv)
    # print("nnDerivOut[0][0]", NNDeriv[0][0])
    return NNDeriv

def f1(alpha, beta, gamma):
    f1 = gamma
    return f1

def f2(alpha, beta, gamma):
    f2 = -beta
    return f2

# cost function for a trial solution of the first derivative ODE
def costFunction(W, X, A1, A2, x0):
    # extract W1 and W2
    W1 = [W[0], W[1], W[2]]
    W2 = [W[3], W[4], W[5]]
    # initialize cost
    cost = 0.
    # error square sum of differential equations
    # k: systems of k
    for k in range(0, 2):
        for xi in X:
            # variable a1
            for a1 in A1:
                # variable a2
                for a2 in A2:
                    # NN for one of the ODE function (k th) in the system: y or dydt
                    nnOut1 = NN(W1, xi, a1, a2)[0][0]
                    psy1 = a1 + (xi - x0)*nnOut1
                    # NN for the other ODE function in the system: y or dydt
                    nnOut2 = NN(W2, xi, a1, a2)[0][0]
                    psy2 = a2 + (xi - x0)*nnOut2
                    # k = 0, apply derivative for W1
                    if k == 0:
                        W = W1
                        nnOut = nnOut1
                    # k = 1, apply derivative for W2
                    else:
                        W = W2
                        nnOut = nnOut2
                    # calculate NN derivative for the k th function
                    nnDerivOut = NNDerivative(W, xi, a1, a2)[0][0]
                    psyDeriv = nnOut + (xi - x0)*nnDerivOut
                    
                    # sum up the errors for each k th
                    if k == 0:
                        f = f1(xi, psy1, psy2)
                    else:
                        f = f2(xi, psy1, psy2)
                    # sum up errors 
                    cost += (psyDeriv - f)**2

                # print("psy1:", psy1)
                # print("psy2:", psy2)
                # print("psyDeriv:", psyDeriv)
                # print("errSqr:", errSqr)
                # print("cost:", cost)
    return cost

def numericalSol(xGrid, aGrid, y_list):
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
    # print("numericalSol:", res_y_dict)
    return res_y_dict

# function expression for result solution
def resultSol(X, W, aGrid, res_a1_dict, res_a2_dict):
    # W1: weights for psy1, W2: weights for psy2
    W1 = [W[0], W[1], W[2]]
    W2 = [W[3], W[4], W[5]]
    # create dict to record data, key=1, value=y
    res_y_dict = {}
    # print("W1:", W1)
    # print("W2:", W2)
    # print("res_a1_dict",res_a1_dict)
    # print("res_a2_dict",res_a2_dict)
    # print("X",X)
    for j in range(1, aGrid+1):
        list_tmp = []
        for xi in X:
            val_tmp = res_a1_dict[j] + xi * NN(W1, xi, res_a1_dict[j], res_a2_dict[j])[0][0]
            # print("valtmp[0]",val_tmp[0])
            # print("valtmp", val_tmp)
            list_tmp.append(val_tmp)
        res_y_dict[j] = list_tmp
    # print("resultSol:", res_y_dict)
    return res_y_dict

# calculate the difference between result solution and analytic solution
def funcDiff(X, W, xGrid, aGrid, y_list, res_a1_dict, res_a2_dict):
    # result solution for NN
    yRes = resultSol(X, W, aGrid, res_a1_dict, res_a2_dict)
    # numerical solution from MATLAB
    ySol = numericalSol(xGrid, aGrid, y_list)
    # calculate solution difference and store it in dict
    diff = {}
    for i in range(1, aGrid+1):
        yRes_nparr = np.array(yRes[i])
        ySol_nparr = np.array(ySol[i])
        diff[i] = yRes_nparr - ySol_nparr
    # print("diff:", diff)
    return diff

# calculate max norm for two functions within boundaries
def maxNorm(X, xGrid, aGrid, W, y_list, res_a1_dict, res_a2_dict):
    maxVal = 0
    diff = funcDiff(X, W, xGrid, aGrid, y_list, res_a1_dict, res_a2_dict)
    # take max value for each assignment of a1 and a2
    for i in range(1, aGrid+1):
        maxVal = max(maxVal, max(np.abs(diff[i])))
    print("maxNorm:", maxVal)
    return maxVal

# calculate max value for a function within boundaries    
def maxFunc(xGrid, aGrid, y_list):
    maxVal = []
    res_y_dict = numericalSol(xGrid, aGrid, y_list)
    # take max value for each assignment of a1 and a2
    for i in range(1, aGrid+1):
        maxVal.append(max(np.abs(res_y_dict[i])))
    print("maxFunc:", max(maxVal))
    return max(maxVal)

# relative error norm; k=1: max norm, otherwise: euclidean norm
def errorEval(a, b, xGrid, aGrid, W, k, y_list, res_a1_dict, res_a2_dict):
    X = np.linspace(a, b, xGrid)
    if k == 1:
        maxN = maxNorm(X, xGrid, aGrid, W, y_list, res_a1_dict, res_a2_dict)
        maxF = maxFunc(xGrid, aGrid, y_list)
        relErrMax = maxN / maxF
        return relErrMax
    # else:
    #     euclideanN = euclideanNorm(a, b, xGrid, m, n, aGrid, W1, W2)
    #     euclideanF = euclideanFunc(a, b, xGrid, m, n, aGrid)
    #     relErr2 = euclideanN / euclideanF
    #     return relErr2

# train neural network
def trainNN(lmb, itr, tol, H, X, A1, A2, W, x0):
    # open records file of result y
    res_file = open("res_test_y_matlab.txt", "r")
    # read data line by line and record results in list
    y_list = []
    for line in res_file.readlines():
        y_tmp = float(line)
        y_list.append(y_tmp)
    # close file
    res_file.close()
    # open records file of result a1 assignment
    res_file_a1 = open("res_test_a1_matlab.txt", "r")
    # create dict to record a1 data, key=1, value=a1
    res_a1_dict = {}
    i = 1
    for line in res_file_a1.readlines():
        a1_tmp = float(line)
        res_a1_dict[i] = a1_tmp
        i += 1
    # close file
    res_file_a1.close()
    # open records file of result a1 assignment
    res_file_a2 = open("res_test_a2_matlab.txt", "r")
    # create dict to record a2 data, key=1, value=a1
    res_a2_dict = {}
    i = 1
    for line in res_file_a2.readlines():
        a2_tmp = float(line)
        res_a2_dict[i] = a2_tmp
        i += 1
    # close file
    res_file_a2.close()

    # apply L2 Normalization
    alpha = 0.5
    # count iteration times
    idx = 0
    # record relative error for Max Norm at each iteration
    relErrMax = np.zeros(itr)
    relErrMax[idx] = errorEval(0, 1, 100, 100, W, 1, y_list, res_a1_dict, res_a2_dict)
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
        costGrad = grad(costFunction, 0)(W, X, A1, A2, x0)
        # update W1
        W[0] = W[0]*(1 - lmb*alpha) - lmb * costGrad[0]
        W[1] = W[1]*(1 - lmb*alpha) - lmb * costGrad[1]
        W[2] = W[2]*(1 - lmb*alpha) - lmb * costGrad[2]
        W[3] = W[3]*(1 - lmb*alpha) - lmb * costGrad[3]
        W[4] = W[4]*(1 - lmb*alpha) - lmb * costGrad[4]
        W[5] = W[5]*(1 - lmb*alpha) - lmb * costGrad[5]
        
        idx += 1
        # relErrMax[idx] = errorEval(0, 20, 100, 0, 5, 100, W1, W2, 1)
        relErrMax[idx] = errorEval(0, 1, 100, 100, W, 1, y_list, res_a1_dict, res_a2_dict)
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


# discretization points for parameter x [0, 1]
xGrid = 10
X = np.linspace(0, 1, xGrid)
# discretization points for parameter a1 [0, 1]
a1Grid = 10
#A1 = np.linspace(0, 1, a1Grid)
###Amin
A1 = np.linspace( 0, 0.1, a1Grid)
# discretization points for parameter a2 [0, 1]
a2Grid = 2
#A2 = np.linspace(0, 1, a2Grid)
####Amin
A2 = np.linspace( 0.99,1,a2Grid)
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
W, cpuTime, wallTime, relErrMax, itrTimes = trainNN(lmb, itr, tol, H, X, A1, A2, W, 0)

# # record result solution
# resultArr = []
# for x in X:
#     for a in A:
#         resTmp = resultSol(X, W1, aGrid)
#         resultArr.append(resTmp)
# # record analytic solution
# analyticArr = []
# for x in X:
#     for a in A:
#         resTmp = numericalSol(xGrid, aGrid)
#         analyticArr.append(resTmp)

print("weight1 from input unit to hidden unit:", W[0])
print("weight1 from hidden unit to the output:", W[1])
print("bias1:", W[2])
print("weight2 from input unit to hidden unit:", W[3])
print("weight2 from hidden unit to the output:", W[4])
print("bias2:", W[5])
print("cost value:", costFunction(W, X, A1, A2, 0))
print("wall time for training NN:", wallTime)
print("CPU time for training NN:", cpuTime)
print("relative error for max norm at the final iteration:", relErrMax[itrTimes])
print("iteration count:", itrTimes + 1)
# print("result solution:", resultArr)
# print("analytic solution:", analyticArr)


# # For plotting, we can use as many points as we like
# numPlotPoints = 100

# # record results in text files
# filePath = 'C:/Users/DELL/Desktop/DT/codes/records'
# now = time.strftime("%m%d%H%M")
# fileName = 'record_' + now + '.txt'
# fileDir = os.path.join(filePath, fileName)

# file = open(fileDir, 'w')

# file.write('The number of discretization points for parameter X: ' + str(xGrid) +
#             '\nThe number of discretization points for parameter A: ' + str(aGrid) +
#             '\nThe number of hidden nodes: ' + str(H) +
#             '\nLearning rate: ' + str(lmb) +
#             '\nSetting Iteration number: ' + str(itr) +
#             '\nThe weight from input unit to hidden unit: ' + str(W[0]) +
#             '\nThe weight from hidden unit to the output: ' + str(W[1]) +
#             '\nThe bias: ' + str(W[2]) +
#             '\nCost value: ' + str(costFunction(W, X, A, 0)) + 
#             '\nWall time for training NN: ' + str(wallTime) +
#             '\nCPU time for training NN: ' + str(cpuTime) +
#             '\nThe number of plotting points: ' + str(numPlotPoints**2) +
#             '\nRelative error for max norm: ' + str(relErrMax) +
#             '\nIteration Times: ' + str(itrTimes+1) + 
#             '\nResult solution: ' + str(resultArr) +
#             '\nAnalytic solution: ' + str(analyticArr))

# file.close()


# # plot relative errors w.r.t. each iteration
# itrPlot = np.linspace(1, itr, itr)

# plt.figure()
# plt.plot(itrPlot, relErrMax, color = 'green')
# plt.savefig('C:/Users/DELL/Desktop/DT/codes/records/fig_relErr_paramODE1_euclidean.png')
# plt.show()

# # plot 3d figure of each H and xGrid with corresponding relative error
# # prepare arrays x: H, y: xGrid, z: relErr
# X = np.linspace(0, 1, numPlotPoints) # X
# Y = X.copy().T # A
# Z = np.zeros([numPlotPoints, numPlotPoints])
# for i in range(len(Y)):
#     for j in range(len(X)):
#         res = resultSol(W, X[j], Y[i])
#         Z[i][j] = res
# X, Y = np.meshgrid(X, Y)

# x = np.linspace(0, 1, numPlotPoints)
# y = x.copy().T
# z = np.zeros([numPlotPoints, numPlotPoints])
# for i in range(len(y)):
#     for j in range(len(x)):
#         res = analyticSol(x[j], y[i])
#         z[i][j] = res
# x, y = np.meshgrid(x, y)

# # plot figure 1
# fig = plt.figure(figsize =(10, 6)) 
# ax = fig.add_subplot(1,2,1, projection='3d')
# surf = ax.plot_surface(X, Y, Z, cmap = plt.get_cmap('rainbow'), edgecolor ='none') 
# # plot figure 2 
# ax2 = fig.add_subplot(1,2,2, projection='3d')
# surf2 = ax2.plot_surface(x, y, z, cmap = plt.get_cmap('rainbow'), edgecolor ='none')

# ax.set_xlabel('X')
# ax.set_ylabel('A')
# ax.set_zlabel('resultSol')

# ax2.set_xlabel('x')
# ax2.set_zlabel('analyticSol')

# fig.colorbar(surf, shrink=0.5, aspect=10)
# plt.savefig('C:/Users/DELL/Desktop/DT/codes/records/fig_3d_paramODE1_euclidean.png')
# # show plot 
# plt.show()
