# discretization points method version 5: L2 Norm with symbolic gradient descent and ReLU.
# u(x, y) = x(1-x)y(1-y)
# u' w.r.t. x = (1-2x)y(1-y); u' w.r.t. y = x(1-x)(1-2y)
# New test function!
# For each H in {64, 128}, run 5 iterations to take the minimum error.
# Modified initialization method: each line intersect the domain.
# 30,000 itr and 50 discretization points
import numpy as np 
import numpy.random as npr
from matplotlib import pyplot as plt
import time
import os.path


def ReLU(x):
    result = np.where(x > 0, x, 0)
    return result

def ReLUFirstDerivative(x):
    dx = np.ones_like(x)
    dx[x <= 0] = 0
    return dx

def NN(W, x, y):
    XY = np.asarray([x, y])
    z = np.dot(XY, W[0]) + W[2]
    a = ReLU(z)
    N = np.dot(a, W[1])
    return N

# 1st derivative of NN
def NNDerivative(W, x, y):
    XY = np.asarray([x, y])
    VW = W[1].T * W[0]
    z = np.dot(XY, W[0]) + W[2]
    NNDeriv = np.dot(VW, ReLUFirstDerivative(z).T)

    NNDeriv_x = NNDeriv[0][0]
    NNDeriv_y = NNDeriv[1][0]
    result = np.asarray([NNDeriv_x, NNDeriv_y])
    return result

# the given function
def U(x, y):
    u = x * (1 - x) * y * (1 - y)
    return u

# the derivative of the given function
def UDerivative(x, y):
    u_x = (1 - 2*x) * y * (1 - y)
    u_y = x * (1 - x) * (1 - 2*y)
    uDerive = np.asarray([u_x, u_y]) 
    return uDerive

def difference_NN_U(W, x, y):
    nnOut = NN(W, x, y)[0][0]
    fOut = U(x, y)
    diff = nnOut - fOut
    return diff

# cost function for the known function
def costFunction(W, T, S):
    # error square sum of points inside the domain
    cost1 = 0.
    for ti in T:
        xi, yi = ti
        # the error sum of NN
        errSqr = (difference_NN_U(W, xi, yi))**2
        cost1 += errSqr
    # error square sum of points on the boundary
    cost2 = 0.
    for si in S:
        xi, yi = si
        # the error sum of NN
        errSqr = (difference_NN_U(W, xi, yi))**2
        cost2 += errSqr
    # cost for the trained NN
    cost = 0.5 * (cost1 + cost2)
    return cost

def partial_derivative_W_helper(W, xi, yi):
    w = W[0]
    v = W[1]
    b = W[2]
    # cost grad initilization
    cost_deriv = np.zeros(w.shape, dtype=np.float)
    # define input values
    inputs = np.asarray([[xi, yi]])
    x = np.asarray([xi, 0])
    y = np.asarray([0, yi])
    # z with respect to inputs x and y
    z = np.dot(inputs, w) + b
    # z with respect to input x or y
    z_x = np.dot(x, w) + b
    z_y = np.dot(y, w) + b

    # difference between NN result and U result
    diff = difference_NN_U(W, xi, yi)
    # matrix multiplication result
    matrix = v * ReLUFirstDerivative(z).T * inputs
    # error sum with respect to NN and original function (U)
    cost_deriv = diff * matrix.T
    return cost_deriv

# cost derivative with respect to weight (W[0])
def partial_derivative_W(W, T, S):
    # cost1: costs grad related to points in set T (within the boundary)
    cost_deriv_1 = np.zeros(W[0].shape, dtype=np.float)
    for ti in T:
        xi, yi = ti
        cost_deriv_1 += partial_derivative_W_helper(W, xi, yi)
    # cost2: costs grad related to points in set S (on the boundary)
    cost_deriv_2 = np.zeros(W[0].shape, dtype=np.float)
    for si in S:
        xi, yi = si
        cost_deriv_2 += partial_derivative_W_helper(W, xi, yi)
    result = cost_deriv_1 + cost_deriv_2
    return result

def partial_derivative_V_helper(W, xi, yi):
    w = W[0]
    v = W[1]
    b = W[2]
    # weights w.r.t. input x
    w_x = W[0][0]
    # weights w.r.t. input y
    w_y = W[0][1]
    # cost derivative initilization
    cost_deriv = np.zeros(v.shape, dtype=np.float)
    # define input values
    inputs = np.asarray([xi, yi])
    x = np.asarray([xi, 0])
    y = np.asarray([0, yi])

    # z with respect to inputs x and y
    z = np.dot(inputs, w) + b
    # z with respect to input x or y
    z_x = np.dot(x, w) + b
    z_y = np.dot(y, w) + b

    # error sum with respect to NN and original function (U)
    diff = difference_NN_U(W, xi, yi)
    cost_deriv = diff * ReLU(z).T
    return cost_deriv

# cost derivative with respect to V (W[1])
def partial_derivative_V(W, T, S):
    # cost1: costs grad related to points in set T (within the boundary)
    cost_deriv_1 = np.zeros(W[1].shape, dtype=np.float)
    for ti in T:
        xi, yi = ti
        cost_deriv_1 += partial_derivative_V_helper(W, xi, yi)
    # cost2: costs grad related to points in set S (on the boundary)
    cost_deriv_2 = np.zeros(W[1].shape, dtype=np.float)
    for si in S:
        xi, yi = si
        cost_deriv_2 += partial_derivative_V_helper(W, xi, yi)
    result = cost_deriv_1 + cost_deriv_2
    return result

def partial_derivative_B_helper(W, xi, yi):
    w = W[0]
    v = W[1]
    b = W[2]
    inputs = np.asarray([xi, yi])
    z = np.dot(inputs, w) + b
    # matrix multiplication result
    matrix = v.T * ReLUFirstDerivative(z)
    # difference between NN result and U result
    diff = difference_NN_U(W, xi, yi)
    # error sum with respect to NN and original function (U)
    cost_deriv = 2 * diff * matrix
    return cost_deriv

# cost derivative with respect to bias (W[2])
def partial_derivative_B(W, T, S):
    # cost1: costs grad related to points in set T (within the boundary)
    cost_deriv_1 = np.zeros(W[2].shape, dtype=np.float)
    for ti in T:
        xi, yi = ti
        cost_deriv_1 += partial_derivative_B_helper(W, xi, yi)
    # cost2: costs grad related to points in set S (on the boundary)
    cost_deriv_2 = np.zeros(W[2].shape, dtype=np.float)
    for si in S:
        xi, yi = si
        cost_deriv_2 += partial_derivative_B_helper(W, xi, yi)
    result = cost_deriv_1 + cost_deriv_2
    return result

def L2Norm(W, points, k):
    relErr = 0.
    for x, y in points:
        # numerator of relative error: N - u
        if k == 1:
            f = difference_NN_U(W, x, y)
        # denominator of relative error: u
        else:
            f = U(x, y)
        err = f**2
        relErr += err
    l2Norm = relErr**0.5
    return l2Norm

# relative error: L2 Norm
# P.S. the area in the formula are cancelled 
def errorEval(W, points):
    l2Norm_1 = L2Norm(W, points, 1) 
    l2Norm_2 = L2Norm(W, points, 0) 
    relErr_l2 = l2Norm_1 / l2Norm_2
    return relErr_l2

# create meshgrid coordinates
def generateCoordinates(xGrid, yGrid):
    # create  x and y axis
    X = np.linspace(0, 1, xGrid)
    Y = np.linspace(0, 1, yGrid)
    # generate meshgrid coordinates
    xv, yv = np.meshgrid(X, Y)
    xlist = xv.ravel()
    ylist = yv.ravel()
    coordinate = zip(xlist, ylist)
    # point coordinates list on the boundary
    S = []
    # point coordinates list within the boundary
    T = []
    for x, y in coordinate:
        point = (x, y)
        if x == 0 or x == 1 or y == 0 or y == 1:
            S.append(point)
        else:
            T.append(point)
    return S, T

# evaluate whether to terminate training 
def terminate_evaluation(relErr, idx, itr, tol):
    # stop: 0, do not stop training; stop: 1, stop training
    stop = 0
    window_size = 100
    # recent relative errors
    rre = np.zeros(window_size)
    # idx less than window size, keep training
    if (idx < window_size - 1):
        return stop
    # idx larger than window size 
    else:
        # take values inside window size 
        for i in range(window_size - 1, -1, -1):
            rre[i] = relErr[idx - i]
        rre = np.flip(rre)
        # minimum value in RRE
        alpha = min(rre)
        # maximum value in RRE
        beta = max(rre)
        # if the minimum value is 0, keep training
        if alpha == 0:
            stop = 0
        else:
            ratio = beta / alpha
            if ratio < 1 + tol:
                stop = 1
            print(ratio)
        return stop

def adam(theta, grad, m, v, t, alpha, beta1, beta2, eps=1e-8):
    # update biased first moment estimate
    m = beta1 * m + (1.0 - beta1) * grad
    # update biased second raw moment estimate
    v = beta2 * v + (1.0 - beta2) * grad**2
    # compute bias-corrected first moment estimate
    mhat = m / (1.0 - beta1**(t + 1))
    # compute bias-corrected second raw moment estimate
    vhat = v / (1.0 - beta2**(t + 1))
    # update parameters
    theta = theta - alpha * mhat / (np.sqrt(vhat) + eps)
    return theta, m, v

# train neural network
def trainNN(lmb, itr, tol, H, W, xGrid, yGrid, eval_xGrid, eval_yGrid):
    # S: list of points on the boundry; T: list of points within the boundary
    S, T = generateCoordinates(xGrid, yGrid)
    # chosen points coordinate list
    s_list, t_list = generateCoordinates(eval_xGrid, eval_yGrid)
    points = s_list + t_list
    # count iteration times
    idx = 0

    # initialize Adam's parameters
    # factor for average gradient
    beta1 = 0.9
    # factor for average squared gradient 
    beta2 = 0.999
    # first moment
    m_weights_x = np.zeros(W[0][0].shape)
    m_weights_y = np.zeros(W[0][1].shape)
    m_actions = np.zeros(W[1].shape)
    m_bias = np.zeros(W[2].shape)
    # second moment vector
    v_weights_x = np.zeros(W[0][0].shape)
    v_weights_y = np.zeros(W[0][1].shape)
    v_actions = np.zeros(W[1].shape)
    v_bias = np.zeros(W[2].shape)

    # record cost list
    costList = np.zeros(itr)
    costList[idx] = costFunction(W, T, S)
    # record relative error for Sobolev Norm at each iteration
    relErr = np.zeros(itr)
    relErr[idx] = errorEval(W, points)

    # start counting wall clock time for traning NN
    startWall = time.time()
    # start counting CPU clock time for traning NN
    startCPU = time.process_time()

    # if the relative error differences are less than tolerence, stop training
    stop = terminate_evaluation(relErr, idx, itr, tol)

    # calculate partial derivative and update W
    while (stop == 0 and idx < itr - 1):
        # symbolic gradient w.r.t. W, V and B
        costGrad_W = partial_derivative_W(W, T, S)
        costGrad_V = partial_derivative_V(W, T, S)
        costGrad_B = partial_derivative_B(W, T, S)
        # apply adam stocasticate gradient descent to update weights and biases
        weight_x, m_weights_x, v_weights_x = adam(W[0][0], costGrad_W[0], m_weights_x, v_weights_x, idx, lmb, beta1, beta2, eps=1e-8)
        weight_y, m_weights_y, v_weights_y = adam(W[0][1], costGrad_W[1], m_weights_y, v_weights_y, idx, lmb, beta1, beta2, eps=1e-8)
        actions, m_actions, v_actions = adam(W[1], costGrad_V, m_actions, v_actions, idx, lmb, beta1, beta2, eps=1e-8)
        bias, m_bias, v_bias = adam(W[2], costGrad_B, m_bias, v_bias, idx, lmb, beta1, beta2, eps=1e-8)
        # update weights and biases
        W[0][0] = weight_x
        W[0][1] = weight_y
        W[1] = actions
        W[2] = bias
        # renew and record values
        idx += 1
        costList[idx] = costFunction(W, T, S)
        relErr[idx] = errorEval(W, points)
        # if the relative error differences are less than tolerence, stop training
        stop = terminate_evaluation(relErr, idx, itr, tol)
        # print training process
        print("New initialization method! L2Norm! Only H = 64/128")
        print("H", H)
        print("iteration", idx)
        print("relErr", relErr[idx])
        print("cost", costList[idx])
        print("stop(ratio)", stop)

    # end counting CPU clock time for training NN
    endCPU = time.process_time()
    cpuTime = endCPU - startCPU
    # end counting wall clock time for traning NN
    endWall = time.time()
    wallTime = endWall - startWall

    return W, cpuTime, wallTime, relErr, idx, costList

def record_results(W, cpuTime, wallTime, relErr, relerr_min_list, itrTimes, costList, H, M, K):
    # record results in text files
    current_path = os.path.abspath(__file__)
    filePath = os.path.join(os.path.abspath(os.path.dirname(current_path) + os.path.sep + "."),'records')
    # filePath = 'C:/Users/DELL/Desktop/DT/codes/records'
    now = time.strftime("%m%d%H%M")
    fileName = 'records_L2_' + str(H) + '_' + now + '_v5.txt'
    fileDir = os.path.join(filePath, fileName)

    file = open(fileDir, 'w')

    file.write('Implement symbolic gradient descent with Adam. L2 norm! New test function and initialization method.' +
                '\nInvestigate the relation between H (2-128) and relative errors with 30,000 itr and 50 discretization points.' +
                '\nStop condition is ratio.' +
                '\nThe number of hidden nodes: ' + str(H) +
                '\nMinimum relative errors:' + str(relerr_min_list) +
                '\nIteration Times: ' + str(itrTimes + 1) +
                '\nThe number of points within the boundary M:' + str(M) +
                '\nThe number of points on the boundary K:' + str(K) +
                '\nThe number of grid points in x axis: ' + str(xGrid) +
                '\nThe number of grid points in y axis: ' + str(yGrid) +
                '\nLearning rate: ' + str(lmb) +
                '\nSetting Iteration number: ' + str(itr) +
                '\nThe weight from input unit to hidden unit: ' + str(W[0]) +
                '\nThe weight from hidden unit to the output: ' + str(W[1]) +
                '\nThe bias: ' + str(W[2]) +
                '\nWall time for training NN: ' + str(wallTime) +
                '\nCPU time for training NN: ' + str(cpuTime) +
                '\nCost value list: ' + str(costList) + 
                '\nRelative error for Soboler Norm: ' + str(relErr))

    file.close()

    # plot relative errors w.r.t. each iteration
    figPath = os.path.join(os.path.abspath(os.path.dirname(current_path) + os.path.sep + "."),'records')
    itrPlot = np.linspace(1, itr, itr)

    fig1_name = 'relative_error_L2_v5_' + str(H) + '.png'
    fig1_save = os.path.join(figPath, fig1_name)

    plt.figure()
    plt.plot(itrPlot, relErr, color = 'blue')
    plt.savefig(fig1_save)
    # plt.show()

    fig2_name = 'costs_L2_v5_' + str(H) + '.png'
    fig2_save = os.path.join(figPath, fig2_name)

    plt.figure()
    plt.plot(itrPlot, costList, color = 'green')
    plt.savefig(fig2_save)
    # plt.show()
    return 0

def initialization(H):
    # randomly initialize v and b for W
    W = [np.zeros((2, H)), npr.randn(H, 1), npr.randn(1, H)]
    # uniformly initialize points within the boundary
    beta = np.linspace(1, 0, H, endpoint=False)[::-1]
    gamma = np.linspace(0, 1, H)
    # randomly intialize w2
    w2 = npr.randn(1, H)
    # set bias to 1
    b = np.ones((1, H))
    # calculate w1 by solving the equation
    w1 = - (w2*gamma + b) / beta
    # assign values to w
    W[0][0] = w1
    W[0][1] = w2
    return W

# evaluate how the number of hidden nodes affect relative errors
def evaluate_H_relerror(lmb, itr, tol, xGrid, yGrid, eval_xGrid, eval_yGrid, M, K):
    # the number of hidden nodes needs to be evaluated
    H_num = 2
    # hidden nodes list 
    H_list = np.asarray([64, 128])
    # list containing minimum relative error for each H
    relerr_min_list = np.zeros(H_num)
    # H index
    i = 0
    for H in H_list:
        # initialize weight and bias: w: W[0], v: W[1], b: W[2]
        W = initialization(H)
        # train a neural network for 5 times
        relerr_min_list_tmp = np.zeros(5)
        for j in range(5):
            W, cpuTime, wallTime, relErr, itrTimes, costList = trainNN(lmb, itr, tol, H, W, xGrid, yGrid, eval_xGrid, eval_yGrid)    
            # append the minimum relative errors 
            relerr_min_list_tmp[j] = min(relErr[:(itrTimes + 1)])
        # append the minimum relative errors for this H
        relerr_min_list[i] = min(relerr_min_list_tmp)
        # record results in a file
        record_results(W, cpuTime, wallTime, relErr, relerr_min_list, itrTimes, costList, H, M, K)
        # print records on the command
        print("Investigate the relation between H (64,128) and relative errors with 30,000 itr.")
        print("L2 Norm, New test function, ReLU")
        print("New initialization method!")
        print("the number of hidden nodes", H)
        print("weight from input unit to hidden unit:", W[0])
        print("weight from hidden unit to the output:", W[1])
        print("bias:", W[2])
        print("wall time for training NN:", wallTime)
        print("CPU time for training NN:", cpuTime)
        print("iteration count:", itrTimes + 1)
        print("cost value at the final iteration:", costList[itrTimes])
        print("relative error for max norm at the final iteration:", relErr[itrTimes])
        print("the minimum relative errors for this H:", relerr_min_list[i])
        i += 1 
    return H_list, relerr_min_list


# number of points within the boundary
M = 2304 #3364
# number of points on the boundary
K = 196 #1196
#  number of grid points in x axis
xGrid = 50 # 300
#  number of grid points in y axis
yGrid = 50 # 300 
# number of evaluation grid point in x axis
eval_xGrid = 70
# number of evaluation grid point in x axis
eval_yGrid = 70
# leaning rate lambda
lmb = 0.001
# iteratio n setting
itr = 30000
# tolerence rate setting
tol = 0.001


# investigate how the number of hidden nodes affects relative errors
H_list, relerr_min_list = evaluate_H_relerror(lmb, itr, tol, xGrid, yGrid, eval_xGrid, eval_yGrid, M, K)

# record results in text files
current_path = os.path.abspath(__file__)
filePath = os.path.join(os.path.abspath(os.path.dirname(current_path) + os.path.sep + "."),'records')
# filePath = 'C:/Users/DELL/Desktop/DT/codes/records'
now = time.strftime("%m%d%H%M")
fileName = 'records_H_relErr_' + now + '_v5.txt'
fileDir = os.path.join(filePath, fileName)

file = open(fileDir, 'w')
file.write('hidden nodes list ' + str(H_list) + 
           '\nminimum relative errors for each H ' + str(relerr_min_list) +
           '\nnumber of dicretization points ' + str(xGrid))
file.close()