# -*- coding: utf-8 -*-

#built-in/std
import numpy as np
import numpy.linalg as la
import math

def LCSubsequence(list1, list2, delta = 0):
    m = len(list1)
    n = len(list2)

    mat = [[0] * (n + 1) for row in range(m + 1)]

    for row in range(1, m + 1):
        for col in range(1, n + 1):
            if abs(list1[row - 1] - list2[col - 1]) <= delta:
                mat[row][col] = mat[row - 1][col - 1] + 1
            else:
                mat[row][col] = max(mat[row][col - 1], mat[row - 1][col])

    return mat[m][n]

def LCSubstring(list1, list2, delta = 0):
    m = len(list1)
    n = len(list2)

    mat = [[0] * (n + 1) for row in range(m + 1)]

    for row in range(1, m + 1):
        for col in range(1, n + 1):
            if abs(list1[row - 1] - list2[col - 1]) <= delta:
                mat[row][col] = mat[row - 1][col - 1] + 1

    max_len = 0
    for row in range(1, m + 1):
        for col in range(1, n + 1):
            if mat[row][col] > max_len:
                max_len = mat[row][col]

    return max_len

def equalize(s,t,default=None):
    # equalize the length of two strings by appending a default value

    if len(s) < len(t):
        s.extend([default] * (len(t) - len(s)))
    elif len(t) < len(s):
        t.extend([default] * (len(s) - len(t)))
    else:
        pass # same length

    return s,t

def non_dtw_distance(s,t,default=None,costf=None):
    # Don't run dynamic time warping, instead just compare actions
    # using the provided cost function and post-pend to make the
    # sequences the same length.

    s,t = equalize(s,t,default)

    return sum([costf(a,b) for a,b in zip(s,t)])

def euclid_distance(s, t):

    dist = sum([abs(a - b)**2 for a,b in zip(s,t)])

    return math.sqrt(dist)

def regular_euclid_distance(s, t):

    dist = sum([abs(a - b)**2 for a,b in zip(s,t)])
    dist = dist/len(s)

    return math.sqrt(dist)

def initialize_dmatrix(rows,cols,window):
    d = np.zeros((rows,cols),dtype='float')

    d[:,0] = 1e6
    d[0,:] = 1e6
    d[0,0] = 0

    for i in range(1,rows):
        for j in range(1,cols):
            if abs(i - j) > window:
                d[i][j] = 1e6

    return d

def initialize_ematrix(rows,cols):
    d = np.zeros((rows,cols),dtype='float')

    d[:,0] = np.arange(rows)
    d[0,:] = np.arange(cols)

    return d

def initialize_smatrix(rows,cols):
    d = np.zeros((rows,cols),dtype='float')

    d[:,0] = -1e6
    d[0,:] = -1e6
    d[0,0] = 0

    return d

def edit_distance(s,t,delta):
    n = len(s)
    m = len(t)
    d = initialize_ematrix(n+1,m+1)

    for i in range(1,n+1):
        for j in range(1,m+1):
            if abs(s[i-1]-t[j-1]) < delta:
                dis = 0
            else:
                dis = 1
            d[i,j] = min(d[i-1,j] + 1, d[i,j-1] + 1, d[i-1,j-1] + dis)

    return d[n,m]

def erp_distance(s,t,g = 0):
    n = len(s)
    m = len(t)
    d = initialize_ematrix(n+1,m+1)

    for i in range(1,n+1):
        for j in range(1,m+1):
            d[i,j] = min(d[i-1,j] + abs(s[i-1]), d[i,j-1] + abs(t[j-1]), d[i-1,j-1] + abs(s[i-1]-t[j-1]))

    return d[n,m]

def edit_distance_vc(s,t, costs = (1,1,1)):
    # edit distance with variable costs
    n = len(s)
    m = len(t)
    d = initialize_ematrix(n+1,m+1)

    for i in range(1,n+1):
        for j in range(1,m+1):
            if s[i-1] == t[j-1]:
                d[i,j] = d[i-1,j-1]
            else:
                d[i,j] = min(d[i-1,j] + costs[0], d[i,j-1] + costs[1], d[i-1,j-1] + costs[2])

    return d[n,m]



def etw_distance(list1, list2, params, costf=lambda x,y: la.norm(x - y)):
    """
    etw_distance : extended time warping
    Use dynamic time warping but apply a cost to (insertion, deletion, match)
    """

    n = len(list1)
    m = len(list2)
    dtw = initialize_dmatrix(n+1,m+1)

    icost = params[0]
    dcost = params[1]
    mcost = params[2]

    for (i,x) in enumerate(list1):
        i += 1
        for (j,y) in enumerate(list2):
            j += 1

            cost = costf(x,y)
            dtw[i,j] = cost + min(dtw[i-1,j] + icost, dtw[i,j-1] + dcost, dtw[i-1][j-1] + mcost)

    return dtw[n,m]

def dtw_distance(list1, list2, costf=lambda x,y: la.norm(x - y) ):

    n = len(list1)
    m = len(list2)
    dtw = initialize_dmatrix(n+1,m+1)

    for (i,x) in enumerate(list1):
        i += 1
        for (j,y) in enumerate(list2):
            j += 1

            cost = costf(x,y)
            dtw[i,j] = cost + min(dtw[i-1,j],dtw[i,j-1],dtw[i-1][j-1])

    return dtw[n,m]

# def ddtw_distance(list1, list2, simf=lambda x,y: cosine(x,y)):
#
#     list1 = derivative(list1)
#     list2 = derivative(list2)
#
#     n = len(list1)
#     m = len(list2)
#     dtw = initialize_smatrix(n+1,m+1)
#
#     for (i,x) in enumerate(list1):
#         i += 1
#         for (j,y) in enumerate(list2):
#             j += 1
#
#             sim = simf(x,y)
#             if i == 1 and j == 1:
#                 dtw[i,j] = 2 * sim + dtw[i-1][j-1]
#             elif i == 1:
#                 dtw[i,j] = dtw[i][j-1] + sim
#             elif j == 1:
#                 dtw[i,j] = dtw[i-1][j] + sim
#             else:
#                 dtw[i,j] = max(dtw[i-1][j-1] + 2 * sim, dtw[i-2][j-1] + 2 * simf(list1[i-2],list2[j-1]) + sim, dtw[i-1][j-2] + 2 * simf(list1[i-1],list2[j-2]) + sim)
#
#     return dtw[n,m]

def ddtw_wdistance(list1, list2, w, symmetry = False, smooth = False, costf=lambda x,y: la.norm(x - y)):
    if symmetry == True:
        list1 = abs_list(list1)
        list2 = abs_list(list2)

    if smooth == False:
        list1 = derivative(list1)
        list2 = derivative(list2)
    elif smooth == True:
        list1 = smooth_derivative(list1)
        list2 = smooth_derivative(list2)

    n = len(list1)
    m = len(list2)
    dtw = initialize_dmatrix(n+1,m+1,w)

    for (i,x) in enumerate(list1):
        i = i + 1
        for j in range(max(0,i-w-1),min(m,i+w)):
            y = list2[j]
            j = j + 1
            #print("(" + str(i) + "," + str(j) + ")")
            cost = costf(x,y)
            dtw[i,j] = cost + min(dtw[i-1,j],dtw[i,j-1],dtw[i-1][j-1])

    return dtw[n,m]


def dtw_wdistance(list1, list2, w, symmetry = False, costf=lambda x,y: la.norm(x - y)):
    if symmetry == True:
        list1 = abs_list(list1)
        list2 = abs_list(list2)

    n = len(list1)
    m = len(list2)
    dtw = initialize_dmatrix(n+1,m+1,w)

    for (i,x) in enumerate(list1):
        i = i + 1
        for j in range(max(0,i-w-1),min(m,i+w)):
            y = list2[j]
            j = j + 1
            #print("(" + str(i) + "," + str(j) + ")")
            cost = costf(x,y)
            dtw[i,j] = cost + min(dtw[i-1,j],dtw[i,j-1],dtw[i-1][j-1])

    return dtw[n,m]

def derivative(list1):
    der_list = []
    for i in range(1, len(list1)):
        der = list1[i] - list1[i-1]
        der_list.append(der)

    return der_list

def smooth_derivative(list1):
    der_list = []
    for i in range(1, len(list1) - 1):
        der = list1[i] - list1[i-1] + (list1[i+1] - list1[i-1]) / 2
        der = der / 2
        der_list.append(der)

    return der_list

def abs_list(list1):
    for i in range(len(list1)):
        list1[i] = abs(list1[i])

    return list1


def cosine(y1, y2, x = 3):

    if y1 * y2 < 0:
        cos_sim = 0
    else:
        cos_sim = (y1 * y2 + x ** 2) / math.sqrt((y1 ** 2 + x ** 2) * (y2 ** 2 + x ** 2))

    return cos_sim

def main():
    list1 = [3,1]
    list2 = [3,3]
    print(euclid_distance(list1, list2))

if __name__ == "__main__":
    main()
