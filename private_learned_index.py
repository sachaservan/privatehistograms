import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn import datasets, linear_model
import pandas as pd
from optparse import OptionParser

class Node:
    node_count = 0
    def __init__(self):
        Node.node_count += 1
        self.m = 0
        self.b = 0
        self.children = []
        self.max = 0
        self.min = 0

def lin_regression_with_sensitivity(X, Y):
    meanX = np.mean(X)
    meanY = np.mean(X)

    sumX = np.sum(X)
    sumY = np.sum(Y)

    max_num_term = 0
    max_num_term_index = 0
    min_num_term = 1000
    min_num_term_index = 0

    numerator = 0
    denom = 0
    for i in range(0,len(X)):
        term = (X[i] - meanX)*(Y[i] - meanY)
        
        if max_num_term < term: 
            max_num_term = term
            max_num_term_index = i 
        
        if max_num_term > term: 
            min_num_term = term
            min_num_term_index = i 

        numerator += term
        denom += (X[i] - meanX)**2

    m1 = 0 
    if denom != 0:
        m1 = numerator/denom

    index = min_num_term_index
    if max_num_term > max_denom_term:
        index = max_num_term_index

    meanX = (sumX - X[index])/(len(X) - 1)
    meanY = (sumY - Y[index])/(len(Y) - 1)
    numerator = 0
    denom = 0
    for i in range(0,len(X)):
        if (i == index):
            continue

        term = (X[i] - meanX)*(Y[i] - meanY)    
        numerator += term
        denom += (X[i] - meanX)**2

    m2 = 0
    if denom != 0:
        m2 = numerator/denom

    b1 = meanY - m1*meanX
    b2 = meanY - m2*meanX

    #return float(m), float(b)
    return abs(m2 - m1)

def build_recursive(X, Y, w, d, current_d):
    X = X.reshape(len(X),1)
    Y = Y.reshape(len(Y),1)
    reg = linear_model.LinearRegression()
    
    noise_m = 0
    noise_b = 0
    
    if len(X) > 1:

        reg = reg.fit(X, Y)
        init_m = reg.coef_[0][0]
        init_b = reg.intercept_[0]


    #     max_m = reg.coef_[0][0]
    #     for i in range(0,len(X)):
    #         xcpy = np.delete(X, [i])
    #         xcpy = xcpy.reshape(len(xcpy),1)
    #         ycpy = np.delete(Y, [i])
    #         ycpy = ycpy.reshape(len(ycpy),1)
            
    #         reg = reg.fit(xcpy, ycpy)            
    #         max_m = max(max_m, reg.coef_[0][0])

        xcpy = np.delete(X, [len(X)-1])
        xcpy = xcpy.reshape(len(xcpy),1)
        ycpy = np.delete(Y, [len(X)-1])
        ycpy = ycpy.reshape(len(ycpy),1)
        
        reg = reg.fit(xcpy, ycpy)
        m0 = reg.coef_[0][0]
        b0 = reg.intercept_[0]

        xcpy = np.delete(X, [0])
        xcpy = xcpy.reshape(len(xcpy),1)
        ycpy = np.delete(Y, [0])
        ycpy = ycpy.reshape(len(ycpy),1)
        
        reg = reg.fit(xcpy, ycpy)
        m1 = reg.coef_[0][0]
        b1 = reg.intercept_[0]

        sm = max(max(m0, m1), init_m) - min(min(m0,m1), init_m)
        sb = max(max(b0, b1), init_b) - min(min(b0,b1), init_b)
        noise_m = np.random.laplace(0, sm/0.1)
        noise_b = np.random.laplace(0, sb/0.1)

    print("noise m: " + str(noise_m) + "noise b: " + str(noise_b))

    reg.fit(X, Y)
    node = Node()   
    
    node.m = reg.coef_[0][0] + noise_m # slope
    node.b = reg.intercept_[0] + noise_b # intercept

    pred = X * node.m + node.b # gen prediction array using mx+b formula for Y values
    node.min = np.min(pred)
    node.max = np.max(pred)
    

    # if desired depth not reached
    if current_d != d and node.max - node.min > 0:
        bins = np.floor(((pred - node.min) / (node.max - node.min)) * (w-1))
        bins = bins.astype(int)
        for wi in range(w):
            mask = bins == wi # list of bins which equal wi
            Xwi = X[mask] # list of values that fall into the bin
            Ywi = Y[mask] 
            
            if len(Xwi) > 0 and len(Ywi) > 0:
                child_node = build_recursive(Xwi, Ywi, w, d, current_d + 1)
                node.children.append(child_node)
            else:
                node.children.append(None)
            
    return node    
    
def predict_recursive(x, w, d, node):
    pred = x * node.m + node.b 
    
    if len(node.children) > 0 and node.max - node.min > 0:
        bin = np.floor(((pred - node.min) / (node.max - node.min)) * (w-1))
        bin = bin.astype(int)
        if bin >= 0 and len(node.children) > bin and node.children[bin] != None:
            pred = predict_recursive(x, w, d, node.children[bin])
        
        
    return pred
            
if __name__ == "__main__": 
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="file", type="str", default="data_1.csv", help="data file")
    parser.add_option("-d", "--depth", dest="depth", type="int", default=2, help="depth of the model tree")
    parser.add_option("-w", "--width", dest="width", type="int", default=10, help="width of the model layers")

    (options, args) = parser.parse_args()

    # load csv and columns
    df = pd.read_csv(options.file)
    Y = df['pos']
    X = df['value']
    X=X.values.reshape(len(X),1)
    Y=Y.values.reshape(len(Y),1)
       
    # setup figures
    f1 = plt.figure()
    f2 = plt.figure()
    ax1 = f1.add_subplot(111)
    ax2 = f2.add_subplot(111)
    ax2.set_xlabel('age')
    ax2.set_ylabel('count')
    ax1.set_xlabel('age')
    ax1.set_ylabel('pos')
      
    # plot "true" cdf
    ax1.scatter(X, Y, color='g', alpha=0.5, s=4)
     
    # build learned index model
    d = options.depth # depth of the recursion
    w = options.width # width of the layers
    node = build_recursive(X, Y, w, d, 0)
    print("number of nodes in model = " + str(Node.node_count))

    # predict the cdf
    predictions = []
    testX = np.linspace(np.min(X), np.max(X), 10000)
    for i in range(len(testX)):
        pred = predict_recursive(testX[i], w, d, node)
        predictions.append(pred)
     
    # plot the predicted cdf
    ax1.plot(testX, predictions, color='blue',linewidth=1)

    # plot the histogram 
    n, bins, patches = ax2.hist(X, 20, facecolor='g', alpha=0.5)

    # predict the histogram
    counts = []
    predicted_hist = []
    for i in range(len(bins) - 1):
        lower = bins[i]
        upper = bins[i + 1]
        pred_upper = predict_recursive(upper, w, d, node)
        pred_lower = predict_recursive(lower, w, d, node)
        c = pred_upper - pred_lower
        
        predicted_hist.append(lower)
        predicted_hist.append(upper)
        counts.append(c)
        counts.append(c)
        print("count of " + str(lower) + " to " + str(upper) + " = " + str(c))
    ax2.plot(predicted_hist, counts, color='blue', linewidth=1)

    plt.show()