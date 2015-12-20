import theano.tensor as T
import theano
import numpy as np
import scipy.io
import cPickle as pickle
from random import Random
import random
import sys

def onehot(x):
    l = []
    d = {}
    n = 0
    k = 0
    for i in range(0, len(x)):
        if not x[i][0] in l:
            l.append(x[i][0])
            n += 1
    res = np.zeros((len(x), n))
    for i in range(0, len(x)):
        if not x[i][0] in d:
            v = np.zeros(n)
            v[k] = 1
            d[x[i][0]] = v
            k += 1
    for i in range(0, len(x)):
        res[i, :] = d[x[i][0]]
    return res


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def rectify(X):
    return T.maximum(X, 0.)

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def model(X, w_h, b_h, w_h2, b_h2, w_o, b_o):
    l1 = rectify(T.dot(X, w_h) +b_h)
    l2 = rectify(T.dot(l1, w_h2) + b_h2)
    py_x = softmax(T.dot(l2, w_o) + b_o)
    return l1,l2,py_x

if __name__=='__main__':
    r = 1
    output_dir = 'models/'
    train = scipy.io.loadmat('../train/train.mat')
    data = train['train'][0, 0]
    xcnndat = data[0]
    xhogdat = data[1]
    ydat = data[2]
    ydat = onehot(ydat)
    xhogdat = xhogdat - np.mean(xhogdat,axis=0)
    xcnndat = xcnndat - np.mean(xcnndat,axis=0)
    xfulldat = np.concatenate((xhogdat, xcnndat), axis=1)
    data_size = xcnndat.shape[0]
    n_class = ydat.shape[1]

    trProp = 0.8
    teProp = 0.1

    # shuffling dataset
    Random(r).shuffle(xcnndat)
    Random(r).shuffle(ydat)
    Random(r).shuffle(xhogdat)
    Random(r).shuffle(xfulldat)

    pre_xhog = xhogdat[int(np.floor((trProp+teProp)*data_size)):-1]
    pre_xcnn = xcnndat[int(np.floor((trProp+teProp)*data_size)):-1]
    pre_ydat = ydat[int(np.floor((trProp+teProp)*data_size)):-1]
    pre_xfulldat = xfulldat[int(np.floor((trProp+teProp)*data_size)):-1]
    y_weight =  np.sum((pre_ydat/np.sum(pre_ydat,axis=0)),axis=1)

    p1 = 0.2
    p2 = 0.5
    b_h0 = pickle.load(open(output_dir+'b_h0','rb'))
    w_h0 = pickle.load(open(output_dir+'w_h0','rb'))
    b_h20 = pickle.load(open(output_dir+'b_h20','rb'))
    w_h20 = pickle.load(open(output_dir+'w_h20','rb'))
    b_o0 = pickle.load(open(output_dir+'b_o0','rb'))
    w_o0 = pickle.load(open(output_dir+'w_o0','rb'))

    B_h = theano.shared(p1*floatX(b_h0))
    W_h = theano.shared(p1*floatX(w_h0))
    B_h2 = theano.shared(p1*p2*floatX(b_h20))
    W_h2 = theano.shared(p1*p2*floatX(w_h20))
    B_o = theano.shared(p2*floatX(b_o0))
    W_o = theano.shared(p2*floatX(w_o0))

    prX = theano.shared(floatX(pre_xfulldat))
    prY = theano.shared(floatX(pre_ydat))
    wY = theano.shared(floatX(y_weight))

    X = T.dmatrix()
    Y = T.dmatrix()

    h1,h2,py_x = model(X,W_h,B_h,W_h2,B_h2,W_o,B_o)
    y_x = T.argmax(py_x,axis=1)
    Cost = 1./n_class*(T.sum(wY*T.neq(T.argmax(Y, axis=1), y_x),axis=0))
    binCost = T.mean(T.neq(T.argmax(Y, axis=1),y_x))

    cost = theano.function(inputs=[],outputs=Cost,
                           givens={
                               X: prX,
                               Y: prY
                           })
    bincost = theano.function(inputs=[],outputs=binCost,
                              givens={
                                  X:prX,
                                  Y:prY
                              })

    print "Cost: "
    print cost()
    print "Bin cost: "
    print bincost()
