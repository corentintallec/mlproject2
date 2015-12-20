import numpy as np

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import scipy.io

import random
from random import Random

import cPickle as pickle

#a couple of functions useful to train neural networks. When executed,
#this file performs standard nn training.

#the code is freely inspired from an Indico tutorial found at:
#

def onehot(x):
    """Transform a categorical values vector into a
    onehot matrix"""
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


def load_dataset(classification='mult'):
    """Open the file located at ../train/train.mat and extracts a tuple
    CNNDat,HogDat,FullDat,YDat"""
    train = scipy.io.loadmat('../train/train.mat')
    data = train['train'][0,0]
    xcnndat = data[0]
    xhogdat = data[1]
    ydat = data[2]
    if(classification=='mult'):
        ydat = onehot(ydat)
    else:
        ydat = ydat > 3
        ydat = onehot(ydat)
    xfulldat = np.concatenate((xhogdat, xcnndat),axis=1)
    return xcnndat,xhogdat,xfulldat,ydat

def shuffle_dataset(x_data,y_data):
    """Randomly shuffles the matrix data along the
    first axis.
    Returns the permutation seed and the shuffled dataset."""
    r = random.randint(1,100)
    Random(r).shuffle(x_data)
    Random(r).shuffle(y_data)
    return r,x_data,y_data

def prepare_plot(frames=3000):
    """Prepare a plot framework to plot train and test errors.
    The format of the plot is a full figure with 2 subplots, one
    showing the evolution of the train error, the other one showing
    the evolution of both the 01 loss and the BER"""
    font = {'family':'normal',
            'weight':'normal',
            'size':20}
    matplotlib.rc('font',**font)
    fig = plt.figure()

    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)

    ax1.set_xlim(-0.1,frames)
    ax1.set_ylim(-0.1,2)
    ax1.set_ylabel('Train log likelihood')
    ax1.set_title('Training curve')

    ax2.set_xlim(-0.1,frames)
    ax2.set_ylim(-0.1,1)
    ax2.set_xlabel('Number of batches')
    ax2.set_ylabel('Error percentage')
    ax2.set_title('Test curves')

    ltrain = ax1.plot([],[],lw=2,color='r')[0]
    l01test = ax2.plot([],[],lw=2,color='b')[0]
    lBERtest = ax2.plot([],[],lw=2,color='g')[0]

    def init():
        ltrain.set_data([],[])
        l01test.set_data([],[])
        lBERtest.set_data([],[])
        return [ltrain,l01test,lBERtest]

    return fig,ltrain,l01test,lBERtest,init

def get_features_number(use_data='full'):
    return {'cnn':36864,'hog':5408,'full':42273}.get(use_data,42273)

def pickle_params(params,suffix,output_dir='models/'):
    """Pickle parameters.
    params is a dictionary of string keys and theano tensor parameters
    suffix is the suffix to be appended to the file name
    output_dir is the directory in which to put the pickled files"""
    for n,p in params.items():
        pickle.dump(p.get_value(),open(output_dir+n+suffix,'wb'))

#The next bunch of functions are theano utilities

def floatX(X):
    """Make sure np arrays have the correct data format."""
    return np.asarray(X, dtype=theano.config.floatX)


def init_weights(shape):
    """Init a theano shared variable of shape shape, with
    independant centered gaussians of std 0.01"""
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))


def rectify(X):
    """Theano implementation of the rectifying function."""
    return T.maximum(X, 0.)


def softmax(X):
    """Theano stable implementation of the softmax function."""
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')


def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    """Theano implementation of the RMSprop gradient descent
    algorithm."""
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(floatX(p.get_value() * 0.))
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def train_cv_test_split(trp,cvp,data):
    """split data into train, cross validation and
    test set, with roughly trp % train data, cvp % cv
    data and the remaining as test data"""
    data_size = data.shape[0]
    train = data[0:int(np.floor(trp*data_size)),:]
    cv = data[int(np.floor(trp*data_size)):
                int(np.floor((trp+cvp)*data_size)),:]
    test = data[int(np.floor((trp+cvp)*data_size)):-1,:]
    return train,cv,test

def dropout(X, srng, p=0.):
    """Theano implementation of dropout."""
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def train(tparams,eparamstr,eparamste,model,classification='mult',
          output_dir='models/',pickle_period=100,cv_period=50,
          use_data='full',batch_size=20,trProp=0.6,cvProp=0.2,
          reg_lambda=0.01,lr=0.0001,frames=1000):
    """Trains the model defined by model and dinamically displays both train
    and test curves.
    params:
    ------
    tparams: a dictionary mapping variable names to theano tensors. These are
    the trainable parameters.
    eparamstr: a dictionary mapping variable names to variables. These are the
    extra parameters used during training (e.g. the dropout coefficients).
    eparamste: a dictionary mapping variable names to variables. These are the
    extra parameters used during test.
    modl: function outputing the model output theano tensor when given the
    theano input X as first argument, the training parameters (with names
    corresponding to those in tparams) and the extra parameters (with names
    corresponding to those in eparams).
    classification: 'mult' or 'bin'. Set to 'mult' for multiclass training, set
    to 'bin' for binary training.
    output_dir: directory in which the trained parameters will be pickled
    pickle_period: parameters are pickled periodically with period pickle_period
    cv_period: cross validation results are reported periodically with period
    cv_period
    use_data: 'full' for full dataset, 'cnn' for CNN features, 'hog' for hog
    features.
    batch_size: the model is learnt with mini-batch gradient descent, with
    batch_size examples in each batch.
    trPop: proportion of the data set used as training data.
    cvProp: proportion of the data set used as cv data.
    reg_lambda: unused at the moment.
    lr: learning rate.
    frames: number of batches before learning stops.
    """

    #loading, shuffling and splitting data set
    x_cnn,x_hog,x_full,y = load_dataset(classification)
    x = {
        'cnn':x_cnn[:,0:-1],
        'hog':x_hog,
        'full':x_full
    }.get(use_data,x_full)
    r,x,y = shuffle_dataset(x,y)
    data_size = x.shape[0]
    n_features = x.shape[1]
    n_class = y.shape[1]
    trx,cvx,tex = train_cv_test_split(trProp,cvProp,x)
    tryy,cvy,tey = train_cv_test_split(trProp,cvProp,y)
    cvyweight = np.sum((cvy/np.sum(cvy,axis=0)),axis=1)
    teyweight = np.sum((tey/np.sum(tey,axis=0)),axis=1)

    #usual theano stuff: converting the np.arrays to shared theano variables
    #for computational matters, we want theano tensors to be passed as argument
    #to theano functions (via givens) instead of np.array
    trX = theano.shared(floatX(trx))
    cvX = theano.shared(floatX(cvx))
    teX = theano.shared(floatX(tex))
    trY = theano.shared(floatX(tryy))
    cvY = theano.shared(floatX(cvy))
    teY = theano.shared(floatX(tey))
    cvwY = theano.shared(floatX(cvyweight))
    tewY = theano.shared(floatX(teyweight))

    n_batches = int(np.floor(trProp*data_size)) / batch_size

    index = T.lscalar()

    X = T.dmatrix()
    Y = T.dmatrix()
    wY = T.dvector()

    aparamstr = dict(tparams)
    aparamste = dict(tparams)
    aparamstr.update(eparamstr)
    aparamste.update(eparamste)

    #creating the computational graph
    tr_py_x = model(X,**aparamstr)
    py_x = model(X,**aparamste)
    y_x = T.argmax(py_x,axis=1)

    #defining costs and updates
    cost = T.mean(T.nnet.categorical_crossentropy(tr_py_x, Y))
    costBER = 1./n_class*(T.sum(wY*T.neq(T.argmax(Y, axis=1), y_x),axis=0))
    cost01 = T.mean(T.neq(T.argmax(Y,axis=1),y_x),axis=0)
    params=tparams.values()
    updates = RMSprop(cost,params,lr=lr)

    #functions to be compiled
    #the first one actually trains the models,
    #the others only report results and do not
    #change the parameters
    train = theano.function(inputs=[index],outputs=cost,updates=updates,
                            givens={
                                X: trX[index*batch_size:(index+1)*batch_size],
                                Y: trY[index*batch_size:(index+1)*batch_size]
                            })
    BERcv = theano.function(inputs=[],outputs=costBER,
                            givens={
                                X:cvX,
                                Y:cvY,
                                wY:cvwY
                            })
    BERte = theano.function(inputs=[],outputs=costBER,
                           givens={
                               X:teX,
                               Y:teY,
                               wY:tewY
                           })
    ZeroOnecv = theano.function(inputs=[],outputs=cost01,
                                givens={
                                    X:cvX,
                                    Y:cvY
                                })
    ZeroOnete = theano.function(inputs=[],outputs=cost01,
                                givens={
                                    X:teX,
                                    Y:teY
                                })

    #matplotlib animate stuffs
    fig,ltrain,lBER,l01,init = prepare_plot(frames)

    ctrs = []
    bers = []
    zos = []
    xs = []

    def animate(i):
        batch = i%n_batches
        ctrs.append(train(batch))
        ltrain.set_data(range(0,i+1),ctrs)
        if i%cv_period == 0:
            bers.append(BERcv())
            zos.append(ZeroOnecv())
            print 'BER: '
            print bers[-1]
            print '01Loss: '
            print zos[-1]
            if len(xs)>0:
                xs.append(xs[-1]+cv_period)
            else:
                xs.append(0)
            lBER.set_data(xs,bers)
            l01.set_data(xs,zos)
        if i%pickle_period == 0:
            pickle_params(tparams,str(i),output_dir)
        if i == frames-1:
            print 'BER Test cost: '
            print BERte()
            print '01Loss test cost: '
            print ZeroOnete()
        return [ltrain,lBER,l01]

    anim = animation.FuncAnimation(fig,animate,init_func=init,
                                   frames=frames,interval=20,repeat=False)
    plt.show()


if __name__=='__main__':
    #trains an actual model (3 layers standard nn with dropout, relUs and
    #RMSprop as learning algorithm)
    srng = RandomStreams()

    #'mult' for multiclass classification, 'bin' for binary classification
    tog = 'mult'
    n_class = {'bin':2,'mult':4}.get(tog,4)

    #actual model definition
    def model(X, w_h, b_h, w_h2, b_h2, w_o, b_o, p1, p2):
        X = dropout(X, srng, p1)
        h = rectify(T.dot(X, w_h)+b_h)

        h = dropout(h, srng, p2)
        h2 = rectify(T.dot(h, w_h2)+b_h2)

        h2 = dropout(h2, srng, p2)
        py_x = softmax(T.dot(h2, w_o)+b_o)
        return py_x

    n_feature = 42273

    #theano stuff (initializing shared variables)
    w_h = init_weights((42273, 400))
    b_h = init_weights((400,))
    w_h2 = init_weights((400, 200))
    b_h2 = init_weights((200,))
    w_o = init_weights((200, n_class))
    b_o = init_weights((n_class,))

    #dictionary to comply with the generic train function
    tparams={'w_h':w_h,'b_h':b_h,
            'w_h2':w_h2,'b_h2':b_h2,
            'w_o':w_o,'b_o':b_o}
    eparamstr={'p1':0.2,'p2':0.5}
    eparamste={'p1':0,'p2':0}

    #training and plotting
    train(tparams,eparamstr,eparamste,model,classification=tog,
          output_dir='models/')
