import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import scipy.io
import sys

srng = RandomStreams()


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


def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))


def rectify(X):
    return T.maximum(X, 0.)


def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')


def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
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


def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X


def model(X, w_h, b_h, w_h2, b_h2, w_o, b_o, p_drop_input, p_drop_hidden):
    X = dropout(X, p_drop_input)
    h = rectify(T.dot(X, w_h)+b_h)

    h = dropout(h, p_drop_hidden)
    h2 = rectify(T.dot(h, w_h2)+b_h2)

    h2 = dropout(h2, p_drop_hidden)
    py_x = softmax(T.dot(h2, w_o)+b_o)
    return h, h2, py_x


if __name__ == '__main__':
    # loading data from the .mat file
    train = scipy.io.loadmat('train/train.mat')
    data = train['train'][0, 0]
    xcnndat = data[0]
    xhogdat = data[1]
    ydat = data[2]
    ydat = onehot(ydat)
    xhogdat = xhogdat - np.mean(xhogdat,axis=0)
    xcnndat = xcnndat - np.mean(xcnndat,axis=0)
    xfulldat = np.concatenate((xhogdat, xcnndat), axis=1)

    # creating the corresponding tensors
    XCNNDat = theano.shared(floatX(xcnndat))
    XHogDat = theano.shared(floatX(xhogdat))
    XFullDat = theano.shared(floatX(xfulldat))
    YDat = theano.shared(floatX(ydat))

    data_size = xhogdat.shape[0]
    n_feature_cnn = xcnndat.shape[1]
    n_feature_hog = xhogdat.shape[1]
    n_feature_full = n_feature_cnn + n_feature_hog
    n_class = ydat.shape[1]
    ydat_i = np.sum(ydat/np.sum(ydat,axis=0),axis=0)
    print ydat_i[0:20]
    sys.exit()

    n_feature = n_feature_full

    batch_size = 20

    trProp = 0.8
    trX = XFullDat[0:int(np.floor(trProp*data_size))]
    teX = XFullDat[int(np.floor(trProp*data_size)+1):-1]
    trY = YDat[0:int(np.floor(trProp*data_size))]
    teY = YDat[int(np.floor(trProp*data_size)+1):-1]


    n_batches = int(np.floor(trProp*data_size)) / batch_size

    index = T.lscalar()

    X = T.dmatrix('X')
    Y = T.dmatrix('Y')

    w_h = init_weights((n_feature, 100))
    b_h = init_weights((100,))
    w_h2 = init_weights((100, 100))
    b_h2 = init_weights((100,))
    w_o = init_weights((100, n_class))
    b_o = init_weights((4,))

    p1 = 0.2
    p2 = 0.5
    reg_lambda = 0.01

    noise_h, noise_h2, noise_py_x = model(X, w_h, b_h, w_h2, b_h2, w_o, b_o, p1, p2)
    h, h2, py_x = model(X, p1*w_h, p1*b_h, p1*p2*w_h2, p1*p2*b_h2, p2*w_o, p2*b_o, 0., 0.)
    y_x = T.argmax(py_x, axis=1)

    regul = (w_h**2).sum(axis=1).mean(axis=0)+(w_h2**2).sum(axis=1).mean(axis=0)+(w_o**2).sum(axis=1).mean(axis=0)
    cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y)) + reg_lambda*regul
    teCost = T.mean(T.eq(T.argmax(Y, axis=1), y_x))
    params = [w_h, w_h2, w_o]
    updates = RMSprop(cost, params, lr=0.0001)

    train = theano.function(inputs=[index], outputs=cost, updates=updates,
                            givens={
                                X: trX[index*batch_size:(index+1)*batch_size],
                                Y: trY[index*batch_size:(index+1)*batch_size]
                            })
    predict = theano.function(inputs=[], outputs=teCost,
                              givens={
                                  X: teX,
                                  Y: teY
                              })

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 10000), ylim=(-0.1, 2))
    line, = ax.plot([], [], lw=2)
    line2, = ax.plot([], [], lw=2)

    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        line2.set_data([], [])
        return [line, line2]

    l = []
    l2 = []
    # animation function.  This is called sequentially
    def animate(i):
        batch = i%n_batches
        cost = train(batch)
        tecost = predict()
        print cost
        print tecost
        l.append(cost)
        l2.append(tecost)
        line.set_data(range(0, len(l)), l)
        line2.set_data(range(0, len(l)), l)
        return [line, line2]

    # call the animator.  blit=True means only re-draw the
    # parts that have changed.
    an = anim.FuncAnimation(fig, animate, init_func=init,
                                        frames=200, interval=20, blit=False)
    plt.show()
