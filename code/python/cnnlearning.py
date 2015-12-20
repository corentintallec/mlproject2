import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.nnet import conv2d

from utils import *

if __name__=='__main__':
    srng = RandomStreams()
    def model(X, w1, w2, w_o, p1, p2):
        """defines the convolutional model."""
        X = T.reshape(X,(T.shape(X)[0],1024,6,6))
        l1 = rectify(conv2d(X, w1))
        l1 = dropout(l1, p1)
        l2 = T.flatten(l1,outdim=2)
        l3 = rectify(T.dot(l2, w2))
        l3 = dropout(l3, p2)

        pyx = softmax(T.dot(l3, w_o))
        return pyx

    n_features = get_features_number(use_data='cnn')
    w1 = init_weights((256,1024,6,6))
    w2 = init_weights((256,32))
    w_o = init_weights((32,4))

    #definition of dictionaries to comply with the generic train function
    tparams={'w1':w1,'w2':w2,'w_o':w_o}
    eparamstr={'p1':0.5,'p2':0.5}
    eparamste={'p1':0,'p2':0}

    train(tparams,eparamstr,eparamste,model,use_data='cnn',
          output_dir='cnn_models/')
