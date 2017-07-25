import numpy as np
np.random.seed(1337) # for reproducibility
from keras import backend as K
from keras.engine.topology import Layer,InputSpec
from keras.layers.convolutional import Convolution1D, MaxPooling1D,Convolution2D, MaxPooling2D
import os
os.environ['THEANO_FLAGS'] = "floatX=float32"

class MaxPiecewisePooling1D(Layer):
    input_dim = 3
    def __init__(self, pool_length = 2, split_index = [0,0,0], stride = None, border_mode='valid',**kwargs):
        super(MaxPiecewisePooling1D, self).__init__(**kwargs)
        if stride is None:
                stride = pool_length
        self.pool_length = pool_length
        self.split_index = split_index
        self.stride = stride
        self.st = (self.stride,1)
        self.pool_size = (pool_length,1)
        assert border_mode in {'valid','same'}
        self.border_mode=border_mode
        self.input_spec = [InputSpec(ndim=3)]

    def get_output_shape_for(self,input_shape):
        length = self.conv_piecevise_length(input_shape[1],self.pool_length,self.border_mode, self.stride)
        return (input_shape[0],length,input_shape[2])


    def conv_piecevise_length(self, input_length, filter_size, border_mode, stride):
        if input_length is None:
            return None
        assert border_mode in {'same', 'valid'}
        if border_mode == 'same':
            output_length = input_length
        elif border_mode == 'valid':
            output_length = input_length - filter_size + 1
        return (output_length + stride - 1) // stride

    def call(self, x, mask=None):

        head_index, between_index , tail_index   = self.split_index
        p1 = x[:,:head_index,:]
        p2 = x[:,head_index:between_index,:]
        p3 = x[:,between_index:tail_index,:]

        z1 = self.one_max_pooling(p1)
        z2 = self.one_max_pooling(p2)
        z3 = self.one_max_pooling(p3)
        z  = K.concatenate([z1,z2,z3],axis=1)
        return z


    def one_max_pooling(self, x):
        x = K.expand_dims(x, -1)   # add dummy last dimension
        x = K.permute_dimensions(x, (0, 2, 1, 3))
        output = self._pooling_function(inputs=x, pool_size=self.pool_size,
                                        strides=self.st,
                                        border_mode=self.border_mode,
                                        dim_ordering='th')
        output = K.permute_dimensions(output, (0, 2, 1, 3))
        return K.squeeze(output, 3)

    def _pooling_function(self, inputs, pool_size, strides,
                          border_mode, dim_ordering):
        output = K.pool2d(inputs, pool_size, strides,
                          border_mode, dim_ordering, pool_mode='max')
        return output

    def get_config(self):
        config = {'stride': self.stride,
                  'split_index':self.split_index,
                  'pool_length': self.pool_length,
                  'border_mode': self.border_mode}
        base_config = super(MaxPiecewisePooling1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))





class ConvAttention(Layer):
    def __init__(self, attention_dim, **kwargs):
        self.attention_dim = attention_dim
        self.W__L = K.eye(self.attention_dim)
        super(ConvAttention,self).__init__(**kwargs)

    def build(self,input_shape):
        input_dim = input_shape[1]
        initial_weight_value = np.random.random((input_dim, self.attention_dim))
        self.U = K.variable(initial_weight_value)
        self.trainable_weights = [self.U]


    def softmax3D(self,x):
        shape = x.shape
        x = x.reshape((-1,shape[-1]))
        softmax_tensor = K.softmax(x)
        return softmax_tensor.reshape(shape)


    def call(self,x,mask = None):
        x_T = K.permute_dimensions(x, (0,2,1))
        G = K.dot(x_T, K.dot(self.U, self.W__L))
        A = self.softmax3D(G)
        return K.batch_dot(x,A)

    def get_output_shape_for(self,input_shape):
        return (input_shape[0],input_shape[1],self.attention_dim)


if __name__ == '__main__':
    da = np.random.rand(100,200,300)
    dy = np.random.rand(100)
    da_index = [10,100,200]
    batch_size = 25
    hidden_dims = 100
    dropout_prob = [0.5,0.25]
    val_size = 0.1
    nb_epoch = 5

    from keras.models import Sequential, Graph, model_from_json, Model
    from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge
    from keras.layers import Input, merge, Embedding
    from keras.layers.convolutional import Convolution1D, MaxPooling1D,Convolution2D, MaxPooling2D
    from keras.optimizers import SGD, RMSprop
    from keras.callbacks import ModelCheckpoint, Callback
    from keras.utils import np_utils

    sentence_input = Input(shape=(200,300),dtype='float32',name='sentence_input')

    cnn2 = [Convolution1D(nb_filter=35,
                     filter_length= 10,
                     border_mode='valid',
                     activation='relu',
                     subsample_length=1)(sentence_input) for fsz in [2,3]]
    pool2 = [MaxPiecewisePooling1D(pool_length=2,split_index=da_index)(item) for item in cnn2]
    #pool2 = [MaxPooling1D(pool_length=3)(item) for item in cnn2]
    flatten2 = [Flatten()(pool_node) for pool_node in pool2]
    merge_cnn2 = merge(flatten2,mode='concat')
    #x2 = Dense(hidden_dims,activation='relu')(merge_cnn2)
    #x3 = Dropout(dropout_prob[1])(x2)
    main_loss = Dense(1,activation='sigmoid',name='main_output')(merge_cnn2)

    model =  Model(input= sentence_input, output=main_loss)
    model.compile(optimizer='adadelta',loss='binary_crossentropy',metrics=['accuracy'])
    model.fit(da,dy,batch_size=batch_size,nb_epoch=nb_epoch,
                validation_split=val_size,
                verbose=2)
