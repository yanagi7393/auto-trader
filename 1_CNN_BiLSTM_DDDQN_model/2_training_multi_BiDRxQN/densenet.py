
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layers
from tensorflow.contrib.layers import flatten
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope

def Batch_Normalization(x, training, S_batch_norm, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        
        if S_batch_norm == 'on':
            return tf.cond(training,
                           lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                           lambda : batch_norm(inputs=x, is_training=training, reuse=True))
        elif S_batch_norm == 'off':
            return x

def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride, padding='SAME')
        
        return network

def Global_Average_Pooling(x, stride=1):
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width, height]

    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride)

def Relu(x):
    return tf.nn.relu(x)

def Average_pooling(x, pool_size=[2,2], stride=2, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def Linear(x, output_size, scope) :
    return tf.layers.dense(inputs=x, units=output_size, name=scope+'linear')


class DenseNet(object):
    def __init__(self, x, output_size, nb_blocks, filters, training, S_batch_norm, scope):
        self.nb_blocks = nb_blocks
        self.filters = filters
        self.output_size = output_size
        self.training = training
        self.S_batch_norm = S_batch_norm
        self.model_output = self.Dense_net(x, scope)

    def bottleneck_layer(self, x, scope):
        with tf.name_scope(scope):
            x = Batch_Normalization(x, self.training, self.S_batch_norm, scope=scope+'_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=4 * self.filters, kernel=[1,1], layer_name=scope+'_conv1')
            
            x = Batch_Normalization(x, self.training, self.S_batch_norm, scope=scope+'_batch2')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[3,3], layer_name=scope+'_conv2')

            return x

    def transition_layer(self, x, scope):
        with tf.name_scope(scope):
            x = Batch_Normalization(x, self.training, self.S_batch_norm, scope=scope+'_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[1,1], layer_name=scope+'_conv1')
            x = Average_pooling(x, pool_size=[2,2], stride=2)

            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
        
            layers_concat = list()
            layers_concat.append(input_x)
            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))
            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = Concatenation(layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
                layers_concat.append(x)

            x = Concatenation(layers_concat)

            return x

    def Dense_net(self, input_x, scope):
        with tf.name_scope(scope):
            x = conv_layer(input_x, filter=self.filters, kernel=[3,3], stride=1, layer_name=scope+'conv0')

            x = self.dense_block(input_x=x, nb_layers=6, layer_name=scope+'dense_1')
            x = self.transition_layer(x, scope=scope+'trans_1')

            x = self.dense_block(input_x=x, nb_layers=6, layer_name=scope+'dense_2')
            x = self.transition_layer(x, scope=scope+'trans_2')

            x = self.dense_block(input_x=x, nb_layers=24, layer_name=scope+'dense_final')

            x = Batch_Normalization(x, self.training, self.S_batch_norm, scope=scope+'linear_batch')            
            x = Relu(x)
            x = Global_Average_Pooling(x)
            x = flatten(x)
            x = Linear(x, self.output_size, scope)

            return x

