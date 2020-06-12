import tensorflow as tf
import numpy as np

def conv_bn_relu(inputs,depth,strides,kernel_size=(3,3),activation='relu',batch_norm=True,activate=True):
    '''
    Build a conv layer, followed by a BatchNormalization and an activation
    Parameters:
    inputs: inputs feature map
    depth: number of filters
    strides: strides in the convolution operator
    kernel_size: default, we set kernel_size=(3,3)
    activation: activation function, here, default we set relu
    batch_norm: decide to use batch norm, here, we set True
    activate: decide to use the activation function, here we set True
    '''
    x=tf.keras.layers.Conv2D(filters=depth,kernel_size=kernel_size,strides=strides,padding='same')(inputs)

    if(batch_norm==True):
        x=tf.keras.layers.BatchNormalization()(x)
    
    if(activate==True):
        x=tf.keras.layers.Activation(activation)(x)
    
    return x

def resblock(output_last_layer,depth,strides=1,first_layer=False):
    '''
    Build a resblock body
    Parameters:
    output_last_layer: the output of the last layer
    '''
    
    if(first_layer):
        x=conv_bn_relu(output_last_layer,depth,strides=2*strides)
        x=conv_bn_relu(x,depth,strides,activate=False)
        y=tf.keras.layers.Conv2D(depth,kernel_size=(3,3),strides=2*strides,padding='same')(output_last_layer)
    else:
        x=conv_bn_relu(output_last_layer,depth,strides)
        x=conv_bn_relu(x,depth,strides,activate=False)
        y=output_last_layer
    
    add=tf.keras.layers.add([y,x])
    outputs=tf.keras.layers.Activation('relu')(add)
    return outputs

def build_resnet(n,input_shape=(32,32,3)):
    '''
    Build resnet.
    Parameters:
    n: number of block per 1 resblock
    input_shape: shape of the input image
    '''
    inputs=tf.keras.Input(shape=input_shape)

    #The first convolution

    x=conv_bn_relu(inputs,depth=16,strides=1)
    for i in range(n):
        x=resblock(x,depth=16,strides=1,first_layer=False)
    depth=32
    for i in range(1,3):
        x=resblock(x,depth=depth*i,strides=1,first_layer=True)
        for j in range(n-1):
            x=resblock(x,depth*i,strides=1,first_layer=False)
    
    x=tf.keras.layers.AveragePooling2D(pool_size=8)(x)

    x=tf.keras.layers.Flatten()(x)

    outputs=tf.keras.layers.Dense(10,activation='softmax')(x)
    model=tf.keras.Model(inputs=inputs,outputs=outputs)

    return model

