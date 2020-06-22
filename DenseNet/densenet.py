import tensorflow as tf
import numpy as np
import pickle

# This program is used for build DenseNet for training CIFAR-10

def bn_relu_conv(inputs,filters,kernel_size=(3,3),strides=1,batch_norm=True,activation='relu',activate=True):
    '''
    Construct a subsequent of BatchNormalization, Activation and Conv2D for DenseNet.
    Parameters:
    inputs: the input tensor
    filters: number of filter used in the convolutional operation
    kernel_size: kernel_size used in the convolutional operation
    strides: strides to use in convolutional operation
    batch_norm: decide to use BatchNormalization, default set to True
    activation: the activation function
    activate: decide to use the activation function, default set to True
    '''
    x=inputs
    if(batch_norm):
        x=tf.keras.layers.BatchNormalization()(x)
    if(activate):
        x=tf.keras.layers.Activation(activation=activation)(x)
    x=tf.keras.layers.Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding='same',use_bias=False,kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x=tf.keras.layers.Dropout(0.2)(x)
    return x

def normal_denseblock(inputs,block_layers,grow_rate=12):
    '''
    Build a normal dense block (without bottleneck)
    Parameters: 
    inputs: the input for this block
    block_layers: number of layers in the block
    grow_rate: the growth_rate factor in this block
    '''
    x=inputs
    y=x
    for i in range(block_layers):
        x=bn_relu_conv(x,filters=grow_rate)
        x=tf.keras.layers.Concatenate(axis=-1)([y,x])
        y=x

    return x

def bottleneck_layers(inputs,grow_rate=12):
    '''
    Construct a bottleneck in DenseNet, named DenseNetBC.
    Parameters:
    inputs: the input tensor, represents the inputs for these layers
    grow_rate: grow_rate factor in DenseNet
    '''

    # Note our bottlenecklayers include: BN-ReLU-Conv11-BN-ReLU-Conv33
    x=inputs
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.Activation('relu')(x)
    x=tf.keras.layers.Conv2D(filters=4*grow_rate,kernel_size=(1,1),strides=1,padding='same',use_bias=False,kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x=tf.keras.layers.Dropout(0.2)(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.Activation('relu')(x)
    x=tf.keras.layers.Conv2D(filters=grow_rate,kernel_size=(3,3),strides=1,padding='same',use_bias=False,kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x=tf.keras.layers.Dropout(0.2)(x)
    return x

def denseblockBC(inputs,block_layers,grow_rate=12):
    '''
    Build a denseblock BC
    Parameters:
    inputs: input tensor of the DenseNet
    block_layers: number of bottleneck layers in a denseblock
    grow_rate: the grow_rate factor in the denseblock
    '''
    x=inputs
    y=x
    for i in range(block_layers):
        x=bottleneck_layers(x,grow_rate)
        x=tf.keras.layers.Concatenate(axis=-1)([x,y])
        y=x
    return x


def build_densenet(n_layers,grow_rate=12,densenetBC=True):
    '''
    Build a densenet. With these parameters:
    n_layers: number of layers in the DenseNet
    grow_rate: the grow_rate factor in the DenseNet
    densenetBC: decides to use densenetBC (use bottleneck design)
    '''
    assert (n_layers-4)%3==0 ,"Number of layers is not suitable"
    block_layers=(n_layers-4)//3
    inputs=tf.keras.Input(shape=(32,32,3))
    if(not densenetBC):
        x=tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),strides=1,padding='same',use_bias=False,kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)
        n_filters=16
        for i in range(2):
            x=normal_denseblock(x,block_layers,grow_rate)
            n_filters+=grow_rate*block_layers
            x=tf.keras.layers.BatchNormalization()(x)
            x=tf.keras.layers.Activation('relu')(x)
            x=tf.keras.layers.Conv2D(filters=n_filters//2,kernel_size=(1,1),strides=1,padding='same',use_bias=False,kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
            x=tf.keras.layers.Dropout(0.2)(x)
            x=tf.keras.layers.AveragePooling2D(pool_size=(2,2))(x)
            n_filters//=2
        
        x=normal_denseblock(x,block_layers,grow_rate)
        x=tf.keras.layers.BatchNormalization()(x)
        x=tf.keras.layers.Activation('relu')(x)
        x=tf.keras.layers.AveragePooling2D(pool_size=(8,8))(x)
        y=tf.keras.layers.Flatten()(x)
        outputs=tf.keras.layers.Dense(10,activation='softmax',use_bias=True,kernel_initializer='he_normal')(y)
    else:
        assert block_layers%2==0, "Number of convolution in a denseblock must be devisible by 2"
        n_bottleneck=block_layers//2 #number of bottleneck layers in a Dense Block
        x=tf.keras.layers.Conv2D(filters=grow_rate*2,kernel_size=(3,3),strides=1,padding='same',use_bias=False,kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)
        n_filters=grow_rate*2
        for i in range(2):
            x=denseblockBC(x,n_bottleneck,grow_rate=grow_rate)
            n_filters+=grow_rate*n_bottleneck
            x=tf.keras.layers.BatchNormalization()(x)
            x=tf.keras.layers.Activation('relu')(x)
            x=tf.keras.layers.Conv2D(filters=n_filters//2,kernel_size=(1,1),strides=1,padding='same',use_bias=False,kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
            x=tf.keras.layers.Dropout(0.2)(x)
            n_filters//=2
            x=tf.keras.layers.AveragePooling2D(pool_size=(2,2))(x)
        x=denseblockBC(x,n_bottleneck,grow_rate)
        x=tf.keras.layers.BatchNormalization()(x)
        x=tf.keras.layers.Activation('relu')(x)
        x=tf.keras.layers.AveragePooling2D(pool_size=(8,8))(x)
        x=tf.keras.layers.Flatten()(x)
        outputs=tf.keras.layers.Dense(10,activation='softmax',kernel_initializer='he_normal')(x)
    model=tf.keras.Model(inputs=inputs,outputs=outputs)
    return model

def retrieve_file(file_name):
    '''
    retrieve file contents correspond to file name
    Parameters:
    file_name: represent the open file
    '''
    with open(file_name,'rb') as fo:
        my_dict=pickle.load(fo,encoding='bytes')
    return my_dict

def get_input(path_to_file):
    '''
    Process the input. Parameters:
    path_to_image: a string, represent the path to the image
    '''

    my_dict_1=retrieve_file(path_to_file+'data_batch_1')
    my_dict_2=retrieve_file(path_to_file+'data_batch_2')
    my_dict_3=retrieve_file(path_to_file+'data_batch_3')
    my_dict_4=retrieve_file(path_to_file+'data_batch_4')
    my_dict_5=retrieve_file(path_to_file+'data_batch_5')
    my_dict_6=retrieve_file(path_to_file+'test_batch') #test image

    data=np.concatenate((my_dict_1[b'data'],my_dict_2[b'data'],my_dict_3[b'data'],my_dict_4[b'data'],my_dict_5[b'data']),axis=0)
    train_labels=np.hstack((my_dict_1[b'labels'],my_dict_2[b'labels'],my_dict_3[b'labels'],my_dict_4[b'labels'],my_dict_5[b'labels']))

    my_dict=retrieve_file(path_to_file+'batches.meta')
    label_names=my_dict[b'label_names']

    m=data.shape[0]
    img1=data[:,:1024].reshape(m,32,32,1)
    img2=data[:,1024:2048].reshape(m,32,32,1)
    img3=data[:,2048:].reshape(m,32,32,1)

    img=np.concatenate((img1,img2,img3),axis=3)


    test_img=my_dict_6[b'data']
    test_labels=np.array(my_dict_6[b'labels'])
    m=test_img.shape[0]
    img1=test_img[:,:1024].reshape(m,32,32,1)
    img2=test_img[:,1024:2048].reshape(m,32,32,1)
    img3=test_img[:,2048:].reshape(m,32,32,1)
    test_img=np.concatenate((img1,img2,img3),axis=3)

    return img,train_labels,label_names,test_img,test_labels

def process_input(train_img,test_img,train_labels,test_labels,num_labels=10):
    '''
    Process the input by perpixel-mean subtraction
    Parameters:
    train_img: the image using on training
    test_img: the image using on testing
    train_labels: the 1D numpy array that encodes the train_labels
    test_labels: the 1D numpy array that encodes the test_labels
    num_labels: number of classes, default set to 10 (CIFAR-10)
    '''
    train_img=train_img.astype(np.float32)/255.0 #normalize the input image
    test_img=test_img.astype(np.float32)/255.0
    #subtract pixel mean
    train_img_mean=np.mean(train_img,axis=0)
    train_img_std=np.std(train_img,axis=0)
    train_img=(train_img-train_img_mean)/train_img_std
    test_img=(test_img-train_img_mean)/train_img_std

    #process the output
    y_train=tf.keras.utils.to_categorical(train_labels,num_labels)
    y_test=tf.keras.utils.to_categorical(test_labels,num_labels)
 
    return train_img,test_img,y_train,y_test

def lr_schedule(epochs):
    '''
    learning rate schedule
    '''
    if(epochs<20):lr=0.001
    elif(epochs<30):lr=0.0001
    else: lr=0.00001
    return lr

def compile_and_fit(train_img,train_labels,model,test_img,test_labels,batch_size=8,epochs=30):
    '''
    Perform compile and fit the training image
    Parameters:
    train_img: training images
    model: model we create above
    batch_size: batch size used for training
    epochs: number of epochs used when training the model
    '''

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule(0)),loss='categorical_crossentropy',metrics=['accuracy'])
    

    lr_scheduler=tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    lr_reducer=tf.keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1),patience=5,cooldown=0,min_lr=1e-6)
    callbacks=[lr_scheduler,lr_reducer]
    hist=model.fit(train_img,train_labels,batch_size=batch_size,epochs=epochs,validation_data=(test_img,test_labels),shuffle=True,callbacks=callbacks)

    return hist,model
