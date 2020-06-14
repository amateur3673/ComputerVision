import tensorflow as tf
import numpy as np
import pickle

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
    x=tf.keras.layers.Conv2D(filters=depth,kernel_size=kernel_size,strides=strides,padding='same',kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)

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
    train_img=train_img-train_img_mean
    test_img=test_img-train_img_mean

    #process the output
    y_train=tf.keras.utils.to_categorical(train_labels,num_labels)
    y_test=tf.keras.utils.to_categorical(test_labels,num_labels)
 
    return train_img,test_img,y_train,y_test

def lr_schedule(epochs):
    '''
    learning rate schedule
    '''
    if(epochs<15):lr=1e-3
    elif(epochs<20):lr=1e-4
    elif(epochs<25):lr=1e-5
    else: lr=1e-6
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
    checkpoint=tf.keras.callbacks.ModelCheckpoint(filepath='logs/a.h5',monitor='val_acc',save_best_only=True,verbose=1)

    lr_scheduler=tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    lr_reducer=tf.keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1),patience=5,cooldown=0,min_lr=1e-6)
    callbacks=[checkpoint,lr_scheduler,lr_reducer]
    hist=model.fit(train_img,train_labels,batch_size=batch_size,epochs=epochs,validation_data=(test_img,test_labels),shuffle=True,callbacks=callbacks)

    return hist,model
