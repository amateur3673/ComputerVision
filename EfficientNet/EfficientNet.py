import tensorflow as tf
import numpy as np
import albumentations as albu
import math
import keras
import efficientnet.keras as eff

class CustomGenerator(tf.keras.utils.Sequence):
    def __init__(self,x_set,y_set,shuffle=True,batch_size=8,augment=True):
        '''
        Class CustomGenerator to generate the data while training
        Parameters:
        x_set: dataset
        y_set: label
        shuffle: decides to shuffle the training set for each epoch
        batch_size: batch size for training
        augment: decides to use augmentation
        '''
        self.batch_size=batch_size #batch_size, set default by 8
        self.img_resolution=(224,224) #image resolution of the network, set default by (224,224)
        self.shuffle=shuffle
        self.X=x_set
        self.Y=y_set
        self.indices=np.arange(len(self.X)) #index image
        self.augment=augment
        self.on_epoch_end() #Call this method for shuffle the training datasets
    def __len__(self):
        '''
        Calculate the length of an epoch
        '''
        return math.ceil(len(self.X)/self.batch_size)
    def on_epoch_end(self):
        self.indices=np.arange(len(self.X))
        if(self.shuffle):
            np.random.shuffle(self.indices)
    def __getitem__(self,index):
        batch_index=self.indices[index*self.batch_size:(index+1)*self.batch_size]
        batch_X=self.X[batch_index,:,:,:]
        batch_Y=self.Y[batch_index]
        batch_X=tf.image.resize(batch_X,self.img_resolution)
        batch_X=batch_X.numpy().astype(np.float32)/255.0
        if(self.augment):
            return self.__process_batch(batch_X),batch_Y
        return batch_X,batch_Y
    def __process_batch(self,batch_img):
        '''
        Process the batch of img for data augmentation
        '''
        new_batch_img=np.zeros((batch_img.shape[0],batch_img.shape[1],batch_img.shape[2],batch_img.shape[3]))
        for i in range(len(new_batch_img)):
            composition=albu.Compose([albu.HorizontalFlip(p=0.5),albu.VerticalFlip(p=0.5),albu.GridDistortion(p=0.2),albu.ElasticTransform(p=0.2)])
            new_batch_img[i]=composition(image=batch_img[i])['image']
        return new_batch_img

def lr_schedule(epoch):
    '''
    Schedule the learning rate
    '''
    lr=1e-3
    factor=0.8
    return lr*factor**epoch

def train(datasets='cifar-10',epochs_1=10,epochs_2=10,batch_size=8,net_model='EfficientNetB0'):
    '''
    Train the efficientnet. Parameters:
    datasets: either cifar-10 or cifar-100
    epochs_1: number of epochs for training the classifiers
    epochs_2: number of epochs for fine-tuning model
    batch_size: batch size
    net_model: choose the EfficientNet model for training
    '''
    if(datasets=='cifar-10'):
        (x_train,y_train),(x_test,y_test)=tf.keras.datasets.cifar10.load_data()
        n_out=10
    elif(datasets=='cifar-100'):
        (x_train,y_train),(x_test,y_test)=tf.keras.datasets.cifar100.load_data()
        n_out=100
    #load the pretrained model
    if(net_model=='EfficientNetB0'):
        net=eff.EfficientNetB0(include_top=False,weights='imagenet',classes=n_out,input_shape=(224,224,3))
    elif(net_model=='EfficientNetB1'):
        net=eff.EfficientNetB1(include_top=False,weights='imagenet',classes=n_out,input_shape=(224,224,3))
    elif(net_model=='EfficientNetB2'):
        net=eff.EfficientNetB2(include_top=False,weights='imagenet',classes=n_out,input_shape=(224,224,3))
    elif(net_model=='EfficientNetB3'):
        net=eff.EfficientNetB3(include_top=False,weights='imagenet',classes=n_out,input_shape=(224,224,3))
    elif(net_model=='EfficientNetB4'):
        net=eff.EfficientNetB4(include_top=False,weights='imagenet',classes=n_out,input_shape=(224,224,3))
    elif(net_model=='EfficientNetB5'):
        net=eff.EfficientNetB5(include_top=False,weights='imagenet',classes=n_out,input_shape=(224,224,3))
    elif(net_model=='EfficientNetB6'):
        net=eff.EfficientNetB6(include_top=False,weights='imagenet',classes=n_out,input_shape=(224,224,3))
    elif(net_model=='EfficientNetB7'):
        net=eff.EfficientNetB7(include_top=False,weights='imagenet',classes=n_out,input_shape=(224,224,3))
    net.summary()
    # Build the classifier

    model=keras.models.Sequential()
    model.add(net)
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(n_out,activation='softmax'))
    
    #Generate custom data generator
    train_gen=CustomGenerator(x_train,y_train,shuffle=True,batch_size=batch_size,augment=True)
    test_gen=CustomGenerator(x_test,y_test,shuffle=False,batch_size=batch_size,augment=False)
    # Freeze the backbone, train the classifier

    net.trainable=False
    
    sgd=keras.optimizers.SGD(lr=0.001,momentum=0.9,nesterov=True)
    es = keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10, restore_best_weights = True, verbose = 1)
    rlrop = keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', mode = 'min', patience = 5, factor = 0.5, min_lr = 1e-6, verbose = 1)

    model.compile(optimizer=sgd,loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    hist_1=model.fit_generator(generator=train_gen,epochs=epochs_1,validation_data=test_gen,callbacks=[es,rlrop])

    #Save keras model

    train_gen=CustomGenerator(x_train,y_train,shuffle=False,batch_size=batch_size,augment=False)
    print('Evaluate training set:')
    model.evaluate_generator(generator=train_gen,verbose=1,use_multiprocessing=True)
    print('Evaluate validation set:')
    model.evaluate_generator(generator=test_gen,verbose=1,use_multiprocessing=True)
    model.save('stage_1.h5')

    #Fine tune
    net.trainable=True

    train_gen=CustomGenerator(x_train,y_train,shuffle=True,batch_size=batch_size,augment=True)

    lr_scheduler=keras.callbacks.LearningRateScheduler(lr_schedule,verbose=1)

    model.compile(optimizer=keras.optimizers.Adam(lr=lr_schedule(0)),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    hist_2=model.fit_generator(generator=train_gen,epochs=epochs_2,validation_data=test_gen,callbacks=[lr_scheduler,es])

    #Evaluate model
    train_gen=CustomGenerator(x_train,y_train,shuffle=False,batch_size=batch_size,augment=False)

    model.save('final.h5')
    print('Evaluate the training set:')
    model.evaluate_generator(generator=train_gen,use_multiprocessing=True,verbose=1)
    print('Evaluate the test set:')
    model.evaluate_generator(generator=test_gen,use_multiprocessing=True,verbose=1)

    return hist_1,hist_2,model
