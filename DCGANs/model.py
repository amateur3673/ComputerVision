import tensorflow as tf
from tensorflow.keras.layers import Dense,Conv2D,Reshape,Activation,LeakyReLU,BatchNormalization,Conv2DTranspose,Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input,Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import RandomNormal
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
import numpy as np

def construct_generator(z_dim=100):

  model=Sequential()

  model.add(Dense(4*4*256,input_dim=z_dim))
  model.add(LeakyReLU(0.2))
  model.add(Reshape((4,4,256)))

  model.add(Conv2DTranspose(filters=128,kernel_size=(5,5),strides=2,padding='same'))
  model.add(LeakyReLU(0.2))

  model.add(Conv2DTranspose(filters=128,kernel_size=(5,5),strides=2,padding='same'))
  model.add(LeakyReLU(0.2))

  model.add(Conv2DTranspose(filters=128,kernel_size=(5,5),strides=2,padding='same'))
  model.add(LeakyReLU(0.2))

  model.add(Conv2D(filters=3,kernel_size=(3,3),strides=1,padding='same',activation='tanh'))
  return model
def construct_discriminator(input_shape=(32,32,3)):
    
  model=Sequential()

  model.add(Conv2D(filters=64,kernel_size=(3,3),strides=1,padding='same',input_shape=input_shape))
  model.add(LeakyReLU(0.2))

  model.add(Conv2D(filters=128,kernel_size=(3,3),strides=2,padding='same'))
  model.add(LeakyReLU(0.2))

  model.add(Conv2D(filters=128,kernel_size=(3,3),strides=2,padding='same'))
  model.add(LeakyReLU(0.2))

  model.add(Conv2D(filters=256,kernel_size=(3,3),strides=2,padding='same'))
  model.add(LeakyReLU(0.2))

  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dropout(0.4))
  model.add(Dense(1,activation='sigmoid'))

  return model

def show_image(g,noise,z_dim=100):
    images=g.predict(noise)
    plt.figure(figsize=(10,10))
    for i in range(images.shape[0]):
        plt.subplot(10,10,i+1)
        img=(images[i]+1)*255.0/2
        plt.imshow(img.astype(np.uint8))
        plt.axis('off')
    plt.show()

def train_model(epochs=201,lr=2e-4,BATCH_SIZE=128,z_dim=100,input_shape=(32,32,3)):
    g=construct_generator(z_dim)
    g.summary()
    d=construct_discriminator(input_shape)
    d.compile(Adam(learning_rate=lr,beta_1=0.5),loss='binary_crossentropy')
    d.summary()
    d.trainable=False
    gan=Sequential()
    gan.add(g)
    gan.add(d)
    gan.compile(Adam(learning_rate=lr,beta_1=0.5),loss='binary_crossentropy')
    gan.summary()
    (X,_),(_,_)=cifar10.load_data()
    X=2*(X.astype(np.float32)/255.0)-1
    real_labels=np.ones((BATCH_SIZE,1))
    fake_labels=np.zeros((BATCH_SIZE,1))
    steps=X.shape[0]//BATCH_SIZE
    losses={'D':[],'G':[]}
    z_fix=np.random.uniform(-1,1,(100,z_dim))
    for e in range(epochs):
       print('Epoch {}'.format(e+1),end=' ')
       for step in range(steps):
          real_img=X[step*BATCH_SIZE:(step+1)*BATCH_SIZE]
          d.trainable=True
          real_loss=d.train_on_batch(real_img,real_labels)
          z=np.random.uniform(-1,1,(BATCH_SIZE,z_dim))
          fake_img=g.predict_on_batch(z)
          fake_loss=d.train_on_batch(fake_img,fake_labels)
          d_loss=real_loss+fake_loss
          
          d.trainable=False
          g_loss=gan.train_on_batch(z,real_labels)
        
       losses['D'].append(d_loss)
       losses['G'].append(g_loss)
       print('d_loss {}'.format(d_loss),end=' ')
       print('g_loss {}'.format(g_loss))

       if(e%10==0):
           show_image(g,z_fix,z_dim)
    return g,d,gan,losses
