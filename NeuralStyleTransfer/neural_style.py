# Implement the neural style transfer problem

#We need tensorflow package for loading the pretrained VGG and train our model
import tensorflow as tf
# numpy for array maltipulation
import numpy as np
# Use opencv to read and process image
import cv2
import matplotlib.pyplot as plt
from collections import Counter
def vgg_layer(combined_layers):
    '''
    This function generates VGG layers for our model
    combined_layers:
    the dictionary represents the layer we want to make output
    '''
    #Load the pretrained VGG19 for imagenet, but don't include the fcn layers
    vgg=tf.keras.applications.VGG19(include_top=False,weights='imagenet')
    vgg.trainable=False #Set the trainable to freeze the network

    outputs = [vgg.get_layer(name).output for name in combined_layers.keys()] #get the output of the network
    model = tf.keras.Model([vgg.input],outputs)
    return model

def generate_gram_matrix(matrix):
    '''
    Generate a gram matrix from matrix
    Parameters:
    matrix: 4D tensor
    return a 2D tensor represents the Gram matrix
    '''
    matrix_shape=tf.shape(matrix).numpy()
    matrix=tf.reshape(matrix,shape=(matrix_shape[1]*matrix_shape[2],matrix_shape[3]))
    m_l=tf.cast(matrix_shape[1]*matrix_shape[2],dtype=tf.float32)
    gram_mat=tf.matmul(tf.transpose(matrix),matrix)
    return gram_mat/m_l


class NeuralStyleModel(tf.keras.Model):
    '''
    This class inherits from tf.keras.Model
    we must override this class to use the GradientTape
    
    '''
    def __init__(self,content_layers,style_layers,combined_layers):
        super(NeuralStyleModel,self).__init__()
        self.model=vgg_layer(combined_layers)
        self.model.trainable=False
        self.content_layers=content_layers
        self.style_layers=style_layers

    def call(self,input_img):
        '''
        Override the call method of tf.keras.Model class
        Expect the input in range [0,1]
        Return a dictionary of the content_output and style gram matrix output
        '''
        input_img=input_img*255.0 #retrieve the original size
        process_image=tf.keras.applications.vgg19.preprocess_input(input_img) #process the image
        
        outputs=self.model(process_image) #output of our model

        #We need to extract the content output and the style output for calculating loss
        content_outputs=outputs[:len(self.content_layers)]

        #Extract the style output and form the Gram matrix

        style_outputs=outputs[len(self.content_layers):]
        style_outs=[generate_gram_matrix(matrix) for matrix in style_outputs]

        return {'content':content_outputs,'style':style_outs}

def keep_range(image):
    '''
    Keep the image in range 0-1.
    image: a 4D tensor
    '''
    return tf.clip_by_value(image,clip_value_min=0.0,clip_value_max=1.0)

def total_loss(net_outputs,content_target,style_target,content_layers_weights,style_layers_weights,content_weights=0.01,style_weights=0.0001):
    '''
    Calculate the loss of Neural Style Transfer
    Parameters:
    net_outputs: the dictionary of the 'content' and 'style'
    content_target: the list of content_layers which we feed our content_image to the network
    style_target: the list of style_layers, which we feed our style_image to the network
    content_layers_weights: the list represents the portion each content_layer contributes to the content_loss
    style_layers_weights: the list represents the portion each style_layer contributes to the style_loss
    content_weights: the coefficient represent the tradeoff between the content_loss and style_loss
    style_weights: same as content_weights
    '''

    content_outputs=net_outputs['content']
    style_outputs=net_outputs['style']

    #Intialize the content_loss and calculate the content_loss
    content_loss=tf.constant(0,dtype=tf.float32)
    for i in range(len(content_layers_weights)):
        content_loss+=content_layers_weights[i]*tf.reduce_mean((content_outputs[i]-content_target[i])**2)
    
    #Initialize the style_loss and calculate the style_loss
    style_loss=tf.constant(0,dtype=tf.float32)
    for i in range(len(style_layers_weights)):
        style_loss+=style_layers_weights[i]*tf.reduce_mean((style_outputs[i]-style_target[i])**2)

    loss=content_weights*content_loss+style_weights*style_loss
    return loss
def train(generated_image,content_target,style_target,opt,model,content_layers_weights,style_layers_weights,content_weights,style_weights,variation_weights=1.0):
    '''
    A step of training
    Parameters:
    generated_image: the generated_image we want to compute
    content_target: list of content layers output when we feed the content image to the model
    style_target: list of style_layers output when we feed the style image to the model
    opt: optimization method
    model: our model
    content_layers_weights: list of portion each layer contributes to the content loss
    style_layers_weights: list of portion each layer contributes to the style loss
    content_weights: the tradeoff coefficient between the content_loss and style_loss
    '''
    derivatives=tf.image.sobel_edges(generated_image)
    with tf.GradientTape() as tape:
        net_ouputs=model(generated_image)
        loss=total_loss(net_ouputs,content_target,style_target,content_layers_weights,style_layers_weights,content_weights,style_weights)
        loss+=variation_weights*tf.reduce_sum(tf.abs(derivatives[...,0])+tf.abs(derivatives[...,1]))
    grad=tape.gradient(loss,generated_image)
    opt.apply_gradients([(grad,generated_image)])
    generated_image=keep_range(generated_image)
    return loss
def convert_to_image(image_tensor):
    '''
    Convert the image tensor to the image
    '''
    convert_img=image_tensor*255.0
    convert_img=convert_img.numpy().astype(np.uint8)
    convert_img=convert_img.reshape(convert_img.shape[1],convert_img.shape[2],convert_img.shape[3])
    return convert_img[:,:,::-1]

def plot_image(image):
    '''
    '''
    plt.imshow(image)
    plt.show()

def modeling(content_image_dir,style_image_dir,content_layers,style_layers,content_weights,style_weights,variation_weights,n_iterations=2500):
    '''
    Train the model
    content_image_dir: directory of the content image
    style_image_dir: directory of the style image
    n_iterations: number of iterations
    '''
    print("Read and create the tensor of content and style image ...",end='')
    content_img=cv2.imread(content_image_dir)
    style_img=cv2.imread(style_image_dir)

    #Transform the content and style image to 4D tensor of (224,224) (input of VGG network)
    content_tf=tf.constant(content_img,dtype=tf.float32)
    content_tf=tf.image.resize(content_tf,size=(224,224))
    content_shape=tf.shape(content_tf).numpy()
    content_tf=tf.reshape(content_tf,[1,content_shape[0],content_shape[1],content_shape[2]])

    style_tf=tf.constant(style_img,dtype=tf.float32)
    style_tf=tf.image.resize(style_tf,size=(224,224))
    style_shape=tf.shape(style_tf).numpy()
    style_tf=tf.reshape(style_tf,[1,style_shape[0],style_shape[1],style_shape[2]])

    print('Done')
    combined_layers=dict(Counter(content_layers)+Counter(style_layers))
    content_layers_weights=[weight for _,weight in content_layers.items()]
    style_layers_weights=[weight for _,weight in style_layers.items()]

    print(combined_layers)
    print(style_layers_weights)
    #Create a model
    model=NeuralStyleModel(content_layers,style_layers,combined_layers)

    #Retrieve content and style target
    content_target=model(content_tf/255.0)['content']
    style_target=model(style_tf/255.0)['style']

    #Optimization method
    opt=tf.keras.optimizers.Adam(learning_rate=0.01,epsilon=1e-2)

    #Initialize the generated image
    generated_image=tf.Variable(content_tf/255.0,trainable=True)

    for i in range(n_iterations):
        loss=train(generated_image,content_target,style_target,opt,model,content_layers_weights,style_layers_weights,content_weights,style_weights,variation_weights)
        if((i+1)%10==0):print('Loss at iteration {0} ={1}'.format(i+1,loss))
        if(i%40==0):
            image=convert_to_image(generated_image)
            plot_image(image)
    
    #convert tensor to numpy image
    image=convert_to_image(generated_image)
    return image
if __name__=='__main__':
    content_image_dir='Images/Tuebingen_Neckarfront.jpg'
    style_image_dir='Images/Shipwreck_turner.jpg'
    content_layers={'block5_conv2':1.0}
    style_layers={'block1_conv1':0.2,'block2_conv1':0.2,'block3_conv1':0.2,'block4_conv1':0.2,'block5_conv1':0.2}
    content_weights=1e-1
    style_weights=1e-4
    variation_weights=1e-2
    image=modeling(content_image_dir,style_image_dir,content_layers,style_layers,content_weights,style_weights,variation_weights,n_iterations=201)
    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
