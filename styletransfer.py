# Here we import our model
import tensorflow as tf
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.preprocessing.image import load_img,img_to_array
from tensorflow.python.keras.applications.vgg19 import preprocess_input
from tensorflow.python.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


model = VGG19(
    include_top = False,
    weights = 'imagenet'
)
model.trainable = False
# model.summary()

# here we are going to do image preprocessing and displaying
def load_and_process_image(image_path):
    img = load_img(image_path)
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img,axis=0)
    return img

def deprocess(image):
    image[:,:,0] +=103.939
    image[:,:,1] +=116.779
    image[:,:,2] +=123.68
    image = image[:,:,::-1]
    image = np.clip(image,0,255).astype('uint8')
    return image

def display_image(image):
    img = np.squeeze(image)
    img = deprocess(img)
    img = Image.fromarray(img, 'RGB')
    img.show()
    

# here we are gettimg the activation of intermediate layers
content_layer = 'block5_conv2'
style_layers = ['block1_conv1','block3_conv1','block5_conv1']

content_model = Model(
    inputs  = model.input,
    outputs = model.get_layer(content_layer).output
)

style_models = [Model(
    inputs  = model.input,
    outputs = model.get_layer(layer).output) for layer in style_layers]


# here we are creating content cost function
def content_cost(content,generated):
    a_C = content_model(content)
    a_G = content_model(generated)
    J_content = tf.reduce_mean(tf.square(a_C-a_G))
    return J_content

# here we are creating style cost function
def gram_matrix(A):
    n_C = int(A.shape[-1])
    a = tf.reshape(A,[-1,n_C])
    n = tf.shape(a)[0]
    G = tf.matmul(a,a,transpose_a=True)
    return G/tf.cast(n,tf.float32)


def style_cost(style,generated):
    J_style =0
    lam = 1./len(style_models)
    for style_model in style_models:
        a_S = style_model(style)
        a_G = style_model(generated)
        GS = gram_matrix(a_S)
        GG = gram_matrix(a_G)
        current_cost = tf.reduce_mean(tf.square(GS-GG))
        J_style +=current_cost*lam
    return J_style


# here we train our model for creating generated image
import time
generated_images = []
def train(content_path,style_path,iterations=20,alpha=10.,beta=20.):
    content = load_and_process_image(content_path)
    style = load_and_process_image(style_path)
    
    generated = content
    generated = tf.Variable(content,dtype = tf.float32)
   
    opt = tf.optimizers.Adam(learning_rate=7.)

    best_cost = 1e12 +0.1
    best_image = None

    start_time = time.time()

    for i in range(iterations):
        with tf.GradientTape() as tape:
            J_content = tf.convert_to_tensor(content_cost(content,generated),np.float32)
            J_style =   tf.convert_to_tensor(style_cost(style,generated),np.float32)
            J_total = tf.convert_to_tensor(alpha*J_content + beta*J_style,np.float32)

        grads = tape.gradient(J_total,generated)
        opt.apply_gradients([(grads,generated)])

        if J_total<best_cost:
            best_cost = J_total
            best_image = generated.numpy()

        print('Cost at {} : {}. Time elapsed: {}'.format(i,J_total,time.time()-start_time))
        generated_images.append(generated.numpy())

    return best_image
        

display_image(train('content.jpg','style.jpg'))