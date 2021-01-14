import tensorflow as tf 
import numpy as np 
import pandas as pd 
from tensorflow.python.keras.utils import conv_utils
import tensorflow.keras.backend as K
from pprint import pprint
import os 
import sys 
import tensorflow_datasets as tfds
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

LR_SIZE = 16 
scale = 1
HR_SIZE = int(LR_SIZE*scale*2)
channels = 3
##################################### Generator Model ################################

#####
""" PixelShuffler from https://gist.github.com/t-ae/6e1016cc188104d123676ccef3264981  """
#####


class PixelShuffler(tf.keras.layers.Layer):
    def __init__(self, size=(2, 2), data_format=None, **kwargs):
        super(PixelShuffler, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.size = conv_utils.normalize_tuple(size, 2, 'size')

    @tf.function
    def call(self, inputs):

        input_shape = K.int_shape(inputs)
        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank ' +
                             str(4) +
                             '; Received input shape:', str(input_shape))

        if self.data_format == 'channels_first':
            batch_size, c, h, w = input_shape
            if batch_size is None:
                batch_size = -1
            rh, rw = self.size
            oh, ow = h * rh, w * rw
            oc = c // (rh * rw)

            out = K.reshape(inputs, (batch_size, rh, rw, oc, h, w))
            out = K.permute_dimensions(out, (0, 3, 4, 1, 5, 2))
            out = K.reshape(out, (batch_size, oc, oh, ow))
            return out

        elif self.data_format == 'channels_last':
            batch_size, h, w, c = input_shape
            if batch_size is None:
                batch_size = -1
            rh, rw = self.size
            oh, ow = h * rh, w * rw
            oc = c // (rh * rw)

            out = K.reshape(inputs, (batch_size, h, w, rh, rw, oc))
            out = K.permute_dimensions(out, (0, 1, 3, 2, 4, 5))
            out = K.reshape(out, (batch_size, oh, ow, oc))
            return out

    def compute_output_shape(self, input_shape):

        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank ' +
                             str(4) +
                             '; Received input shape:', str(input_shape))

        if self.data_format == 'channels_first':
            height = input_shape[2] * self.size[0] if input_shape[2] is not None else None
            width = input_shape[3] * self.size[1] if input_shape[3] is not None else None
            channels = input_shape[1] // self.size[0] // self.size[1]

            if channels * self.size[0] * self.size[1] != input_shape[1]:
                raise ValueError('channels of input and size are incompatible')

            return (input_shape[0],
                    channels,
                    height,
                    width)

        elif self.data_format == 'channels_last':
            height = input_shape[1] * self.size[0] if input_shape[1] is not None else None
            width = input_shape[2] * self.size[1] if input_shape[2] is not None else None
            channels = input_shape[3] // self.size[0] // self.size[1]

            if channels * self.size[0] * self.size[1] != input_shape[3]:
                raise ValueError('channels of input and size are incompatible')

            return (input_shape[0],
                    height,
                    width,
                    channels)

    def get_config(self):
        config = {'size': self.size,
                  'data_format': self.data_format}
        base_config = super(PixelShuffler, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

class Residual_block_s(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(Residual_block_s,self).__init__(**kwargs)
        self._conv = tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=(1,1),padding="same")
        self._BN = tf.keras.layers.BatchNormalization(momentum=0.99)
        self._activation = tf.keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)
        self._conv_2 = tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=(1,1),padding="same")
        self._BN_2 = tf.keras.layers.BatchNormalization(momentum=0.99)

    @tf.function
    def call(self,input):
        x = self._conv(input)
        x = self._BN(x)
        x = self._activation(x)
        x = self._conv_2(x)
        x = self._BN_2(x)
        x = tf.keras.layers.Add()([x, input])
        return x 
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
        })
        return config

class Shuffle(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(Shuffle,self).__init__(**kwargs)
        self._conv = tf.keras.layers.Conv2D(filters=256,kernel_size=3,strides=(1,1))
        self._shuffle = PixelShuffler()
        self._activation = tf.keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)

    @tf.function
    def call(self,x):
        x = self._conv(x)
        x = self._shuffle(x)
        x = self._activation(x)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
        })
        return config

class Generator(tf.keras.Model):

    def __init__(self,nb_residual,x_input,y_input,channels,scale,**kwargs):
        super(Generator,self).__init__(**kwargs)
        self._nb_residual = nb_residual
        self._channels = channels
        self._scale = scale 
        self._conv2DI = tf.keras.layers.Conv2D(filters=64,kernel_size=9,strides=(1,1),input_shape=(x_input,y_input,self._channels),padding="same",dtype=tf.float64)
        self._activation = tf.keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)
        self._residual_block = [Residual_block_s() for _ in range(self._nb_residual)]
        self._shuffler = [PixelShuffler() for _ in range(self._scale)]
        self._middle_conv  = tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=(1,1),padding="same")
        self._BN = tf.keras.layers.BatchNormalization(momentum=0.99)
        self._last_conv  = tf.keras.layers.Conv2D(filters=channels,kernel_size=9,strides=(1,1),padding="same")

    

    @tf.function
    def call(self,x):
        x = self._conv2DI(x)
        x = self._activation(x)
        skip = x
        for layer in self._residual_block :
            x = layer(x)
        x = self._middle_conv(x)
        x = self._BN(x)
        x = tf.keras.layers.Add()([x,skip]) 
        for layer in self._shuffler :
            x = layer(x)
        x = self._last_conv(x)
        pprint(x)
        return x 

    def get_config(self):
        config = super().get_config().copy()
        config.update({
        })
        return config

##################################### Discriminator model #########################################""

class Discriminator_blocks(tf.keras.Model):
    def __init__(self,param,**kwargs):
        super(Discriminator_blocks,self).__init__(**kwargs)
        self._conv = tf.keras.layers.Conv2D(filters=param[1],kernel_size=param[0],strides=(param[2],param[2]),padding="same")
        self._norm = tf.keras.layers.BatchNormalization(momentum=0.99)
        self._activation = tf.keras.layers.LeakyReLU(alpha=0.3)

    @tf.function
    def call(self,x):
        x = self._conv(x)
        x = self._norm(x)
        x = self._activation(x)
        return x 


class Discriminator(tf.keras.Model):
    def __init__(self,channels,**kwargs):
        super(Discriminator,self).__init__(**kwargs)
        self._channels = channels
        self._first_conv = tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=(1,1),input_shape=(1,HR_SIZE,HR_SIZE,self._channels),padding="same")
        self._activation = tf.keras.layers.LeakyReLU(alpha=0.1)
        self._parameters=[[3,64,2],[3,128,1],[3,128,2],[3,256,1],[3,256,2],[3,512,1],[3,512,2]]
        self._discblock = [Discriminator_blocks(self._parameters[i]) for i in range(len(self._parameters))]
        self._dense1 = tf.keras.layers.Dense(units=1024)
        self._activation = tf.keras.layers.LeakyReLU(alpha=0.1)
        self._dense2 = tf.keras.layers.Dense(units=1)
        self._lastactivation = tf.keras.layers.Activation("sigmoid")


    @tf.function
    def call(self,x):
        x = self._first_conv(x)
        x = self._activation(x)
        for layer in self._discblock :
            x = layer(x)
        x = self._dense1(x)
        x = self._activation(x)
        x = self._dense2(x)
        x = self._lastactivation(x)
        return x 

##################################################### Training functions ######################################
def discriminator_loss(yreal,yfalse):
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    diff_r = cross_entropy(tf.ones_like(yreal), yreal)
    diff_f = cross_entropy(tf.zeros_like(yfalse), yfalse)
    return diff_r,diff_f 

def generator_loss(y):
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    return cross_entropy(tf.ones_like(y), y)


@tf.function
def train_step(hr_img,lw_img,batch_size=10):
    lw_img = tf.cast(tf.reshape(lw_img,[batch_size,LR_SIZE,LR_SIZE,channels]),dtype=tf.float32)
    hr_img = tf.cast(tf.reshape(hr_img,[batch_size,HR_SIZE,HR_SIZE,channels]),dtype=tf.float32)
    with tf.GradientTape(persistent=True) as tape :
        generated_images = generator(lw_img)

        fake_output = discriminator(generated_images)
        real_output = discriminator(hr_img)
        disc_loss_r,disc_loss_f = discriminator_loss(real_output, fake_output)
        gen_loss = generator_loss(fake_output)
      

    gradients_of_discriminator_r = tape.gradient(disc_loss_r, discriminator.trainable_variables)
    gradients_of_discriminator_f = tape.gradient(disc_loss_f, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator_r, discriminator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator_f, discriminator.trainable_variables))

    gradients_of_generator = tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    del tape
    return gen_loss,disc_loss_r

@tf.function
def train_step_init(hr_img,lw_img,batch_size=10):
    lw_img = tf.cast(tf.reshape(lw_img,[batch_size,LR_SIZE,LR_SIZE,channels]),dtype=tf.float32)
    hr_img = tf.cast(tf.reshape(hr_img,[batch_size,HR_SIZE,HR_SIZE,channels]),dtype=tf.float32)
    with tf.GradientTape() as tape :
        generated_images = generator(lw_img)
        gen_loss = cost = tf.reduce_mean(tf.math.squared_difference(generated_images, hr_img))
    gradients_of_generator = tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    return gen_loss


      






######################################## Loading images ##########################

def create_train_data(test=False,alpha=0.2,first=True):
    from PIL import Image
    import PIL
    comp=0
    training_data_x = np.zeros((5000,64,64,3))
    training_data_y = np.zeros((5000,16,16,3))
    path = "../Tinder_bot/TINDER_BOT/like_resize"
    listing = os.listdir(path)
    j= 0
    for img in listing :
        path = "../Tinder_bot/TINDER_BOT/like_resize"
        try : 
            path = os.path.join(path, img)
            if path[-5:] !="e.png" and path[-5:] !="d.jpg"  :
                print(path)
                img = PIL.Image.open(path)
                img2 = img.convert('RGB')
                img = img.convert('RGB')
                img = img.resize((64, 64), Image.ANTIALIAS)
                img = np.array(img)
                img2 = img2.resize((16, 16), Image.ANTIALIAS)
                img2 = np.array(img2)
                #img2 = (img2 - img2.mean()) / (img2.std() + 1e-8)
                #img = (img - img.mean()) / (img.std() + 1e-8)
                training_data_x[comp,:,:,:]=np.array(img)
                training_data_y[comp,:,:,:]=np.array(img2)
                comp=comp+1
        except Exception as e :
            print(e)
    path = "../Tinder_bot/TINDER_BOT/dislike_resize"
    listing = os.listdir(path)
    for img in listing :
        path = "../Tinder_bot/TINDER_BOT/dislike_resize"
        try : 
            path = os.path.join(path, img)
            if path[-5:] !="e.png" and path[-5:] !="d.jpg"  :
                print(path)
                img = PIL.Image.open(path)
                img2 = img.convert('RGB')
                img = img.convert('RGB')
                img = img.resize((64, 64), Image.ANTIALIAS)
                img = np.array(img)
                img2 = img2.resize((16, 16), Image.ANTIALIAS)
                img2 = np.array(img2)
                training_data_x[comp,:,:,:]=np.array(img)
                training_data_y[comp,:,:,:]=np.array(img2)
                comp=comp+1
        except Exception as e :
            print(e)
    training_data_x = training_data_x[: comp,:,:,:]
    training_data_y = training_data_y[: comp,:,:,:]

    return training_data_x,training_data_y




import matplotlib.pyplot as plt 
from skimage.transform import rescale, resize, downscale_local_mean


 # MAIN 

if __name__ == "__main__" :
    x_input = LR_SIZE
    y_input = LR_SIZE
    #X,Y = create_train_data()
    generator = Generator(4,x_input,y_input,channels,scale)
    discriminator = Discriminator(channels)
    nb_epochs = 100
    nb_batch  = 10
    generator_optimizer = tf.keras.optimizers.Adam(1e-5)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-7)
    #generator_optimizer.init_moments(generator.variables)
    #discriminator_optimizer.init_moments(discriminator.variables)
    (img_train, label_train), (img_test, label_test) = tfds.as_numpy(tfds.load(
    'cifar100',
        split=['train', 'test'],
        batch_size=-1,
        as_supervised=True,
    ))
    
    for j in range(0,600):
        X_train = img_train
        Y_train = tf.image.resize(img_train,[LR_SIZE,LR_SIZE])
        for i in range(0,X_train.shape[0]-nb_batch,nb_batch):
            gen_loss  = train_step_init(X_train[i:i+10,:,:,:],Y_train[i:i+10,:,:,:])
            #print("="*j+">"+"."*(int((nb_epochs-j)/20))+"|"+" loss value is :{} at epoch : {}".format(loss,j))
            if  i % 100 == 0 :
                sys.stdout.write("\r"+"="*int(i/X_train.shape[0]*100)+">"+"."*int((X_train.shape[0]-i)/X_train.shape[0]*100)+"|"+"gen loss is : {} at epoch : {}".format(gen_loss,j))
                sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()
    yw_img = tf.reshape(Y_train[1,:,:,:],[1,LR_SIZE,LR_SIZE,channels])
    img = generator.predict(yw_img)
    plt.imshow(img.reshape(HR_SIZE,HR_SIZE,channels))
    plt.show()

    for j in range(0,nb_epochs):
        X_train = img_train
        Y_train = tf.image.resize(img_train,[LR_SIZE,LR_SIZE])
        history = {"gen":[],"disc":[]}
        for i in range(0,X_train.shape[0]-nb_batch,nb_batch):
            gen_loss,disc_loss  = train_step(X_train[i:i+10,:,:,:],Y_train[i:i+10,:,:,:])
            history["gen"].append(gen_loss)
            history["disc"].append(disc_loss)
            #print("="*j+">"+"."*(int((nb_epochs-j)/20))+"|"+" loss value is :{} at epoch : {}".format(loss,j))
            if  i % 100 == 0 :
                sys.stdout.write("\r"+"="*int(i/X_train.shape[0]*100)+">"+"."*int((X_train.shape[0]-i)/X_train.shape[0]*100)+"|"+" disc loss value is :{}  and gen loss is : {} at epoch : {}".format(disc_loss,gen_loss,j))
                sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()

    """yw_img = tf.reshape(Y_train[1,:,:,:],[1,LR_SIZE,LR_SIZE,channels])
    img = generator.predict(yw_img)

    
    generator.load_weights("generator")  
    image_downscaled = downscale_local_mean(img_train[10,:,:,:], (2, 2,1))
    img = generator.predict(image_downscaled.reshape(1,16,16,3))

   

    plt.imshow(img.reshape(HR_SIZE,HR_SIZE,channels))
    plt.show()

    plt.imshow(img_train[10,:,:,:].reshape(32,32,3))
    plt.show()"""

    generator.save_weights("generator", save_format='tf')
    discriminator.save_weights("discriminator", save_format='tf')
