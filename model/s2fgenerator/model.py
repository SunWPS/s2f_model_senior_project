from abc import ABC, abstractmethod

from keras.initializers import RandomNormal
from keras import Input, activations
from keras.models import Model
from keras.layers import Concatenate
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Activation


class Base_model(ABC):
    """
        Base model class for Generator and Discriminator
        *** For image size (W 256 x H 256) ***
    """
    def __init__(self):
        self.image_shape = (256, 256, 3)
    
    @abstractmethod
    def build(self):
        pass
    
    
    @abstractmethod
    def load_model(self, model_path):
        pass
    
       
class Generator(Base_model):
    """
        Generator model for (W 256 x H 256)
        
        :param n_kernel: number of kernels for Convolution (default=4)
        :type n_kernel: int
        
        :param n_strides: number of strides for Convolution (default=1)
        :type n_strides: int
    """
    
    def __init__(self, n_kernels=4, n_strides=2):
        super().__init__()
        self.n_kernels = n_kernels
        self.n_strides = n_strides
    
    # overrideing abstract method
    def build(self):
        """
            Building generator model using UNET
            
            :return: generator model
            :rtype: keras.engine.functional.Functional
        """ 
        
        def encoder(prev_layer, n_filters, do_batchNorm=True):
            """
                Encoder block for UNET
                
                :param prev_layer: previous layer
                :type prev_layer: keras.engine.keras_tensor.KerasTensor
                
                :param n_filters: number of filters for Convolution
                :type n_filters: int
                
                :param do_batchNorm: Does the block use batchNorm layer or not?
                :type do_batchNorm: boolean
                
                :return: encoder block
                :rtype: keras.engine.keras_tensor.KerasTensor
            """
            # weight initialization
            init = RandomNormal(stddev=0.02)
        
            use_bias = False if do_batchNorm == True else True
            e = Conv2D(n_filters, kernel_size=self.n_kernels, strides=self.n_strides, 
                       padding="same", kernel_initializer=init, use_bias=use_bias)(prev_layer)
            if do_batchNorm == True:
                e = BatchNormalization()(e, training=True)
            e = LeakyReLU(alpha=0.2)(e)

            return e


        def decoder(prev_layer, skip_layer, n_filters, do_dropout=True):
            """
                Decoder block for UNET
                
                :param prev_layer: previous layer
                :type prev_layer: keras.engine.keras_tensor.KerasTensor
                
                :param skip_layer: layer that link to this block
                :type skip_layer: keras.engine.keras_tensor.KerasTensor
                
                :param n_filters: number of filters for Convolution
                :type n_filters: int
                
                :param do_dropout: Does the block use dropout layer or not?
                :type do_dropout: boolean
                
                :return: decoder block
                :rtype: keras.engine.keras_tensor.KerasTensor
            """
            # weight initialization
            init = RandomNormal(stddev=0.02)

            d = Conv2DTranspose(n_filters, kernel_size=self.n_kernels, strides=self.n_strides, 
                                padding="same",  kernel_initializer=init, use_bias=False)(prev_layer)
            d = BatchNormalization()(d, training=True)
            if do_dropout == True:
                d = Dropout(0.5)(d, training=True)
            d = Concatenate()([d, skip_layer])
            d = Activation(activations.relu)(d)

            return d
        
        # weight initialization
        init = RandomNormal(stddev=0.02)
        input_layer = Input(self.image_shape)
        
        # encoders
        e1 = encoder(input_layer, 64, do_batchNorm=False)
        e2 = encoder(e1, 128)
        e3 = encoder(e2, 256)
        e4 = encoder(e3, 512,)
        e5 = encoder(e4, 512,)
        e6 = encoder(e5, 512,)
        e7 = encoder(e6, 512,)

        # bottleneck
        bottleneck = Conv2D(512,  kernel_size=self.n_kernels, strides=self.n_strides, padding="same",  kernel_initializer=init)(e7)
        bottleneck = Activation(activations.relu)(bottleneck)

        # decoders
        d1 = decoder(bottleneck, e7, 512)
        d2 = decoder(d1, e6, 512)
        d3 = decoder(d2, e5, 512)
        d4 = decoder(d3, e4, 512, do_dropout=False)
        d5 = decoder(d4, e3, 256, do_dropout=False)
        d6 = decoder(d5, e2, 128, do_dropout=False)
        d7 = decoder(d6, e1, 64, do_dropout=False)

        output_layer = Conv2DTranspose(3, kernel_size=self.n_kernels, strides=self.n_strides, padding='same',  kernel_initializer=init)(d7)
        output_layer = Activation(activations.tanh)(output_layer)

        model = Model(inputs=input_layer, outputs=output_layer, name='Generator')
        return model
    
    # overrideing abstract method
    def load_model(self, model_weight_path):
        """
            Build and load model's weight
            
            :param model_weight_path: path of saved model's weight
            :type model_wight: string
            
            :return: generator model with saved weight
            :rtype: keras.engine.functional.Functional
        """ 
        g = self.build()
        g.load_weights(model_weight_path)
        return g


class Discriminator(Base_model):
    """
        Discriminator model for (W 256 x H 256)
        
        :param optimizer: optimizer for discriminator model
        
        :param loss_func: loss function for discriminator model (default='binary_crossentropy')
        
        :loss_weight: loss weight for discriminator model (defalut=[0.5])
        
        :param n_kernel: number of kernels for Convolution (default=4)
        :type n_kernel: int
        
        :param n_strides: number of strides for Convolution (default=1)
        :type n_strides: int
    """
    def __init__(self, optimizer, loss_func='binary_crossentropy', loss_weight=[0.5], n_kernels=4, n_strides=2):
        super().__init__()
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.loss_weight = loss_weight
        self.n_kernels = n_kernels
        self.n_strides = n_strides
        
        
    # overrideing abstract method
    def build(self):
        """
            Building discriminator model using UNET
            
            :return: discriminator model
            :rtype: keras.engine.functional.Functional
        """ 
        def conv2d_block(prev_layer, n_filters, weight_init, do_batchNorm=True):
            """
                Conv2D block for discriminator
                
                :param prev_layer: previous layer
                :type prev_layer: keras.engine.keras_tensor.KerasTensor
                
                :param n_filters: number of filters for Convolution
                :type n_filters: int
                
                :param do_batchNorm: Does the block use batchNorm layer or not?
                :type do_batchNorm: boolean
                
                :param weight_init: weight initialization
                :type weight_init: tf.keras.initializers.Initializer
                
                :return: Conv2D block
                :rtype: keras.engine.keras_tensor.KerasTensor
            """
            use_bias = False if do_batchNorm == True else True

            c = Conv2D(n_filters, kernel_size=self.n_kernels, strides=self.n_strides, 
                             padding="same", kernel_initializer=weight_init, use_bias=use_bias)(prev_layer)
            
            if do_batchNorm == True:
                c = BatchNormalization()(c)
            c = LeakyReLU(alpha=0.2)(c)

            return c
        
        # weight initialization
        init = RandomNormal(stddev=0.02)

        src_input_layer = Input(self.image_shape)
        target_input_layer = Input(self.image_shape)
        
        concat_input = Concatenate()([src_input_layer, target_input_layer])
        
        c1 = conv2d_block(concat_input, 64, init, False)
        c2 = conv2d_block(c1, 128, init, True)
        c3 = conv2d_block(c2, 256, init, True)
        c4 = conv2d_block(c3, 512, init, True)
        
        c5 = Conv2D(512, kernel_size=self.n_kernels, padding="same", kernel_initializer=init, use_bias=False)(c4)
        c5 = BatchNormalization()(c5)
        c5 = LeakyReLU(alpha=0.2)(c5)
        
        out_layer = Conv2D(1, kernel_size=self.n_kernels, 
                           padding="same", kernel_initializer=init)(c5)
        out_layer = Activation(activations.sigmoid)(out_layer)
        
        model = Model(inputs=[src_input_layer, target_input_layer], outputs=out_layer, name='Discriminator')
        
        model.compile(loss=self.loss_func, optimizer=self.optimizer, loss_weights=self.loss_weights, metrics=['accuracy'])
        return model
    
    # overrideing abstract method
    def load_model(self, model_weight_path):
        """
            Build and load model's weight
            
            :param model_weight_path: path of saved model's weight
            :type model_wight: string
            
            :return: discriminator model with saved weight
            :rtype: keras.engine.functional.Functional
        """ 
        d = self.build()
        d.load_weights(model_weight_path)
        d.compile(loss=self.loss_func, optimizer=self.optimizer, loss_weights=self.loss_weights, metrics=['accuracy'])
        return d


class GAN(Base_model):
    """
        GAN model for (W 256 x H 256)
        
        :param generator: generator model
        :type generator: keras.engine.functional.Functional
        
        :param discriminator: discriminator model
        :type discriminator: keras.engine.functional.Functional
        
        :param optimizer: optimizer for GAN model
        
        :param loss_func: loss function for GAN model (default=['binary_crossentropy', 'mae'])
        
        :param loss_weights: loss weights for GAN model (default=[1,100])
    """
    def __init__(self, generator, discriminator, optimizer, loss_func=['binary_crossentropy', 'mae'], loss_weights=[1,100]):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.loss_weights = loss_weights
    
    # overrideing abstract method
    def build(self):
        """
            Connect Generator model with Discriminator model
            
            :return: GAN model
            :rtype: keras.engine.functional.Functional
        """ 
        for layer in self.discriminator.layers:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = False

        input_layer = Input(self.image_shape)
        
        generator_layer = self.generator(input_layer)
        discriminator_layer = self.discriminator([input_layer, generator_layer])
        
        model = Model(inputs=input_layer, outputs=[discriminator_layer, generator_layer], name='GAN')
        model.compile(loss=self.loss_func, optimizer=self.optimizer, loss_weights=self.loss_weights)
        return model
    
    # overrideing abstract method
    def load_model(self, model_weight_path):
        """
            Build and load model's weight
            
            :param model_weight_path: path of saved model's weight
            :type model_wight: string
            
            :return: GAN model with saved weight
            :rtype: keras.engine.functional.Functional
        """ 
        g = self.build()
        g.load_weights(model_weight_path)
        g.compile(loss=self.loss_func, optimizer=self.optimizer, loss_weights=self.loss_weights)
        return g
        