#!/usr/bin/env python
# coding: utf-8

# # **Optimize loss model**

# ## **Import Library**

# In[3]:


import os
import h5py
import numpy as np
import pandas as pd
from numpy.random import default_rng
from matplotlib import pyplot as plt
from datetime import datetime

import tensorflow as tf
import keras.backend as kb
from keras.initializers import RandomNormal
from keras import Input, activations
from keras.models import Model
from keras.layers import Dense
from keras.layers import Concatenate
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Activation
from keras.losses import mean_absolute_error

from tensorflow.keras.optimizers import Adam
from keras.utils.vis_utils import plot_model


# ## **Set main parameters**

# In[12]:


# for images
img_width = 256
img_height = 256
img_channels = 3

img_shape = (img_width, img_height, img_channels)

batches = 10


# ## **Build Generator**

# ### Encoder

# In[13]:


def encoder(prev_layer, n_filters, n_kernels=4, n_strides=1, do_batchNorm=True):
  # weight initialization
  init = RandomNormal(stddev=0.02)
  
  use_bias = False if do_batchNorm == True else True

  encoder = Conv2D(n_filters, kernel_size=n_kernels, strides=n_strides, padding="same", kernel_initializer=init, use_bias=use_bias)(prev_layer)
  if do_batchNorm == True:
    encoder = BatchNormalization()(encoder, training=True)
  encoder = LeakyReLU(alpha=0.2)(encoder)

  return encoder


# ### Decoder

# In[14]:


def decoder(prev_layer, skip_layer, n_filters, n_kernels=4, n_strides=1, do_dropout=True):
  # weight initialization
  init = RandomNormal(stddev=0.02)

  decoder = Conv2DTranspose(n_filters, kernel_size=n_kernels, strides=n_strides, padding="same",  kernel_initializer=init, use_bias=False)(prev_layer)
  decoder = BatchNormalization()(decoder, training=True)
  if do_dropout == True:
    decoder = Dropout(0.5)(decoder, training=True)
  decoder = Concatenate()([decoder, skip_layer])
  decoder = Activation(activations.relu)(decoder)

  return decoder


# ### Generator

# In[15]:


def build_generator(img_shape=img_shape, n_kernels=4, n_strides=1):
  # weight initialization
  init = RandomNormal(stddev=0.02)

  input_layer = Input(img_shape)

  # encoders
  encoder_1 = encoder(input_layer, 64, n_kernels=n_kernels, n_strides=n_strides, do_batchNorm=False)
  encoder_2 = encoder(encoder_1, 128, n_kernels=n_kernels, n_strides=n_strides)
  encoder_3 = encoder(encoder_2, 256, n_kernels=n_kernels, n_strides=n_strides)
  encoder_4 = encoder(encoder_3, 512, n_kernels=n_kernels, n_strides=n_strides)
  encoder_5 = encoder(encoder_4, 512, n_kernels=n_kernels, n_strides=n_strides)
  encoder_6 = encoder(encoder_5, 512, n_kernels=n_kernels, n_strides=n_strides)
  encoder_7 = encoder(encoder_6, 512, n_kernels=n_kernels, n_strides=n_strides)

  bottleneck = Conv2D(512,  kernel_size=n_kernels, strides=n_strides, padding="same",  kernel_initializer=init)(encoder_7)
  bottleneck = Activation(activations.relu)(bottleneck)

  # decoders
  decoder_1 = decoder(bottleneck, encoder_7, 512, n_kernels=n_kernels, n_strides=n_strides)
  decoder_2 = decoder(decoder_1, encoder_6, 512, n_kernels=n_kernels, n_strides=n_strides)
  decoder_3 = decoder(decoder_2, encoder_5, 512, n_kernels=n_kernels, n_strides=n_strides)
  decoder_4 = decoder(decoder_3, encoder_4, 512, n_kernels=n_kernels, n_strides=n_strides, do_dropout=False)
  decoder_5 = decoder(decoder_4, encoder_3, 256, n_kernels=n_kernels, n_strides=n_strides, do_dropout=False)
  decoder_6 = decoder(decoder_5, encoder_2, 128, n_kernels=n_kernels, n_strides=n_strides, do_dropout=False)
  decoder_7 = decoder(decoder_6, encoder_1, 64, n_kernels=n_kernels, n_strides=n_strides, do_dropout=False)

  output_layer = Conv2DTranspose(3, kernel_size=n_kernels, strides=n_strides, padding='same',  kernel_initializer=init)(decoder_7)
  output_layer = Activation(activations.tanh)(output_layer)

  model = Model(inputs=input_layer, outputs=output_layer, name='generator')
  return model


# ## **Build Discriminator**

# In[16]:


def build_discriminator(img_shape=img_shape, n_kernels=4, n_strides=1):

  # weight initialization
  init = RandomNormal(stddev=0.02)

  # src_input = Sequential()
  src_input = Input(shape=img_shape)

  # target_input = Sequential()
  target_input = Input(shape=img_shape)

  concat_input = Concatenate()([src_input, target_input])

  layer = Conv2D(64, kernel_size=n_kernels, strides=n_strides, padding="same",  kernel_initializer=init)(concat_input)
  layer = LeakyReLU(alpha=0.2)(layer)

  layer = Conv2D(128, kernel_size=n_kernels, strides=n_strides, padding="same",  kernel_initializer=init, use_bias=False)(layer)
  layer = BatchNormalization()(layer)
  layer = LeakyReLU(alpha=0.2)(layer)

  layer = Conv2D(256, kernel_size=n_kernels, strides=n_strides, padding="same",  kernel_initializer=init, use_bias=False)(layer)
  layer = BatchNormalization()(layer)
  layer = LeakyReLU(alpha=0.2)(layer)

  layer = Conv2D(512, kernel_size=n_kernels, strides=n_strides, padding="same",  kernel_initializer=init, use_bias=False)(layer)
  layer = BatchNormalization()(layer)
  layer = LeakyReLU(alpha=0.2)(layer)
  
  layer = Conv2D(512, kernel_size=n_kernels, padding="same", kernel_initializer=init, use_bias=False)(layer)
  layer = BatchNormalization()(layer)
  layer = LeakyReLU(alpha=0.2)(layer)

  layer = Conv2D(1, kernel_size=n_kernels, padding="same", kernel_initializer=init)(layer)

  out_layer = Activation(activations.sigmoid)(layer)

  model = Model(inputs=[src_input, target_input], outputs=out_layer, name='discriminator')
  
  return model


# ## **Connecting generator and discriminator to build GAN**

# In[17]:


def build_gan(generator, discriminator, img_shape=img_shape):
  for layer in discriminator.layers:
    if not isinstance(layer, BatchNormalization):
      layer.trainable = False

  input_layer = Input(shape=img_shape)
  
  generator_layer = generator(input_layer)

  discriminator_layer = discriminator([input_layer, generator_layer])

  model = Model(inputs=input_layer, outputs=[discriminator_layer, generator_layer], name='GAN')
  
  return model


# ## **Loss Function**

# In[18]:


def pixel_loss(y_true, y_pred):
  return kb.mean(kb.abs(y_true - y_pred))


def contextual_loss(y_true, y_pred):
  a = tf.image.rgb_to_grayscale(tf.slice(y_pred, [0, 0, 0, 0], [batches, 256, 256, 3]))
  b = tf.image.rgb_to_grayscale(tf.slice(y_true, [0, 0, 0, 0], [batches, 256, 256, 3]))

  y_pred = tf.divide(tf.add(tf.reshape(a, [tf.shape(a)[0], -1]), 1), 2)
  y_true = tf.divide(tf.add(tf.reshape(b, [tf.shape(b)[0], -1]), 1), 2)

  p_shape = tf.shape(y_true)
  q_shape = tf.shape(y_pred)

  p_ = tf.divide(y_true, tf.tile(tf.expand_dims(tf.reduce_sum(y_true, axis=1), 1), [1,p_shape[1]]))
  q_ = tf.divide(y_pred, tf.tile(tf.expand_dims(tf.reduce_sum(y_pred, axis=1), 1), [1,p_shape[1]]))
    
  return tf.reduce_sum(tf.multiply(p_, tf.math.log(tf.divide(p_, q_))), axis=1)


def total_loss(y_true, y_pred):
  pix_loss = pixel_loss(y_true, y_pred)
  cont_loss = contextual_loss(y_true, y_pred)
  return (0.2 * pix_loss) + (0.8 * cont_loss)


# ## **Prepare before training**
# 
# label
# 
# 1: real
# 
# 0: fake

# ## **Prepare Data**

# In[28]:


# get data from h5 file
def load_data(file_name):
  with h5py.File(file_name, "r+") as file:
    images = np.array(file['/images']).astype('uint8')
    sketches = np.array(file['/sketches']).astype('uint8')

    # convert to 3 channels
    sketches = np.stack((sketches,)*3, axis=-1)
    
    # (0,255) -> (-1,1)
    images = (images / 127.5) - 1
    sketches = (sketches / 127.5) - 1

  return images, sketches

# random pairs of images for a batch
def get_real_sample_images_data(images, sketches, n_samples, n_patches=1, seed=None):
  # random instance
  rnd = default_rng(seed=seed)
  rand_i = rnd.choice(images.shape[0], n_samples, replace=True)
  X_images, X_sketches = images[rand_i], sketches[rand_i]

  # add label 1
  y = np.ones((n_samples, n_patches, n_patches, 1))
  # y = np.random.uniform(0.7, 1, (n_samples, n_patches, n_patches, 1))

  return X_images, X_sketches, y

def generate_sample_fake_data(generator, samples, n_patches=1):
  # generate fake images
  X = generator.predict(samples)

  # add label 0
  y = np.zeros((len(X), n_patches, n_patches, 1))

  return X, y


# ## **Summarize**

# In[20]:


def rescale(images):
  # (-1,1) -> (0,1) for matplotlib
  return (images + 1) / 2.0

def summarize(iteration, generator, images, sketches, n_samples, data_id):
  # real
  X_images, X_sketches, _ = get_real_sample_images_data(images, sketches, n_samples, seed=42)

  # generate fake images
  X_fake_images, _ = generate_sample_fake_data(generator, X_sketches)
  
  plt.figure(figsize=(20,12))

  X_sketches = rescale(X_sketches)
  X_images = rescale(X_images)
  X_fake_images = rescale(X_fake_images)
 
  # same sample images
  for i in range(n_samples):
    sketches_ax = plt.subplot2grid((3,n_samples), (0,i))
    real_ax = plt.subplot2grid((3,n_samples), (1,i))
    gen_ax = plt.subplot2grid((3,n_samples), (2,i))

    sketches_ax.set_xticks([])
    sketches_ax.set_yticks([])
    real_ax.set_xticks([])
    real_ax.set_yticks([])
    gen_ax.set_xticks([])
    gen_ax.set_yticks([])

    if i == 0:
      sketches_ax.set_ylabel("Sketches", fontsize=20)
      real_ax.set_ylabel("Real images", fontsize=20)
      gen_ax.set_ylabel("Generated images", fontsize=20)

    sketches_ax.imshow(X_sketches[i])
    real_ax.imshow(X_images[i][...,::-1])
    gen_ax.imshow(X_fake_images[i][...,::-1])

  plt.savefig(f"sample_pic/set_{data_id}/sample_{str(iteration+1).rjust(7,'0')}.png")

  # generator.save(f"/content/model/generator_{str(iteration+1).rjust(7,'0')}.h5")


# In[37]:


def plot_history(list_d_loss1, list_d_loss2, list_g_loss, list_d_acc1, list_d_acc2, data_id):
  plt.subplot(311)
  plt.plot(list_g_loss, label="g_loss")
  plt.xlabel("iteration")
  plt.ylabel("loss")
  plt.legend()

  plt.subplot(312)
  plt.plot(list_d_loss1, label="d_loss1")
  plt.plot(list_d_loss2, label="d_loss2")
  plt.xlabel("iteration")
  plt.ylabel("loss")
  plt.legend()
  
  plt.subplot(313)
  plt.plot(list_d_acc1, label="d_acc1")
  plt.plot(list_d_acc2, label="d_acc2")
  plt.xlabel("iteration")
  plt.ylabel("accuracy")
  plt.legend()
    
  plt.savefig(f"graph/graph_{data_id}.png")


# ## **Train**
# d_loss1: discriminator (real images)
# 
# d_loss2: discriminator (generated images)
# 
# g_loss: generator

# In[38]:


def training(generator, discriminator, gan, images, sketches, epochs=100, batches=1, data_id=0):
  path = f"sample_pic/set_{data_id}"
  os.mkdir(path)

  n_patches = discriminator.output_shape[1]

  # number of batches per epoch
  batches_per_epoch = int(len(sketches) / batches) 
  n_iterations = batches_per_epoch * epochs

  list_d_loss1 = []
  list_d_loss2 = []
  list_g_loss = []
  list_d_acc1 = []
  list_d_acc2 = []

  for i in range(n_iterations):

    X_images, X_sketches, y_real = get_real_sample_images_data(images, sketches, batches, n_patches)

    X_fake_images, y_fake = generate_sample_fake_data(generator, X_sketches, n_patches)

    d_loss1, d_acc1 = discriminator.train_on_batch([X_sketches, X_images], y_real)
    d_loss2, d_acc2 = discriminator.train_on_batch([X_sketches, X_fake_images], y_fake)
    g_loss, _, _ = gan.train_on_batch(X_sketches, [y_real, X_images])

    print(">>> iteration %d | G[loss: %.3f] D[loss1: %.3f, loss2: %.3f, acc1: %.3f, acc2: %.3f]" % (i+1, g_loss, d_loss1, d_loss2, d_acc1, d_acc2))

    list_d_loss1.append(d_loss1)
    list_d_loss2.append(d_loss2)
    list_g_loss.append(g_loss)
    list_d_acc1.append(d_acc1)
    list_d_acc2.append(d_acc2)

    if (i+1) % (batches_per_epoch * 10) == 0 or i in [0, 1]:
      summarize(i, generator, images, sketches, 5, data_id=data_id)
  

  # save model
  generator.save_weights(f'model/generator_{data_id}.h5')
  discriminator.save_weights(f'model/discriminator_{data_id}.h5')
  gan.save_weights(f'model/gan_{data_id}.h5')

  summarize(n_iterations, generator, images, sketches, 5, data_id=data_id)
  plot_history(list_d_loss1, list_d_loss2, list_g_loss, list_d_acc1, list_d_acc2, data_id=data_id)

  
  
  # remove data file
  os.remove(f"data/{data_id}_images.h5")


# In[39]:
d = build_discriminator(n_strides=2)
print(d.summary())


# In[ ]:





# In[ ]:




