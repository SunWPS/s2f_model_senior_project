from s2fgenerator.model import Generator, Discriminator, GAN
from keras.utils.vis_utils import plot_model

g = Generator().build()
print(g.summary())
d = Discriminator().build()
print(d.summary())

gan = GAN(g, d).build()
print(gan.summary())

plot_model(gan, to_file='model_architecture_img/gan.png', show_shapes=True, show_layer_names=True)
