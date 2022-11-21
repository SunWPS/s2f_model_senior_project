from s2fgenerator.model import Generator, Discriminator, GAN

g = Generator().load_model("model_saved/generator_weight.h5")
d = Discriminator().load_model("model_saved/discriminator_weight.h5")

gan = GAN(g, d).load_model("model_saved/gan_weight.h5")

print(gan.summary())
