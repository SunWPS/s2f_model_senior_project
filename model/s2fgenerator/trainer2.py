import numpy as np
from numpy.random import default_rng
from matplotlib import pyplot as plt
from datetime import datetime


class GAN_trainer2:
    """
        GAN trainer for (W 256 x H 256)
        
        :param generator: Generator model
        :type generator: keras.engine.functional.Functional
        
        :param discriminator: Discriminator model
        :type discriminator: keras.engine.functional.Functional
        
        :param gan: GAN model
        :type discriminator: keras.engine.functional.Functional
        
        :param in_images: list of input images
        :type in_images: np.ndarray
        
        :param tg_images: list of target images
        :type tg_images: np.ndarray
        
        :param sample_out_dir_path: Directory path for sample output while training
        :type sample_out_dir_path: string
        
        :param history_graph_out_path: File name path for history graph image
        :type history_graph_out_path: string
        
        :param save_model_dir_path: Directory path for sample saving model after training successfully
        :type save_model_dir_path: string
        
        :param epochs: number of epochs for training (default=100)
        :type epochs: int
        
        :param batches: number of batches per epoch for training (default=1)
        :type batches: int
    """
    def __init__(self, generator, discriminator, gan, in_img1, tg_img1, in_img2, tg_img2, in_img3, tg_img3, 
                 sample_out_dir_path, history_graph_out_path, save_model_dir_path, epochs=100, batches=1):
        self.generator = generator
        self.discriminator = discriminator
        self.gan = gan
        self.in_images = in_img1
        self.tg_images = tg_img1
        self.in_images2 = in_img2
        self.tg_images2 = tg_img2
        self.in_images3 = in_img3
        self.tg_images3 = tg_img3
        self.sample_out_dir_path = sample_out_dir_path
        self.history_graph_out_path = history_graph_out_path
        self.save_model_dir_path = save_model_dir_path
        self.epochs = epochs
        self.batches = batches
    

    def get_real_sample_images_data(self, n_samples, n_patch=1, seed=None):
        """
            Random images for training 
            
            :param n_samples: number of random images
            :type n_samples: int
            
            :param n_patch: patch shape (default=1)
            :type n_patch: int
            
            :param seed: a random seed (default=None)
            :type seed: int
            
            :return: sample of images data and label
            :rtype: np.ndarray, np.ndarray, np.ndarray
        """
        # random isinstance
        rnd = default_rng(seed=seed)
        rand_i_1 = rnd.choice(self.in_images.shape[0], 5, replace=True)
        rand_i_2 = rnd.choice(self.in_images2.shape[0], 3, replace=True)
        rand_i_3 = rnd.choice(self.in_images3.shape[0], 2, replace=True)
        
        # X_real_A is input, X_real_B is target
        X_real_A, X_real_B = self.in_images[rand_i_1], self.tg_images[rand_i_1]
        tmp_real_A_1, tmp_real_B_1 = self.in_images2[rand_i_2], self.tg_images2[rand_i_2]
        tmp_real_A_2, tmp_real_B_2 = self.in_images3[rand_i_3], self.tg_images3[rand_i_3]
        
        X_real_A = np.append(X_real_A, tmp_real_A_1, 0)
        X_real_A = np.append(X_real_A, tmp_real_A_2, 0)
        X_real_B = np.append(X_real_B, tmp_real_B_1, 0)
        X_real_B = np.append(X_real_B, tmp_real_B_2, 0)

        # add label 1
        y = np.ones((n_samples, n_patch, n_patch, 1))
        
        return X_real_A, X_real_B, y
    
    
    def generate_sample_fake_data(self, samples, n_patch=1):
        """
            Generate fake image from sample data
            
            :param samples: sample data
            :type samples: np.ndarray
            
            :param n_patch: patch shape (default=1)
            :type n_patch: int
            
            :return: Generated images and label
            :rtype: np.ndarray, np.ndarray
        """
        # generate fake images
        X = self.generator.predict(samples)

        # add label 0
        y = np.zeros((len(X), n_patch, n_patch, 1))

        return X, y
    
    
    def summarize(self, iteration, n_samples, seed=None):
        """
            Summarize and generate sample images while training
            
            :param iteration: number of iteration 
            :type iteration: int
            
            :param n_sample: number of images that want to show in summarize image
            :type n_sample: int
            
            :param seed: a random seed (default=None)
            :type seed: int
        """
        
        def rescale(images):
            """
                rescale image from (-1,1) to (0,1)
                
                :param images: list of images
                :type images: np.ndarray
                
                :return iamges: list of rescaled images
                :rtype: np.ndarray
            """
            return (images + 1) / 2.0
        
        # real
        X_real_A, X_real_B, _ = self.get_real_sample_images_data(self.batches, seed=seed)
        
        # generate fake images
        X_fake, _ = self.generate_sample_fake_data(X_real_A)
        
        plt.figure(figsize=(20,12))

        X_real_A = rescale(X_real_A)
        X_real_B = rescale(X_real_B)
        X_fake = rescale(X_fake)
        
        # Create sample images
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

            sketches_ax.imshow(X_real_A[i])
            real_ax.imshow(X_real_B[i][...,::-1])
            gen_ax.imshow(X_fake[i][...,::-1])

        plt.savefig(f"{self.sample_out_dir_path}/sample_{str(iteration+1).rjust(7,'0')}.png")
    
    
    def plot_history(self, list_d_loss1, list_d_loss2, list_g_loss, list_d_acc1, list_d_acc2):
        """
            Plot the history graph of the model after training. It will show g_loss, d_loss1, d_loss2, d_acc1, and d_acc2.
            
            :param list_d_loss1: list of d_loss1
            :type list_d_loss1: list
            
            :param list_d_loss2: list of d_loss2
            :type list_d_loss2: list
            
            :param list_g_loss: list of g_loss
            :type list_g_loss: list
            
            :param list_d_acc1: list of d_acc1
            :type list_d_acc1: list
            
            :param list_d_acc2: list of d_acc2
            :type list_d_acc2: list
        """
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
            
        plt.savefig(self.history_graph_out_path)
    
    
    def training(self):
        """
        Training GAN model and save it after training
        """
        n_patch = self.discriminator.output_shape[1]
        
        # number of batches per epoch
        batches_per_epoch = int(len(self.in_images) / self.batches) 
        n_iterations = batches_per_epoch * self.epochs
        
        list_d_loss1 = []
        list_d_loss2 = []
        list_g_loss = []
        list_d_acc1 = []
        list_d_acc2 = []
        
        start = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        for i in range(n_iterations):
            
            X_real_A, X_real_B, y_real = self.get_real_sample_images_data(self.batches, n_patch)
            
            X_fake, y_fake = self.generate_sample_fake_data(X_real_A, n_patch)
            
            d_loss1, d_acc1 = self.discriminator.train_on_batch([X_real_A, X_real_B], y_real)
            d_loss2, d_acc2 = self.discriminator.train_on_batch([X_real_A, X_fake], y_fake)
            g_loss, _, _ = self.gan.train_on_batch(X_real_A, [y_real, X_real_B])
            
            print(">>> iteration %d | G[loss: %.3f] D[loss1: %.3f, loss2: %.3f, acc1: %.3f, acc2: %.3f]" % (i+1, g_loss, d_loss1, d_loss2, d_acc1, d_acc2))

            list_d_loss1.append(d_loss1)
            list_d_loss2.append(d_loss2)
            list_g_loss.append(g_loss)
            list_d_acc1.append(d_acc1)
            list_d_acc2.append(d_acc2)
            
            if (i+1) % (batches_per_epoch * 10) == 0 or i in [0, 1]:
                self.summarize(i, 5, seed=42)
        
        # save model
        self.generator.save_weights(f'{self.save_model_dir_path}/generator_weight.h5')
        self.discriminator.save_weights(f'{self.save_model_dir_path}/discriminator_weight.h5')
        self.gan.save_weights(f'{self.save_model_dir_path}/gan_weight.h5')
        
        self.summarize(n_iterations, 5, seed=42)
        self.plot_history(list_d_loss1, list_d_loss2, list_g_loss, list_d_acc1, list_d_acc2)
        
        end = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print(f"Successfully | start: {start}, end: {end}")
