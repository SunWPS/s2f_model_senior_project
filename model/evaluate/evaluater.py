from skimage.metrics import structural_similarity
from skimage.transform import resize
from keras.applications.inception_v3 import InceptionV3
import numpy as np
import tensorflow as tf
from keras.losses import kullback_leibler_divergence as D_kl


class Evaluater:
    def __init__(self):
        pass
    
    
    def calculate_psnr(self, real_images, gen_images):
        
        pixel_max = 255.0
        psnr_score = []
        
        for i in range(len(real_images)):
            mse_score = ((real_images[i] - gen_images[i]) ** 2).mean()
            if mse_score == 0:
                psnr_score.append(100)
            else:
                psnr = 20 * np.log10(pixel_max / np.sqrt(mse_score))
                psnr_score.append(psnr)
        
        return np.mean(psnr_score)
                
    
    def calculate_ssim(self, real_images, gen_images):
        ssim_list = []
        
        for i in range(len(real_images)):
            score, _ = structural_similarity(real_images[i], gen_images[i], full=True)
            ssim_list.append(score)
        
        final_ssim = np.mean(ssim_list)
        return (final_ssim + 1) / 2
    
    
    def __preprocess_images_for_is(self, images):
        tf_images = tf.convert_to_tensor(images)
        
        reshape_images = tf.image.resize(tf_images, [299, 299])
        preprocess_images = reshape_images / 255.0
        return preprocess_images
    
    
    def calculate_inception_scroe(self, images):
        
        preprocess_images = self.__preprocess_images_for_is(images)
        
        inception_model = InceptionV3()
        p_yx = inception_model(preprocess_images, training=False)
        
        # add eps to prevent division by zeor
        eps = 1e-12
        p_yx += eps
        
        
        # compute KL divergence
        # kl_divergence = p_yx * log(p_yx / p_y)
        n_classes = p_yx.shape[1]
        p_y = tf.ones((n_classes,)) / n_classes
        
        kl_divergence = tf.reduce_mean(D_kl(p_yx, p_y))
        
        # compute IS
        is_score = tf.exp(kl_divergence)
                
        return tf.get_static_value(is_score)
    
    
    def calculate_psnr_and_ssim(self, real_images, gen_images):
        mse = self.calculate_psnr(real_images, gen_images)
        ssim = self.calculate_ssim(real_images, gen_images)
        
        return "PSNR: %.2f SSIM %.2f" %(mse, ssim)
    
  