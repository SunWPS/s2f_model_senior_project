## All helper function
import h5py
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

def load_h5_data(file_name):
    """
        load data from .h5
        
        :param file_name: data file .h5
        :type file_name: string
        
        :return: sketches and target images
        :rtype: np.ndarray, np.ndarray
    """
    with h5py.File(file_name, "r+") as file:
        sketches = np.array(file['/sketches']).astype('uint8')
        images = np.array(file['/images']).astype('uint8')
    
        # convert to 3 channels
        sketches = np.stack((sketches,)*3, axis=-1)
        
        # (0,255) -> (-1,1)
        sketches = (sketches / 127.5) - 1
        images = (images / 127.5) - 1
        
    return sketches, images


def rescale(images):
    """
        rescale image from (-1,1) to (0-1)
        
        :param images: list of image
        :type images: np.ndarray
        
        :return: rescaled images
        :rtype: np.ndarray
    """
    return (images + 1) / 2.0


def sketch(img):
    """
        prepare data before predict
        
        :param img: image
        :type img: np.ndarray
        
        :return: sketch image
        :rtype: np.np.ndarray
        
    """
    img_invert = cv2.bitwise_not(img)
    blur_img=cv2.GaussianBlur(img_invert, (71,71),0)
    invblur_img=cv2.bitwise_not(blur_img)
    sketch_img=cv2.divide(img, invblur_img, scale=256.0)
    return sketch_img


def resize(image):
    """
        resize image to W 256 x H 256
        
        :param image: image
        :type image: np.ndarray
        
        :return: resize image
        :rtype: np.ndarray
    """
    return cv2.resize(image, (256,256), interpolation = cv2.INTER_AREA)


def predict_one_img(generator, img, out_path, base_black_img_path=None, b_level=False):
    """
        predict image
        
        :param generator: Generator model
        :type generator: keras.engine.functional.Functional
        
        :param img: sketch image
        :type img: np.ndarray
        
        :param out_path: output path for save generated image
        :type out_path: string
        
        :return: generated image
        :rtype: np.ndar ray
    """
    if b_level == True:
        base_black_img = cv2.imread(base_black_img_path)
    img = resize(img)
    if b_level == True:
        img = black_level(base_black_img, img)
    img = sketch(img)
    img = np.stack((img,)*3, axis=-1)
    img = (img / 127.5) - 1
    
    gen = generator.predict(np.array([img]))
    
    cv2.imwrite(out_path, rescale(gen[0]) * 255)
    
    return gen
    
    
def plot_one_gen_image(gen_result):
    """
        plot generate image
        
        :param gen_result: generated image
        :type gen_result: np.ndarray
    """
    plt.imshow(rescale(gen_result[0])[...,::-1])
    

def load_images(n_images, dir_path, gray=False):
    image_list = []
    
    for i in range(1, n_images + 1):
        if gray == False:
            img = cv2.imread(dir_path + f"/{i}.jpg")
        else:
            img = cv2.imread(dir_path + f"/{i}.jpg", 0)
        img = resize(img)
        image_list.append(img)
    return np.array(image_list)


def black_level(base_img, tg_img):
    gray1 = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY).astype(np.int16)
    gray2 = tg_img.astype(np.int16)
    
    min_pixel1 = gray1.min()
    min_pixel2 = gray2.min()
    
    diff = min_pixel1 - min_pixel2
    
    return np.clip(tg_img.astype(np.int16) + diff, 0, 255).astype(np.uint8)
