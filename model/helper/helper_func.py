## All helper function
import h5py
import numpy as np
import cv2

def load_h5_data(file_name):
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
    return (images + 1) / 2.0


def sketch(img):
    grey_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_invert = cv2.bitwise_not(grey_img)
    blur_img=cv2.GaussianBlur(img_invert, (71,71),0)
    invblur_img=cv2.bitwise_not(blur_img)
    sketch_img=cv2.divide(grey_img,invblur_img, scale=256.0)
    return sketch_img

def resize(image):
    return cv2.resize(image, (256,256), interpolation = cv2.INTER_AREA)


def predict_one_img(generator, img, out_path):
    img = resize(img)
    img = sketch(img)
    img = np.stack((img,)*3, axis=-1)
    img = (img / 127.5) - 1
    
    gen = generator.predict(np.array([img]))
    
    cv2.imwrite(out_path, rescale(gen[0]) * 255)
    
    
    