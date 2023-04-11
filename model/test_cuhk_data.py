## All helper function
import h5py
import numpy as np

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

sketches, images = load_h5_data("data/cuhk_images.h5")

print(sketches.shape, images.shape)
print(sketches[0])