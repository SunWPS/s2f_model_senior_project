import sys

import h5py
import numpy as np

import config


def check_h5(f):
    file = h5py.File(config.output + f, "r+")

    images = np.array(file['/images']).astype('uint8')
    sketches = np.array(file['/sketches']).astype('uint8')

    print(f"file name: {f}")
    print(f"There are {len(images)} images.")
    print(f"There are {len(sketches)} sketches.") 
    print(f"sample image shape is {images[0].shape}")  
    

if __name__ == "__main__":
    file = sys.argv[1]
    check_h5(file)