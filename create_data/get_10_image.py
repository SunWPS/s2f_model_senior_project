import h5py
import numpy as np

import config

import sys

def save_as_h5(all_images, all_sketches, d):

    all_images_np = np.array(all_images)
    all_sketches_np = np.array(all_sketches)

    file = h5py.File(config.output + f"test{d}_images.h5", "w")

    real_images = file.create_dataset(
        "images", np.shape(all_images_np), h5py.h5t.STD_U8BE, data=all_images_np
    )

    sketches_images = file.create_dataset(
        "sketches", np.shape(all_sketches_np), h5py.h5t.STD_U8BE, data=all_sketches_np
    )

    file.close()
    print(f"compress finished: {len(all_images_np)} images")


def check_h5(d):
    file = h5py.File(f'hdf5/{d}_images.h5', "r+")

    images = np.array(file['/images']).astype('uint8')
    sketches = np.array(file['/sketches']).astype('uint8')

    images_10 = images[:10]
    sketches_10 = sketches[:10]

    save_as_h5(images_10, sketches_10, d)

if __name__ == "__main__":
    data  = sys.argv[1]
    check_h5(data)
