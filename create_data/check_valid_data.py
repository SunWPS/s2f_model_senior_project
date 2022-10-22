import sys

import h5py
import numpy as np

import config


def check_h5(f):
    print("start")
    file = h5py.File(config.output + f, "r+")

    images = np.array(file['/images']).astype('uint8')
    sketches = np.array(file['/sketches']).astype('uint8')

    print(f"images: {len(images)}")
    print(f"sketches: {len(sketches)}")

    for i in range(len(images)):
        all_zeros_image = not np.any(images[i])
        all_zeros_sketch = not np.any(sketches[i])

        if all_zeros_image or all_zeros_sketch:
            print(f"patiition {f} error at {i}")
    
    file.close()
    print("finish")
    

if __name__ == "__main__":
    n = sys.argv[1]
    file = f"{n}_images.h5"
    check_h5(file)