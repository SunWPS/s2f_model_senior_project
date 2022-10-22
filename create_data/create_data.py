from multiprocessing import Process
import os.path

import cv2
import numpy as np
import h5py

import config


def sketch(img):
    grey_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_invert = cv2.bitwise_not(grey_img)
    blur_img=cv2.GaussianBlur(img_invert, (71,71),0)
    invblur_img=cv2.bitwise_not(blur_img)
    sketch_img=cv2.divide(grey_img,invblur_img, scale=256.0)
    return sketch_img


def  resize(img):
    return cv2.resize(img, config.img_size, interpolation = cv2.INTER_AREA)


def check_size(img):
    if img.shape[0] < 100:
        return False
    return True


def save_as_h5(all_images, all_sketches, start):

    all_images_np = np.array(all_images)
    all_sketches_np = np.array(all_sketches)

    print(all_images_np[0])

    file = h5py.File(config.output + f"{start//50000}_images.h5", "w")

    real_images = file.create_dataset(
        "images", np.shape(all_images_np), h5py.h5t.STD_U8BE, data=all_images_np
    )

    sketches_images = file.create_dataset(
        "sketches", np.shape(all_sketches_np), h5py.h5t.STD_U8BE, data=all_sketches_np
    )

    file.close()
    print(f"batch start at {start} finished: {len(all_images_np)} images")


def create_data(start):
    images = []
    sketches = []

    end =  start + 50000

    for i in range(start, end):

        if (i%10 == 0):
            print(f"batch start at {start}: {i}/{end-1}")

        f = config.input + f"{i}".rjust(6, "0") + ".jpg"

        if os.path.exists(f):
            img = cv2.imread(f)

            if check_size(img):
                re_size = resize(img)
                sketch_img = sketch(re_size)
                images.append(re_size)
                sketches.append(sketch_img)
    
    save_as_h5(images, sketches, start)


def main():

    print("start")

    process = [Process(target=create_data, args=(x, )) for x in range(0,265517,50000)]
    
    for p in process:
        p.start()
    for p in process:
        p.join()
    
    print("Happy ending")


if __name__ == "__main__":
    main()




