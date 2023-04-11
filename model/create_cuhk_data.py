import h5py
import cv2
import numpy as np


def save_as_h5(all_images, all_sketches):

    all_images_np = np.array(all_images)
    all_sketches_np = np.array(all_sketches)
    
    file = h5py.File(f"data/cuhk_images.h5", "w")

    real_images = file.create_dataset(
        "images", np.shape(all_images_np), h5py.h5t.STD_U8BE, data=all_images_np
    )

    sketch_images = file.create_dataset(
        "sketches", np.shape(all_sketches_np), h5py.h5t.STD_U8BE, data=all_sketches_np
    )

    file.close()
    print(f"finish")
    
    
def create_data1():
    images = []
    sketches = []

    for i in range(1, 89):

       img = cv2.imread(f"data2/photos/1 ({i}).jpg") 
       img = img[50:, 0:]
       img = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
       cv2.imwrite(f"data2/photos/1 ({i}).jpg", img)
       
       sketch = cv2.imread(f"data2/sketch/1 ({i}).jpg", 0)
       sketch = sketch[50:, 0:]
       sketch = cv2.resize(sketch, (256,256), interpolation = cv2.INTER_AREA)
       cv2.imwrite(f"data2/sketch2/1 ({i}).jpg", sketch)

       images.append(img)
       sketches.append(sketch)
    
    return images, sketches


images, sketches = create_data1()

all_imnp = np.array(images, dtype=np.float32)
all_sketchp = np.array(sketches, dtype=np.float32)

print(all_imnp.shape)
print(all_sketchp.shape)
print(sketches[0])

save_as_h5(images, sketches)