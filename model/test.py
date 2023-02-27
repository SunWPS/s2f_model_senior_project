import cv2
import numpy as np
import sys

from s2fgenerator.model import Generator
from enhancer.gfpgan import GFPGAN
from helper.helper_func import predict_one_img


def main():
    try:
        img_path = sys.argv[1]
    except IndexError:
        print('*****Need input image path*****')
        return
    
    try:
        ver = sys.argv[2]
    except IndexError:
        print("******Need model version**********")
        print("1. old")
        print("2. new")
        return
    ## Load model
    
    print(ver)
    if ver == "1":
        generator = Generator().load_model('model_saved/generator_weight_1.h5')
        gfpgan = GFPGAN('model_saved/GFPGANv1.3.pth')
    else:
        generator = Generator().load_model('model_saved/generator_weight.h5')
        gfpgan = GFPGAN('model_saved/GFPGANv1.4.pth')
    
    ## Predict stage 1 by ours model
    if ver == "1":
        image = cv2.imread(img_path, 0)
        predict_one_img(generator, image, 'img_output/1_generated/gen2.png', base_black_img_path="og.jpg", b_level=True)
    else:
        image = cv2.imread(img_path, 0)
        predict_one_img(generator, image, 'img_output/1_generated/gen2.png', base_black_img_path="og.jpg", b_level=True)
    
    ## enchance image by gfpgan
    gen_image = cv2.imread('img_output/1_generated/gen2.png', cv2.IMREAD_COLOR)
    gfpgan.enhance(gen_image, 'img_output/2_enhanced/enhanced2.png')
    
    ## enchance image by gfpgan
    # gen_image = cv2.imread('img_output/2_enhanced/enhanced2.png', cv2.IMREAD_COLOR)
    # gfpgan.enhance(gen_image, 'img_output/3_enhanced/enhanced3.png')
    
    image = cv2.imread(img_path, 0)
    img = cv2.imread('img_output/3_enhanced/enhanced3.png')
    
    image2 = cv2.resize(image2, (512,512), interpolation = cv2.INTER_AREA)
    
    cv2.imshow("sketch", image)
    cv2.imshow("gen", image2)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    print("Finish")
    return
    

if __name__ == '__main__':
    main()