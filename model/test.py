import cv2
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
    
    ## Load model
    generator = Generator().load_model('model_saved/generator_weight.h5')
    gfpgan = GFPGAN('model_saved/GFPGANv1.3.pth')
    
    ## Predict stage 1 by ours model
    image = cv2.imread(img_path, 0)
    predict_one_img(generator, image, 'img_output/1_generated/gen2.png')
    
    ## enchance image by gfpgan
    gen_image = cv2.imread('img_output/1_generated/gen2.png', cv2.IMREAD_COLOR)
    gfpgan.enhance(gen_image, 'img_output/2_enhanced/enhanced2.png')
    
    print("Finish")
    return
    

if __name__ == '__main__':
    main()