import cv2

from s2fgenerator.model import Generator
from enhancer.gfpgan import GFPGAN
from helper.helper_func import predict_one_img


def main():
    ## Load model
    generator = Generator().load_model('model_saved/t_generator_0.h5')
    gfpgan = GFPGAN('model_saved/GFPGANv1.4.pth')
    
    ## Predict stage 1 by ours model
    for i in range(1,101):
        image = cv2.imread(f'evauate_images/sketches/{i}.jpg', 0)
        predict_one_img(generator, image, f'evauate_images/gen1/{i}.jpg')
    
        ## enchance image by gfpgan
        gen_image = cv2.imread(f'evauate_images/gen1/{i}.jpg', cv2.IMREAD_COLOR)
        gfpgan.enhance(gen_image, f'evauate_images/gen2/{i}.jpg')
        
        gen_image = cv2.imread(f'evauate_images/gen2/{i}.jpg', cv2.IMREAD_COLOR)
        gfpgan.enhance(gen_image, f'evauate_images/gen2/{i}.jpg')
    
        print(f"image {i} finished")
    
    print("Finish all")
    return
    

if __name__ == '__main__':
    main()
