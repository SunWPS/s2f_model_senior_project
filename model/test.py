from s2fgenerator.model import Generator
from helper.helper_func import predict_one_img
import cv2

generator = Generator().load_model("model_saved/generator_weight.h5")

img = cv2.imread("test_images/input/test_img.jpg")
img = img[50:, 0:]

predict_one_img(generator, img, "test_images/output/gen_img.jpg")