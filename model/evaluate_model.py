from evaluate.evaluater import Evaluater
from helper.helper_func import load_images

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# real_images = load_images(100, "evauate_images/real", gray=True)
# gen1_gray_images = load_images(100, "evauate_images/gen1", gray=True)
# gen2_gray_images = load_images(100, "evauate_images/gen2", gray=True)

# gen1_rgb_images = load_images(100, "evauate_images/gen1")
gen2_rgb_images = load_images(100, "evauate_images/gen2")

evaluater = Evaluater()

# print("real and gen from ours")

# print(evaluater.calculate_psnr_and_ssim(real_images, gen1_gray_images))
# print("IS Score:", evaluater.calculate_inception_scroe(gen1_rgb_images))

print("IS score from pix2pix + gfpgan")

# print(evaluater.calculate_psnr_and_ssim(real_images, gen2_gray_images))
print("IS Score:", evaluater.calculate_inception_scroe(gen2_rgb_images))
