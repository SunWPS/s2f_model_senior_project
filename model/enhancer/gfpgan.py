from gfpgan import GFPGANer
from basicsr.utils import imwrite


class GFPGAN:
    """
        GFPGAN for enchance image
        Credit: https://github.com/TencentARC/GFPGAN
        
        @InProceedings{wang2021gfpgan,
            author = {Xintao Wang and Yu Li and Honglun Zhang and Ying Shan},
            title = {Towards Real-World Blind Face Restoration with Generative Facial Prior},
            booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
            year = {2021}
        }
        
        :param model_path: pre-train GFPGAN path
        :type model_path: string
    """
    def __init__(self, model_path):
        self.enhancer = GFPGANer(model_path=model_path,
                                 upscale=2,
                                 arch='clean',
                                 channel_multiplier=2,
                                 bg_upsampler=None)
    
    
    def enhance(self, img, out_path):
        """
            Enhance image
            
            :param img: input image
            :type img: string
            
            :param out_path: output path for saving enchanced image
            :type outpath: string
        """
        _, _, enhanced_img = self.enhancer.enhance(
            img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=0.5)
        
        imwrite(enhanced_img, out_path)
        return enhanced_img
