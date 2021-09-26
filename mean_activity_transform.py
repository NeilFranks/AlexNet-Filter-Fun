import numpy as np
from PIL import Image

class mean_activity_transform(object):
    """
    From the AlexNet paper:
    "We did not pre-process the images in any other way, except for subtracting the mean activity 
    over the training set from each pixel. 
    So we trained our network on the (centered) raw RGB values of the pixels.
    """  
    
    def __call__(self, img):
        """
        :param img: tensor image

        :return: mean-adjusted image
        """
         # just work with the center 224x224 portion of the 256x256 average image... whoops
        average_img = (np.asarray(
            Image.open("./mean_activity.JPEG"), 
            dtype=np.float64)/255)[16:240, 16:240]
        
        return img - np.reshape(average_img, img.shape)

    def __repr__(self):
        return self.__class__.__name__+'()'