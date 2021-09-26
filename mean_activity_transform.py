import cv2
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
        :param img: PIL): Image 

        :return: mean-adjusted image
        """
        
        return cv2.subtract(img, Image.open("./mean_activity.JPEG"))

    def __repr__(self):
        return self.__class__.__name__+'()'