import numpy as np

class intensity_transform(object):
    """
    From the AlexNet paper:
    "The second form of data augmentation consists of altering the intensities of the RGB channels in
    training images. Specifically, we perform PCA on the set of RGB pixel values throughout the
    ImageNet training set. To each training image, we add multiples of the found principal components,
    with magnitudes proportional to the corresponding eigenvalues times a random variable drawn from
    a Gaussian with mean zero and standard deviation 0.1. 
    
    Therefore to each RGB image pixel Ixy = [IRxy, IGxy, IBxy]T 
    we add the following quantity: [p1, p2, p3][α1λ1, α2λ2, α3λ3]T
    
    where pi and λi are ith eigenvector and eigenvalue of the 3 × 3 covariance matrix of RGB pixel
    values, respectively, and αi is the aforementioned random variable. 
    
    Each αi is drawn only once for all the pixels of a particular training image until that image 
    is used for training again, at which point it is re-drawn. 
    
    This scheme approximately captures an important property of natural images, namely, that object 
    identity is invariant to changes in the intensity and color of the illumination. 
    This scheme reduces the top-1 error rate by over 1%.
    """  
    
    def __call__(self, img):
        """
        :param img: PIL): Image 

        :return: intensity-adjusted image
        """
        # apparently faster to work with values 0-1
        orig_img = img.astype(float).copy()/255.0

        # flatten image to columns of RGB
        img_rs = img.reshape(-1, 3)

        # center mean
        img_centered = img_rs - np.mean(img_rs, axis=0)

        # eigen values and eigen vectors
        eig_vals, eig_vecs = np.linalg.eigh(
            np.cov(img_centered, rowvar=False)
            )

        # sort values and vector
        sort_perm = eig_vals[::-1].argsort()
        eig_vals[::-1].sort()
        eig_vecs = eig_vecs[:, sort_perm]

        # [p1, p2, p3][α1λ1, α2λ2, α3λ3]T
        quanitity_to_add = np.column_stack((eig_vecs)) * (np.random.normal(0, 0.1)* eig_vals[:])

        for idx in range(3):   # RGB
            orig_img[..., idx] += quanitity_to_add[idx]

        # get it back to 0-255
        return np.clip(orig_img, 0.0, 255.0).astype(np.uint8)

    def __repr__(self):
        return self.__class__.__name__+'()'