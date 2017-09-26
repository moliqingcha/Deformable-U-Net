import scipy
import scipy.misc
import numpy as np

def imsave(image, path):
    label_colours = [
        (0,0,0),
        (0,0,64),(0,64,0),(64,0,0),(0,0,128),(0,128,0),(128,0,0),(0,0,192),(0,192,0),(192,0,0),
        (0,64,64),(64,0,64),(64,64,0),(0,128,128),(128,0,128),(128,128,0),(0,192,192),(192,0,192),(192,192,0),
        (0,64,128),(64,0,128),(64,128,0),(0,128,64),(128,0,64),(128,64,0),
        (0,64,192),(64,0,192),(64,192,0),(0,192,64),(192,0,64),(192,64,0),
        (0,128,192),(128,0,192)]
    images = np.ones(list(image.shape)+[3])
    for j_, j in enumerate(image):
        for k_, k in enumerate(j):
            if k < 33:
                images[j_, k_] = label_colours[int(k)]
    scipy.misc.imsave(path, images)


