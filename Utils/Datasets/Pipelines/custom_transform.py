from xml.etree.ElementTree import SubElement
import cv2
import numpy as np

__all__ = ['Img2Edge', 'Img2Gray', 'ToSingleChannel']

class Img2Edge(object):
    def __init__(self):
        pass

    def __call__(self, results):
        img = cv2.cvtColor(results['img'], cv2.COLOR_RGB2GRAY)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        sobelCombined = cv2.Canny(img, 30, 150)
        results['img'] = np.tile(sobelCombined[..., None], [1, 1, 3])
        #print (results['img'].shape)
        if False:
            import matplotlib.pyplot as plt
            plt.gray()
            plt.subplot(2,1,1)
            plt.imshow(img)
            plt.subplot(2,1,2)
            plt.imshow(sobelCombined)
            plt.show()
            exit()
        
        return results

class Img2Gray(object):
    def __init__(self):
        pass
    
    def __call__(self, results):
        img = cv2.cvtColor(results['img'], cv2.COLOR_RGB2GRAY)[..., None]
        img = np.tile(img, [1, 1, 3])
        results['img'] = img

        return results


class ToSingleChannel(object):
    def __init__(self):
        pass

    def __call__(self, results):
        img = results['img']
        results['img'] = img[..., 0:1]

        return results