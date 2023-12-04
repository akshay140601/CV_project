import cv2
from matplotlib import pyplot as plt
import numpy as np

from dataloader import DataLoader

class BB_resize():
    def __init__(self) -> None:
        self.new_size = (1024, 1024)

    def resize(self, image, bounding_boxes,labels):
        resized_image = cv2.resize(image, self.new_size)

        scale_x = self.new_size[0] / image.shape[1]
        scale_y = self.new_size[1] / image.shape[0]

        resized_boxes = []
        for box in bounding_boxes:
            box = np.reshape(box, (4, 2))
            box[:, 0] *= scale_x
            box[:, 1] *= scale_y
            resized_boxes.append(box.flatten())

        return resized_image, resized_boxes, labels
    
'''if __name__ == '__main__':
    dataloader = DataLoader()
    coords = dataloader.data('labelTxt-v1.5')
    coords_0 = np.array(coords[0])
    img_0 = cv2.imread('Images//P0000.png')
    resize = BB_resize()
    original_img = dataloader.ImageVisualization(img_0, coords_0)
    res_img, res_box = resize.resize(img_0, coords_0)
    print(res_box)'''
    #new_img = dataloader.ImageVisualization(res_img, res_box)
