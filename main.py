import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataloader import DataLoader
import os
import glob
from pathlib import Path
from resizing_BB import BB_resize

from splitting_img import splitting

if __name__ == '__main__':
    
    #splitting of images 
    images_path = 'images2'
    list_images = []
    for filename in os.listdir(images_path):
        file_path = os.path.join(images_path, filename)
        list_images.append(file_path)
    
    list_annots = []
    annots_path = 'labels2'
    for filename in os.listdir(annots_path):
        file_path = os.path.join(annots_path, filename)
        list_annots.append(file_path)

    dataloader = DataLoader()
    all_boundingbox_coords, all_labels = dataloader.data(annot_path=annots_path)
    
    resize = BB_resize()
    split_image = splitting()
    for i, image_path in enumerate(list_images):
        bounding_boxes = np.array(all_boundingbox_coords[i])
        labels_boxes = np.array(all_labels[i])
        image = cv2.imread(f'{image_path}')
        sub_images, sub_boxes, labels = split_image.split(image, bounding_boxes=bounding_boxes, labels=labels_boxes)
        file_name = image_path[8:13]
        print(file_name)

        for subplot in range(9):
            #img = dataloader.ImageVisualization(sub_images[subplot], sub_boxes[subplot])
            #print(i)
            #print((np.array(sub_boxes[i])))
            res_img, res_box, label_finals = resize.resize(sub_images[subplot], sub_boxes[subplot], labels=labels)
            res_box = np.array(res_box)
            #print(res_box)
            img = dataloader.ImageVisualization(res_img, res_box)

            #print(f'images_resized\{file_name}_{subplot}')
            cv2.imwrite(f'images_resized3\{file_name}_{subplot}.png', img)
            file_txt = f'{file_name}_{subplot}.txt'
            folder_name = 'bounding_box_resized3'

            with open(os.path.join(folder_name, file_txt), 'w') as file:
                i=0
                for annot in res_box:
                    
                    for item in annot:
                        file.write("%s " % item)
                    file.write("%s" % label_finals[subplot][i])
                    i = i+1
                    file.write("\n")

             
                




    #all_annotations

