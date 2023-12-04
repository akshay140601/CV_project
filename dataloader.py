import dataclasses
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

class DataLoader():
    def __init__(self):
        pass

    def data(self, annot_path):
        annot_path = Path(annot_path)
        file_list = list(annot_path.glob('*.txt'))
        annots_in_all_files = []
        labels_in_all_files = []
        '''categories = {'small-vehicle':0, 
              'storage-tank' :1,
              'plane':2,
              'large-vehicle':3,
              'ship':4,
              'harbor':5,
              'tennis-court':6,
              'swimming-pool':7,
              'baseball-diamond':8,
              'soccer-ball-field':9,
              'roundabout':10,
              'basketball-court':11,
              'ground-track-field':12,
              'helicopter':13,
              'bridge':14,
              'container-crane':15
              }'''
        
        for file_path in file_list:
            annots_in_file= []
            label_in_file=[]
            with open(file_path, 'r') as file:
                lines = 0
                for line_number, line in enumerate(file, start=1):
                    lines += 1  
                    if line_number <= 2:
                        continue
                    line_value = line.rstrip().split()
                    coords = [float(num) for num in line_value[:8]]
                    label = line_value[8]
                    label_in_file.append(label)
                    annots_in_file.append(coords)

                lines+=1
            labels_in_all_files.append(label_in_file)
            annots_in_all_files.append(annots_in_file)
            #print(labels_in_all_files)
        
        
        

        return annots_in_all_files, labels_in_all_files 

    def ImageVisualization(self, image, coords):
        for coord in coords:
            corners = coord.astype(int)
            corners = (corners).reshape(4,2) 
            cv2.polylines(image, [corners.astype(np.int32)], True , (0,255,0), 2)
        plt.imshow(image)
        plt.axis('off')
        plt.show()
        
        return image


'''if __name__ =='__main__':
    dataloader = DataLoader()
    coords, labels = dataloader.data('label_txt')
    image_0_coords = coords[0]
    labels_0 = labels[0]
    #print(len(labels_0))
    image_0_coords = (np.array(image_0_coords))
    shape = image_0_coords.shape[]
    image_0_coords = image_0_coords.reshape(323,4,2)
    #print(image_0_coords)
    image_0 = cv2.imread('images\P0066.png')
    image_0_with_bounding_boxes = dataloader.ImageVisualization(image_0, image_0_coords)
    #plt.imshow(image_0_with_bounding_boxes)
    #plt.gca().set_aspect('auto')
    #plt.show()'''