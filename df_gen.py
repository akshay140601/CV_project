import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import cv2

annot_path = 'C:\\Subjects\\Subjects\\Object Extraction Project\\bounding_boxes_resized'
file_paths = glob.glob(annot_path + '/*.txt')
df = {'image_path': [],'x1': [], 'y1': [], 'x2': [], 'y2': [], 'x3': [], 'y3': [], 'x4': [], 'y4': [], 'outcome': [] }
#df_with_coords = {'image_path': [],'coords': [], 'outcome': []}

for file_path in file_paths:
    image_path = f'C:\\Subjects\\Subjects\\Object Extraction Project\\images_resized\\{file_path[70:77]}.png'
    with open(file_path,'r') as file:
        lines = 0
        for line_number, line in enumerate(file):
            line_value = line.rstrip().split()
            coords = [float(num) for num in line_value[:8]]
            df['image_path'].append(image_path)
            df['x1'].append(coords[0])
            df['y1'].append(coords[1])
            df['x2'].append(coords[2])
            df['y2'].append(coords[3])
            df['x3'].append(coords[4])
            df['y3'].append(coords[5])
            df['x4'].append(coords[6])
            df['y4'].append(coords[7])
            df['outcome'].append(line_value[8])

df_from_dict = pd.DataFrame.from_dict(df)

categories = {'small-vehicle':0, 
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
              }
df_from_dict['outcome'] = df_from_dict['outcome'].apply(lambda x:  categories[x])
#df_from_dict.to_csv('dataframe.csv', index=False, encoding='utf-8')
#print(df_from_dict)
