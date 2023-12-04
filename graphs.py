import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataloader import DataLoader
import os
import glob
from pathlib import Path
from resizing_BB import BB_resize
from splitting_img import splitting
from collections import Counter
import itertools

if __name__ == '__main__':
    
    #splitting of images 
    images_path = 'images2'
    list_images = []
    list_labels = []
    for filename in os.listdir(images_path):
        file_path = os.path.join(images_path, filename)
        list_images.append(file_path)
    
    list_annots = []
    annots_path = 'labels2'
    dataloader = DataLoader()
    for filename in os.listdir(annots_path):
        file_path = os.path.join(annots_path, filename)
        list_annots.append(file_path)
        all_boundingbox_coords, all_labels = dataloader.data(annot_path=annots_path)
        for i in all_labels:
            list_labels.append(i)
    print((np.array(list_labels)).shape)
    flat_annotations = list(itertools.chain.from_iterable(
    [ann] if not isinstance(ann, list) else map(str, ann) for ann in list_labels
))
    annotation_counts = Counter(flat_annotations)

    # Extract labels and counts for the pie chart
    labels = list(annotation_counts.keys())
    counts = list(annotation_counts.values())

    plt.figure(figsize=(10, 6))  # Set the size of the bar graph
    bars = plt.bar(labels, counts, color='gray')
    plt.xlabel('Annotations')  # Label for x-axis
    plt.ylabel('Frequency')  # Label for y-axis
    plt.title('Number of Annotations in the Dataset')  # Add a title
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability

    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, count, ha='center', va='bottom')
    
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()


    
    