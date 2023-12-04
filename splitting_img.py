import cv2
from matplotlib import pyplot as plt
import numpy as np

from dataloader import DataLoader

class splitting():
    def __init__(self) -> None:
        self.num_rows = 3
        self.num_cols = 3

    def split(self, image, bounding_boxes, labels):
        img_height, img_width = image.shape[:2]
        sub_img_height = img_height // self.num_rows
        sub_img_width = img_width // self.num_cols

        # Create a list to store the sub-images
        sub_images = []
        sub_boxes = []
        label_boxes = []

        for row in range(self.num_rows):
            for col in range(self.num_cols):
                # Define the region of interest for the current sub-image
                y_start = row * sub_img_height
                y_end = (row + 1) * sub_img_height
                x_start = col * sub_img_width
                x_end = (col + 1) * sub_img_width
                current_sub_boxes = []
                current_labels = []
                for idx, box in enumerate(bounding_boxes):
                    box = box.reshape((4, 2))
                    if (
                        np.min(box[:, 0]) >= x_start and np.max(box[:, 0]) <= x_end and
                        np.min(box[:, 1]) >= y_start and np.max(box[:, 1]) <= y_end
                    ):
                        adjusted_box = box - np.array([x_start, y_start])
                        current_sub_boxes.append(adjusted_box)
                        current_labels.append(labels[idx])

                label_boxes.append(current_labels)
                sub_images.append(image[y_start:y_end, x_start:x_end])
                sub_boxes.append(current_sub_boxes)
        
        return sub_images, sub_boxes, label_boxes
    
'''if __name__ == '__main__':
    dataloader = DataLoader()
    coords = dataloader.data('labelTxt')
    coords_0 = np.array(coords[0])
    print(coords_0.shape)
    img_0 = cv2.imread('Images//P0000.png')
    split_inst = splitting()
    sub_images, sub_boxes = split_inst.split(img_0, coords_0)
    #print(np.array(sub_boxes[1]))
    img = dataloader.ImageVisualization(sub_images[4], sub_boxes[4])
    #print(len(sub_boxes[9]))

    fig, axes = plt.subplots(5, 5, figsize=(10, 10))

    for i, (sub_img, sub_boxes_per_sub_img) in enumerate(zip(sub_images, sub_boxes)):
        row, col = divmod(i, 5)
        axes[row, col].imshow(cv2.cvtColor(sub_img, cv2.COLOR_BGR2RGB))

        # Draw bounding boxes for all objects in the current sub-image
        for box in sub_boxes_per_sub_img:
            for j in range(4):
                x = box[j][0]
                y = box[j][1]
                axes[row, col].plot(x, y, 'g-', lw=2)
                axes[row, col].plot(x, y, 'go', markersize=5)

        axes[row, col].axis('off')

    plt.show()'''

'''    for i, sub_img in enumerate(sub_images):
        plt.imshow(sub_img)
        plt.pause(10)'''
