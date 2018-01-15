import cv2
import numpy as np

import random

class Tools:
    def __init__(self, dataset_name='pascalVOC'):
        if dataset_name == 'pascalVOC':
            self.box_colors = []

            for n in range(20):
                self.box_colors.append(
                                        (random.randrange(50, 255),
                                         random.randrange(50, 255),
                                         random.randrange(50, 255))
                                      )


    def show_image(self, img, text='Image Show'):
        cv2.imshow(text, img)
        cv2.waitKey(1)
        cv2.destroyAllWindows()

    def save_image(self, img, filename='test.jpg'):
        cv2.imwrite(filename, img)

    def draw_bounding_boxes(self, img, box, label):
        if len(box) != len(label):
            raise IndexError

        for i in range(len(box)):
            cv2.rectangle(img, box[i][0], box[i][1], self.box_colors[label[i]], thickness=2)

        return img

