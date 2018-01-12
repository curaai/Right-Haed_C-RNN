import cv2
import xml.etree.ElementTree as et
import random
import os

bounding_boxes = []
crop_images = []
labels = []

class_names = {}
classes_number = 0

class pascalVOC:
    def __init__(self, resize=True, resize_width=500, resize_height=400):
        self.num_images = 9963
        self.resize = resize
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.all_objects = []
        self.current_object = []
        self.current_batch = 0

    def get_next_batch(self, batch_size):
        batch = []
        for i in range(1, batch_size+1):
            img = self.get_image(self.current_batch * 100 + i)
            ano = self.get_annotation_root(self.current_batch * 100 + i)
            batch.append(img)

        return batch

    def get_image(self, number):
        image_uri = './images/' + str(number).zfill(6) + '.jpg'
        img = cv2.imread(image_uri)
        return img

    def get_annotation_root(self, number):
        annotation_uri = './annotations/' + str(number).zfill(6) + '.xml'
        annotation = et.parse(annotation_uri)
        return annotation.getroot()

    def get_image_size(self, root):
        image_size = {}
        # get size infomations : width, height, depth
        size = root.find("size")
        image_size['width'] = size[0].text
        image_size['height'] = size[1].text
        image_size['depth'] = size[2].text

        return image_size

    def find_all_objects(self, root):
        for obj in root.findall('object'):
            new_obj = {}
            bndbox = []

            # get that object's elements : name, pose, truncated, difficult
            new_obj['name'] = obj.find('name').text
            new_obj['pose'] = obj.find('pose').text
            new_obj['truncated'] = obj.find('truncated').text
            new_obj['difficult'] = obj.find('difficult').text

            # get bndbox sub-element
            # get that object's bounding box informations : xmin, ymin, xmax, ymax
            for box_element in obj.find('bndbox'):
                bndbox.append(box_element.text)

            # append to objects array
            new_obj['bndbox'] = bndbox
            self.current_object.append(new_obj)

        return self.current_object

    def make_bounding_boxes(self, img, image_size):
        if self.resize:
            img = cv2.resize(img, (self.resize_width, self.resize_height))

        bounding_boxes = []

        for i in range(len(self.current_object)):
            xmin = int((self.current_object[i]['bndbox'])[0])
            ymin = int((self.current_object[i]['bndbox'])[1])
            xmax = int((self.current_object[i]['bndbox'])[2])
            ymax = int((self.current_object[i]['bndbox'])[3])

            if self.resize:
                im_width = int(image_size['width'])
                im_height = int(image_size['height'])
                xmin = int((float(xmin) / float(im_width)) * float(self.resize_width))
                xmax = int((float(xmax) / float(im_width)) * float(self.resize_width))
                ymin = int((float(ymin) / float(im_height)) * float(self.resize_height))
                ymax = int((float(ymax) / float(im_height)) * float(self.resize_height))

            top_left = (xmin, ymin)
            bottom_right = (xmax, ymax)
            bounding_boxes.append([top_left, bottom_right])

        return bounding_boxes

    def crop_images(self, img, bounding_boxes):
        crop_images = []
        for i in range(len(bounding_boxes)):
            crop_images.append(img[bounding_boxes[i][0][1]:bounding_boxes[i][1][1], bounding_boxes[i][0][0]:bounding_boxes[i][1][0]])

        return crop_images

    def save_croped_images(self, crop_images, number):
        for i in range(len(crop_images)):
            cv2.imwrite('./image_crop_tests/' + str(number).zfill(6) + '_crop_' + str(i) + '.jpg', crop_images[i])

    def draw_ground_truth(self, img, bounding_boxes):
        for i in range(len(bounding_boxes)):
            r, g, b = random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)
            cv2.rectangle(img, bounding_boxes[i][0], bounding_boxes[i][1], (r, g, b), 3)

    def save_ground_truth_image(self, img, number):
        cv2.imwrite('./image_ground_truths/' + str(number).zfill(6) + '_truth.jpg', img)

    def get_all_classes(self, ):

    for n in range(1, num_images+1):
        objects = []
        img = get_image(n)
        root = get_annotation_root(n)

        image_size = get_image_size(root)



        for i in range(len(objects)):
            if objects[i].get('name') not in class_names:
                class_names[objects[i].get('name')] = classes_number
                classes_number += 1

        if n % 100 == 0:
            print(n, 'Complete')

    print(class_names)