import cv2
import xml.etree.ElementTree as et
import random
import os

resize_width, resize_height = 500, 400
resize = True

num_images = 100

bounding_boxes = []
crop_images = []
labels = []

class_names = {}
classes_number = 0

for n in range(1, num_images+1):
    objects = []
    image_size = {}
    image_uri = './images/' + str(n).zfill(6) + '.jpg'
    annotation_uri = './annotations/' + str(n).zfill(6) + '.xml'
    img = cv2.imread(image_uri)
    tree = et.parse(annotation_uri)

    root = tree.getroot()

    # get size infomations : width, height, depth
    size = root.find("size")
    image_size['width'] = size[0].text
    image_size['height'] = size[1].text
    image_size['depth'] = size[2].text

    # get all 'object' sub-elements
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
        objects.append(new_obj)

    # get image
    # draw all ground-truth bounding
    if resize:
        img = cv2.resize(img, (resize_width, resize_height))

    bounding_boxes = []
    crop_images = []
    for i in range(len(objects)):
        xmin = int((objects[i]['bndbox'])[0])
        ymin = int((objects[i]['bndbox'])[1])
        xmax = int((objects[i]['bndbox'])[2])
        ymax = int((objects[i]['bndbox'])[3])

        if resize:
            im_width = int(image_size['width'])
            im_height = int(image_size['height'])
            xmin = int((float(xmin) / float(im_width)) * float(resize_width))
            xmax = int((float(xmax) / float(im_width)) * float(resize_width))
            ymin = int((float(ymin) / float(im_height)) * float(resize_height))
            ymax = int((float(ymax) / float(im_height)) * float(resize_height))

        top_left = (xmin, ymin)
        bottom_right = (xmax, ymax)
        bounding_boxes.append([top_left, bottom_right])

        crop_images.append(img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]])

    for i in range(len(crop_images)):
        cv2.imwrite('./image_crop_tests/' + str(n).zfill(6) + '_crop_' + str(i) + '.jpg', crop_images[i])

    for i in range(len(bounding_boxes)):
        r, g, b = random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)
        cv2.rectangle(img, bounding_boxes[i][0], bounding_boxes[i][1], (r, g, b), 3)

    cv2.imwrite('./image_ground_truths/' + str(n).zfill(6) + '_truth.jpg', img)

    for i in range(len(objects)):
        if objects[i].get('name') not in class_names:
            class_names[objects[i].get('name')] = classes_number
            classes_number += 1

    if n % 100 == 0:
        print(n, 'Complete')

print(class_names)