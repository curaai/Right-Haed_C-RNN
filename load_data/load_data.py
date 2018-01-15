import cv2
import xml.etree.ElementTree as et
import os

class pascalVOC:
    def __init__(self, resize=False, resize_width=500, resize_height=400):
        self.name = 'pascalVOC'

        # number of images in directory
        file_num = len([n for n in os.listdir('./images')])

        self.num_images = file_num
        self.resize = resize
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.images = []
        self.annotations = []

        # self.all_objects = []
        # self.current_object = []
        self.current_batch = 0

        self.bounding_boxes = []
        self.labels = []
        #self.objects = []
        self.image_sizes = []

        self.truth_labels ={'dog': 0,
                      'person': 1,
                      'train': 2,
                      'sofa': 3,
                      'chair': 4,
                      'car': 5,
                      'pottedplant': 6,
                      'diningtable': 7,
                      'horse': 8,
                      'cat': 9,
                      'cow': 10,
                      'bus': 11,
                      'bicycle': 12,
                      'aeroplane': 13,
                      'motorbike': 14,
                      'tvmonitor': 15,
                      'bird': 16,
                      'bottle': 17,
                      'boat': 18,
                      'sheep': 19,
                      }

    # get next batch datas : images, labels, bounding boxes
    def get_next_batch(self, batch_size=10):
        self.images = []
        self.labels = []
        self.bounding_boxes = []
        self.annotations = []

        # to handle 9900 ~ 9963 iteration
        if self.current_batch + batch_size >= self.num_images:
            iterator = range(self.current_batch + 1, self.num_images)
        else:
            iterator = range(self.current_batch + 1, self.current_batch + batch_size + 1)

        for n in iterator:
            img = self.read_image(n)
            root = self.read_annotation(n)

            size = self.get_size(root)
            objects = self.get_objects(root)
            label = self.get_labels(objects)
            bounding_box = self.get_bounding_boxes(objects, size)

            self.images.append(img)
            self.annotations.append(root)

            self.image_sizes.append(size)
            #self.all_objects.append(objects)
            self.labels.append(label)
            self.bounding_boxes.append(bounding_box)

        # set next batch index
        self.current_batch = self.current_batch + batch_size

        return self.images, self.labels, self.bounding_boxes

    # read image at uri
    def read_image(self, number):
        image_uri = './images/' + str(number).zfill(6) + '.jpg'
        img = cv2.imread(image_uri)

        # handle resizing
        if self.resize:
            img = cv2.resize(img, (self.resize_width, self.resize_height))

        return img

    def read_annotation(self, number):
        annotation_uri = './annotations/' + str(number).zfill(6) + '.xml'
        tree = et.parse(annotation_uri)
        return tree.getroot()

    # get one annotation's size data
    def get_size(self, root):
        _size_dict = {}
        size = root.find('size')
        _size_dict['width'] = size[0].text
        _size_dict['height'] = size[1].text
        _size_dict['depth'] = size[2].text
        return _size_dict

    # get one annotation's objects data
    def get_objects(self, root):
        objects = []
        for obj in root.findall('object'):
            new_obj = {}
            bound_box = []

            new_obj['name'] = obj.find('name').text
            new_obj['pose'] = obj.find('name').text
            new_obj['truncated'] = obj.find('truncated').text
            new_obj['difficult'] = obj.find('difficult').text

            for box_element in obj.find('bndbox'):
                bound_box.append(box_element.text)

            new_obj['bndbox'] = bound_box

            objects.append(new_obj)

        return objects

    # get all object's labels based on name
    def get_labels(self, objects):
        labels = []

        for i in range(len(objects)):
            labels.append(self.truth_labels[objects[i]['name']])

        return labels

    # get all object's bounding boxes
    def get_bounding_boxes(self, objects, size):
        bounding_boxes = []
        for i in range(len(objects)):
            xmin = int(objects[i]['bndbox'][0])
            ymin = int(objects[i]['bndbox'][1])
            xmax = int(objects[i]['bndbox'][2])
            ymax = int(objects[i]['bndbox'][3])

            if self.resize:
                im_width = int(size['width'])
                im_height = int(size['height'])
                xmin = int((float(xmin) / float(im_width)) * float(self.resize_width))
                xmax = int((float(xmax) / float(im_width)) * float(self.resize_width))
                ymin = int((float(ymin) / float(im_height)) * float(self.resize_height))
                ymax = int((float(ymax) / float(im_height)) * float(self.resize_height))

            top_left = (xmin, ymin)
            bottom_right = (xmax, ymax)
            bounding_boxes.append([top_left, bottom_right])

        return bounding_boxes