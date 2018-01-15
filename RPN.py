import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


def type1_convert_box_expression(x, y, w, h):
    """x, y, w, h -> (x1, y1, x2, y2)"""
    x1 = x - w / 2
    x2 = x + w / 2
    y1 = y - h / 2
    y2 = y + h / 2
    return x1, y1, x2, y2


def type2_convert_box_expression(x1, y1, x2, y2):
    """x1, y1, x2, y2 -> (x, y, w, h)"""
    small_x = x1 if x1 < x2 else x2
    small_y = y1 if y1 < y2 else y2

    w = abs(x1 - x2)
    h = abs(y1 - y2)
    x = small_x + w / 2
    y = small_y + h / 2

    return x, y, w, h


def intersection_over_union(box1, box2):
    """
    get iou between two boxes
    it needs x1, y1, x2, y2. Not x, y, w h
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    area_intersection = (x2 - x1) * (y2 - y1)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[2])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[2])
    area_union = area_box1 + area_box2 - area_intersection

    iou = area_intersection / area_union
    return iou


def non_max_suppression_fast(boxes, overlapThresh):
    # overlapThresh usually use 0.05
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


def generate_anchor(feat_shape, image_shape, anchor_sizes, anchor_ratios):
    """
    s: stride
    w: width, h: height
    x, y: coordinate(center of anchor)
    f: feature
    i: image

    :return: shape(feature w*h, 9, 4)
    """

    w_f, h_f = feat_shape[0], feat_shape[1]
    w_i, h_i = image_shape[0], image_shape[1]

    anchors = list()

    for anchor_size in anchor_sizes:
        for anchor_ratio in anchor_ratios:

            n = feat_shape[0] * feat_shape[1]

            w_s = int(w_i / w_f)
            h_s = int(h_i / h_f)

            w = np.full(n, anchor_size * anchor_ratio[0])
            h = np.full(n, anchor_size * anchor_ratio[1])
            x = np.array([[x] * h_f for x in range(w_f)]).reshape((-1))
            y = np.array([list(range(w_f)) * 20]).reshape((-1))
            x = (x + 0.5) * w_s - w * 0.5
            y = (y + 0.5) * h_s - h * 0.5

            # if anchor over image boundary
            x[x < 0] = 0
            y[y < 0] = 0
            w = np.minimum(w, w_i - x)
            h = np.minimum(h, h_i - y)

            x = np.expand_dims(x, 1)
            y = np.expand_dims(y, 1)
            w = np.expand_dims(w, 1)
            h = np.expand_dims(h, 1)

            anchors.append(np.concatenate((x, y, w, h), axis=1))

    return np.array(anchors, np.int32)


def labeling_anchor(anchor, gt_bbox, gt_cls):
    x,y,w,h = anchor
    anchor_bbox = type1_convert_box_expression(x, y, w, h)

    iou = intersection_over_union(anchor_bbox, gt_bbox)
    # type 1: positive sample, 0:negative sample
    if iou > 0.7:
        type = 1
        anchor_cls = gt_cls
    elif iou < 0.2:
        type = 0
        anchor_cls = -1
    else:
        return None

    return type, anchor_cls


def anchor_generate_test():
    # test generate_anchor

    feat_shape = (20, 20)
    image_shape = (640, 640)
    anchor_size = [500]
    anchor_ratio = [(1 / 2 ** 0.5, 2 ** 0.5), (1, 1), (2 ** 0.5, 1 / 2 ** 0.5)]
    
    generate_anchor(feat_shape, image_shape, anchor_size, anchor_ratio)


class RPN:
    def __init__(self, last_feat):
        self.last_feat = last_feat
        self.anchor_sizes = [64, 128, 256]
        self.anchor_ratios = [(1 / 2 ** 0.5, 2 ** 0.5), (1, 1), (2 ** 0.5, 1 / 2 ** 0.5)]

        self.name = 'RPN'
        self.build()

    def build(self):
        with tf.variable_scope(self.name):
            conv = slim.conv2d(self.last_feat, 256, [5, 5])
            rpn_cls = slim.conv2d(conv, 2 * len(self.anchor_sizes) * len(self.anchor_ratios), [1, 1])
            rpn_cls = tf.reshape(rpn_cls, [-1, len(self.anchor_sizes) * len(self.anchor_ratios), 2])
            rpn_reg = slim.conv2d(conv, 4 * len(self.anchor_sizes) * len(self.anchor_ratios), [1, 1])
            rpn_reg = tf.reshape(rpn_reg, [-1, len(self.anchor_sizes) * len(self.anchor_ratios), 4])

            tf.nn.softmax_cross_entropy_with_logits()



if __name__ == '__main__':
    anchor_generate_test()
