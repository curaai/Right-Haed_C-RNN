import numpy as np


# get iou between two boxes
# it's x1, y1, x2, y2. Not x, y, w h
def intersection_over_union(box1, box2):
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


def anchor_generate_test():
    # test generate_anchor

    feat_shape = (20, 20)
    image_shape = (640, 640)
    anchor_size = [500]
    anchor_ratio = [(1 / 2 ** 0.5, 2 ** 0.5), (1, 1), (2 ** 0.5, 1 / 2 ** 0.5)]
    
    generate_anchor(feat_shape, image_shape, anchor_size, anchor_ratio)


if __name__ == '__main__':
    anchor_generate_test()