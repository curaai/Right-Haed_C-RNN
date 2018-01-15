# In dev...

1. Download Pascal VOC 2007 devkit on web. ([Link](https://pjreddie.com/projects/pascal-voc-dataset-mirror/))

2. Make soft link './images' which links 'VOCdevkit/VOC2007/JPEGImages' in the directory same with your python source file.

3. Make soft link './annotations' which links VOCdevkit/VOC2007/Annotations' in the directory same with your source file.

4. Make directory image_crop_tests in the directory same with your source file.

5. Make directory image_ground_truths in the directory same with your source file.

### Memo

Can we give label to model by just concatenating labels? Label classification is work of RPN?

Surprisingly, cv2 image is based on numpy.ndarray!
