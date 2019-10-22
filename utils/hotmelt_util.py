#!/usr/bin/env python
# coding: utf-8

# Copyright © 2019 Hsu Shih-Chieh

import numpy as np
from keras.preprocessing.image import ImageDataGenerator


def overSampling(data_x, data_y):
    '''
    img_hotmelt.ipynb裡面使用, 當類別的影像數量不平衡時, 用oversampling的方式增加影像數量
    '''
    print('before:',data_x.shape,' ', data_y.shape)
    data_y_ = np.argmax(data_y, axis=1)
    unique, counts = np.unique(data_y_, return_counts=True)
    newqty=int(np.mean(counts)*2)
    print('over sampling qty:',newqty)
    datagen_aug = ImageDataGenerator(
        #zoom_range=0.98,
        vertical_flip=True, horizontal_flip=True,
        #shear_range=0.5,
        rotation_range=0, fill_mode='nearest',
        width_shift_range=0.02, height_shift_range=0.02,
        rescale=None,
    )
    oversample_idx = np.where(counts<newqty)[0]
    for i in oversample_idx:
        dataqty=counts[i]
        defect_idx = np.where( data_y_==i)[0]
        defect_x = data_x[defect_idx]
        datagen_aug.fit(defect_x)
        for x_batch, y_batch in datagen_aug.flow(defect_x, data_y[defect_idx], shuffle=False):
            data_x = np.vstack((data_x,x_batch))
            data_y = np.vstack((data_y,y_batch))
            dataqty = dataqty+x_batch.shape[0]
            if dataqty > newqty:
                break
    print('after:',data_x.shape,' ', data_y.shape)
    return data_x, data_y


import os, colorsys
from keras import backend as K
from keras.layers import Input
from .yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss, yolo_eval, yolo_head
from .yolo3.utils import get_random_data, letterbox_image


class YOLOV3(object):
    '''
    img_hotmelt.ipynb裡面使用, 用來抓出影像的ROI
    
    methods:
        - detect_image: 輸入影像(array), 輸出boxes
        - predict_roi: 輸入影像(Image), 輸出 ROI image array
    '''
    _defaults = {
        "model_path": 'model/hotmelt_yolov3weight.h5',
        "score" : 0.01,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = ['roi']
        anchors_str= '10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326'
        self.anchors = np.array([float(x) for x in anchors_str.split(',')]).reshape(-1, 2)
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()


    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        self.yolo_model = yolo_body(Input(shape=(None,None,3)), 3, 1)
        self.yolo_model.load_weights('model/hotmelt_yolov3weight.h5') # make sure model, anchors and classes match

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        #print(image_data.shape,image.size[1],image.size[0])
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        return out_boxes
    
    def predict_roi(self, img):
        boxes = self.detect_image(img)
        roi_size = [224,224]
        if len(boxes)==0:
            roiimg = np.array(img.resize(roi_size))
            print('predict_roi-warning-')
        else:
            upper, left, lower, right = boxes[0]
            roiimg = img.crop((left, upper, right, lower))        
            roiimg = np.array(roiimg.resize(roi_size))
        return roiimg
    

    def close_session(self):
        self.sess.close()





