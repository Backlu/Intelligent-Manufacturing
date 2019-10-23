Facial beauty dataset
---------------------------

**Data Set Characteristics:**

    :照片數量: 5500
 
    :分數範圍: 1~5分
    
    :source code: datasets/base.py
    
    :method: load_facialbeauty(img_width=350, img_height=350, qty=None)


**讀取數據Sample Code**

::

    from datasets import load_facialbeauty
    data = load_facialbeauty(qty=1000) 
 
 
**Model Characteristics**

    :Regression Model: InceptionResNetV2
    
    :input size: (350,350,3)
    
    :output size: 1
    
    :pretrain: imagenet pretrain model, keras builtin weight
    
    :image preprocess: normalize (use inception_resnet_v2 preprocess_input function)
    
    :output model format: tflite
    
    :source code: img_facialbeauty.ipynb
    
**Training Hyperparameter**  

    :lose: mse
    
    :optimizer: adam
    
    :batch size: 21
    
    :EarlyStopping: patience=10
    
    :ReduceLROnPlateau: factor=0.5, patience=10
    
    :epoch: 50



 
專案說明：
South China University published a paper and a dataset about “Facial Beauty Prediction”. The data set includes 5500 people that have a score of 1 through 5 of how attractive they are.

..  image:: https://miro.medium.com/max/1372/1*MEG1LZPHtp72xaKuBH-JPg.png
    :height: 400
    :width: 400

References:

- How Attractive Are You in the Eyes of Deep Neural Network? (`link1`_)

- Facial Beauty Prediction (`link2`_)


@inproceedings{liang2017SCUT,
  title     = {SCUT-FBP5500: A Diverse Benchmark Dataset for Multi-Paradigm Facial Beauty Prediction},
  author    = {Liang, Lingyu and Lin, Luojun and Jin, Lianwen and Xie, Duorui and Li, Mengru},
  booktitle={Proc. ICPR},
  year      = {2018}
}

.. _link1: https://towardsdatascience.com/how-attractive-are-you-in-the-eyes-of-deep-neural-network-3d71c0755ccc
.. _link2: https://arxiv.org/abs/1801.06345