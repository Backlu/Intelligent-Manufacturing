

絕緣片 Defect dataset
---------------------------

**Data Set Characteristics::**

    :raw data format: jpg image
    
    :labelme format: 使用labelme標注defect位置, 標註檔為json format
    
    :unet training format: 由於unet訓練時需要一對一對的影像檔與defect mask檔案, 所以需要再透過labelme_json_to_dataset這隻工具json檔都轉換成(image, mask) pair, 但由於這個工具只能一次轉換一個標註檔, 所以可以將這個檔案置換為修正後的程式(參考datastes/utils/json_to_dataset.py), 可以指定資料夾, 一次轉換所有照片 (command: labelme_json_to_dataset ../labeling)
    
    :number of images: 由於這個專案還在快速POC階段, 所以只有標注了85張照片, 快速驗證成效
    
    :image size:
        - raw image: (2456,2058,3)
        - training image: (224,224,3)
        - raw mask: ((2456,2058,3)
        - training mask: (224,224,1)


**讀取數據Sample Code**

::
    data = load_ADS_generator()
    tr_fen = data.dataGenerator
    funs = Bunch(fillmeangray=fillmeangray, imgaugmentation=imgaugmentation, extractDefect=extractDefect)
    batch = next(tr_fen(4, 'tr', funs=funs))

**Model Characteristics**

    :UNet++: A Nested U-Net Architecture for Medical Image Segmentation
    
    :backbone: densenet169
    
    :encoder_weights: imagenet
    
    :optimizer: adam
    
    :loss: binary_crossentropy
    
    :metrics: mean_iou


專案說明：
透過UNET++ image segmentation model檢測絕緣片上的細微刮痕, 在這個專案因為標註的圖片較少, 所以使用了image augmentation的方法來增加訓練圖片的數量, 但因為套用了image augmentation之後照片數量又太多了, 一次load到RAM裡面很佔用記憶體空間, 所以在實作上嘗試了generator的方法, 每次只讀一個batch的照片進來, 可以大幅降低RAM的需求, 但相對的訓練的速度就慢很多, 需要相對較長的訓練時間. 後續可以在generator這個function裡面優化程式執行時間.


..  image:: https://i.imgur.com/So9eHxq.jpg
    :height: 400
    :width: 400


References
- 標注工具 labelme (`link1`_)
- json_to_dataset (`link2_`)
- UNET++ (`link3`_)
- segmentation tools (`link4`_)

.. _link1: https://github.com/wkentaro/labelme
.. _link2: https://github.com/wkentaro/labelme/blob/master/labelme/cli/json_to_dataset.py
.. _link3: https://github.com/MrGiovanni/UNetPlusPlus
.. _link3: https://github.com/qubvel/segmentation_models
