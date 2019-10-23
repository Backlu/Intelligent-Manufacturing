#!/usr/bin/env python
# coding: utf-8

# Copyright © 2019 Hsu Shih-Chieh

import os, csv, sys, shutil, warnings, joblib, glob, ntpath, hashlib, chardet, cv2, random, datetime, json
from os import environ, listdir, makedirs, getcwd
from os.path import dirname, exists, expanduser, isdir, join, splitext
from collections import namedtuple
import numpy as np
import pandas as pd
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from .utils import Bunch
from .utils import checkencoding
from .utils import readTDMSasDF
from .utils import MCase
import logging
logging.basicConfig(level="ERROR")
np.set_printoptions(suppress=True)
from sklearn.utils import shuffle
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm
from PIL import Image, ImageEnhance
from skimage.filters import try_all_threshold, threshold_mean, threshold_local, threshold_minimum, threshold_otsu
import segmentation_models as sm
from collections import Counter, defaultdict
from functools import partial

def load_facialbeauty(img_width=350, img_height=350, qty=None):
    """
    Load and return facialbeauty dataset (regerssion
    .. versionadded:: 20191008

    Parameters:
        img_width - resize image width to img_width
        img_height - resize image height to img_height

    Returns:
        data : Bunch, dictionary like data 
        
    Example:
        data = load_facialbeauty()
    """
    module_path=dirname(__file__)
    ratings = joblib.load(join(module_path, 'images/FacialBeauty/All_Ratings.pkl'))
    labels_df = ratings.groupby('Filename')['Rating'].mean()    
    n_samples = len(os.listdir(join(module_path, 'images/FacialBeauty/Images')))
    if qty:
        if n_samples > qty:
            n_samples=qty
    
    data = np.empty((n_samples, img_width, img_height, 3), dtype=np.float32)
    target = np.empty((n_samples, 1), dtype=np.float32)
    data_names = []
    for idx, imgpath in enumerate(glob.glob(os.path.join(module_path,'images/FacialBeauty/Images/*'))):
        if idx >= n_samples:
            break        
        img = load_img(imgpath, target_size=(img_width, img_height))
        img_array = img_to_array(img)
        fname = ntpath.basename(imgpath)
        rating = labels_df.loc[fname]
        data[idx] = img
        target[idx] = rating
        data_names.append(fname)

    with open(join(module_path,'descr', 'facialbeauty.rst')) as rst_file:
        fdescr = rst_file.read()
        
    #n_testsamples = len(os.listdir(join(module_path, 'images/FacialBeauty/testimages')))        
    n_testsamples = len(glob.glob(os.path.join(module_path,'images/FacialBeauty/testimages/big*')))
    test_data = np.empty((n_testsamples, img_width, img_height, 3), dtype=np.float32)
    for idx, imgpath in enumerate(glob.glob(os.path.join(module_path,'images/FacialBeauty/testimages/big*'))):
        img = load_img(imgpath, target_size=(img_width, img_height))
        img_array = img_to_array(img)
        test_data[idx] = img
    
    data = Bunch(data=data, target=target, data_names=data_names, DESCR=fdescr, test_data=test_data)
    return data



def load_hotmelt():
    """
    專案：十字彈片
    Load and return hotmelt dataset (classification)
    此處抓的資料是專案中期手動收集與標注的影像, 後期自動收錄的影像尚未放進來
    .. versionadded:: 20191009

    Parameters:

    Returns:
        data : Bunch, dictionary like data 
        
    Example:
        data = load_hotmelt()
    """        
    module_path=dirname(__file__)
    #module_path=''
    path = join(module_path, 'images/HotMelt/phase1')

    datagen = ImageDataGenerator()
    gen = datagen.flow_from_directory(path, target_size=(2050//2,2432//2), batch_size=30, shuffle=False, color_mode='rgb') 
    gen.batch_size = gen.samples
    data,target=gen.next()
    target_names = gen.class_indices
    data_names = np.array(gen.filenames)

    with open(join(module_path,'descr', 'hotmelt.rst')) as rst_file:
        fdescr = rst_file.read()
    data = list(map(lambda x: x.astype('uint8'), data))    
    
    bunch = Bunch(data=data, target=target, data_names=data_names, DESCR=fdescr)
    return bunch    


def load_hotmelt_generator():
    """
    專案：十字彈片
    Load and return hotmelt dataset generator
    此處抓的資料是專案中期手動收集與標注的影像, 後期自動收錄的影像尚未放進來
    這個method練習實作data generator, 如果數據量非常大無法一次全部load到RAM裡面的時候, 可以改用generator的方式批次讀入. 
    
    .. versionadded:: 20191011

    Parameters:

    Returns:
        data : Bunch, dictionary like data 
        
    Example:
        data = load_hotmelt_generator()
    """       
    module_path=dirname(__file__)
    path = join(module_path, 'images/HotMelt/phase1')
    classnames = sorted(os.listdir(path))    
    def data_generator(batch_size, augfun, dtype='tr', oversampling=True):
        '''data generator for fit_generator'''

        imgpaths = []
        targets = []
        path = join(module_path, 'images/HotMelt/phase1')
        classnames = sorted(os.listdir(path))
        for i, f in enumerate(classnames):
            path = join(module_path, f'images/HotMelt/phase1/{f}/*')
            for p in glob.glob(path):
                imgpaths.append(p)
                targets.append(i)     
        
        imgpaths_tr, imgpaths_ts, targets_tr, targets_ts = train_test_split(imgpaths, targets, test_size=0.2, random_state=40)        
        imgpaths_tr, imgpaths_val, targets_tr, targets_val = train_test_split(imgpaths_tr, targets_tr, test_size=0.2, random_state=40)
        
        if dtype=='tr':
            if oversampling:
                ros = RandomOverSampler()
                imgpaths_tr, targets_tr = ros.fit_resample(np.reshape(imgpaths_tr, (-1, 1)), targets_tr)   
                imgpaths_tr = imgpaths_tr.flatten()
            imgpaths = imgpaths_tr
            targets = targets_tr
        elif dtype=='val':
            imgpaths = imgpaths_val
            targets = targets_val         
        else:
            imgpaths = imgpaths_ts
            targets = targets_ts      
        
        targets = pd.get_dummies(targets).values                
        n, i = len(imgpaths), 0
        while True:
            batch_img = []
            batch_y = []
            for b in range(batch_size):
                if i==0:
                    imgpaths, targets = shuffle(imgpaths, targets)
                img  = load_img(imgpaths[i])
                img_array = img_to_array(img)
                y = targets[i]
                batch_img.append(img_array)
                batch_y.append(y)
                i = (i+1) % n
            batch_img=np.array(batch_img)
            batch_y=np.array(batch_y)
            X, Y = augfun(batch_img, batch_y, batch_size)
            yield X,Y
            
    with open(join(module_path,'descr', 'hotmelt.rst')) as rst_file:
        fdescr = rst_file.read()            
            
    bunch = Bunch(dataGenerator=data_generator, data_names=classnames, DESCR=fdescr)
    return bunch


def load_hotmeltyolodata():
    """
    專案：十字彈片
    Load and return hotmelt ROI dataset description file (train.txt)
    這個function只讀取train.txt, 並回傳內容, 實際抓取數據在img_hotmelt_ROI.ipynb裡面的generator實作
    .. versionadded:: 20191009

    Parameters:

    Returns:
        data : Bunch, dictionary like data 
        
    Example:
        data = load_hotmeltyolodata()
    """            
    module_path=dirname(__file__)
    path = join(module_path, 'images/HotMelt/objectdetection/train.txt')    
    with open(path) as f:
        lines = f.readlines()    
        
    with open(join(module_path,'descr', 'hotmelt_roi.rst')) as rst_file:
        fdescr = rst_file.read() 
        
    bunch = Bunch(data=lines, DESCR=fdescr)    
    return bunch 



def __yolo3_annotaion_generate__():
    """
    專案：十字彈片
    這個function只用來將labelImg標注好的檔案(xxx.xml)轉換成yolov3要用的格式(train.txt)
    目前是offline使用
    Example:
        data = __yolo3_annotaion_generate__()
    """        
    def convert_annotation(image_id, list_file):
        in_file = open('images/HotMelt/objectdetection/label/%s.xml'%(image_id))
        tree=ET.parse(in_file)
        root = tree.getroot()
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
            list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

    classes = ['roi']
    wd = getcwd()    
    image_ids = os.listdir('images/HotMelt/objectdetection/img/')
    list_file = open('images/HotMelt/objectdetection/train.txt', 'w')
    cnt=0
    for image_id in image_ids:
        image_id = image_id[:-4]
        xmlfile = 'images/HotMelt/objectdetection/label/%s.xml'%(image_id)
        if not os.path.exists(xmlfile):
            print('skip',image_id)
            continue        
        cnt=cnt+1
        list_file.write('%s/images/HotMelt/objectdetection/img/%s.jpg'%(wd, image_id))    
        convert_annotation(image_id, list_file)
        list_file.write('\n')
    list_file.close()    


def load_germination():
    """
    專案：植物工廠發芽率檢測
    Load and return germination dataset
    分為兩種影像, 一種是後期用8k攝影機拍攝的, 另一種是專案初期用iphone手機拍的
    .. versionadded:: 20191010

    Parameters:

    Returns:
        data : Bunch, dictionary like data
        回傳兩種影像，8k與iphone
        
    Example:
        data = load_germination()
    """    
    
    module_path=dirname(__file__)
    #module_path=''
    path = join(module_path, 'images/Germination/image/8k/*')
    files = glob.glob(path)
    data_8k=[]
    data_8knames=[]
    for p in files:
        img  = load_img(p)
        img_array = img_to_array(img)
        fname = ntpath.basename(p)
        data_8k.append(img_array)
        data_8knames.append(fname)
    data_8k = list(map(lambda x: x.astype('uint8'), data_8k))        
        
    path = join(module_path, 'images/Germination/image/iphone/*')
    files = glob.glob(path)
    data_iphone=[]
    data_iphonenames=[]
    for p in files:
        img  = load_img(p)
        img_array = img_to_array(img)
        fname = ntpath.basename(p)
        data_iphone.append(img_array)
        data_iphonenames.append(fname)        

    data_iphone = list(map(lambda x: x.astype('uint8'), data_iphone))        
    
    with open(join(module_path,'descr', 'germination.rst')) as rst_file:
        fdescr = rst_file.read()
    
    
    bunch = Bunch(data_iphone=data_iphone, data_iphonenames=np.array(data_iphonenames), data_8k=data_8k, data_8knames=np.array(data_8knames), DESCR=fdescr)
    return bunch    


def load_rca():
    """
    專案: 良率異常集中性分析
    Load and return Level 10生產數據 dataset
    
    .. versionadded:: 20191011

    Parameters:

    Returns:
        data : Bunch, dictionary like data
        回傳三種數據: SFC, Parts, Test
        
    Example:
        data = load_rca()
    """    
    module_path=dirname(__file__)
    path = join(module_path, 'data/rca/data/IPPD-L10_PARTS_SFC.txt')    
    parts_df = pd.read_csv(path, encoding=checkencoding(path))
    path = join(module_path, 'data/rca/data/IPPD-L10_SFC.txt')    
    sfc_df = pd.read_csv(path, encoding=checkencoding(path))
    path = join(module_path, 'data/rca/data/IPPD-L10_TEST.txt')    
    test_df = pd.read_csv(path, encoding=checkencoding(path))
    
    test_df.columns = ['SN','Station','Stationcode','Machine','start_time','end_time','isTestFail','symptom','desc','uploadtime','emp','ver1','ver2']
    sfc_df.columns = ['ID','SN','WO','HH_Part','CUST_Part','assembline','scantime','na1','na2','product','floor']
    parts_df.columns = ['ID','PARTSN','scantime','opid','assembly_station','part','HH_Part','CUST_Part','line','na1','na2'] 
        
        
    with open(join(module_path,'descr', 'rca.rst')) as rst_file:
        fdescr = rst_file.read()
        
    bunch = Bunch(sfc=sfc_df, parts=parts_df, test=test_df, sfc_names=sfc_df.columns, parts_names=parts_df.columns, test_names=test_df.columns, DESCR=fdescr)        
    return bunch
    

def load_newscommentary_v14(train_perc=20, val_prec=1, MAX_LENGTH = 40, BATCH_SIZE = 128):
    '''
    下載訓練數據, 格式為中英文配對的tuple list, 對訓練數據建立中英文字典
    前處理數據
        - 在example的前後加入BOS, EOS索引值
        - 過濾掉長度超過40的example
        - padding: padded_batch 函式能幫我們將每個batch裡頭的序列都補0到跟當下 batch 裡頭最長的序列一樣長。
        - shuffle: 將examples洗牌確保隨機性
    .. versionadded:: 20191012

    Parameters:
        MAX_LENGTH: 每個example保留的長度
        BATCH_SIZE: training的批次大小

    Returns:
        data : Bunch, dictionary like data
        - 訓練數據, 驗證數據, 英文字典, 中文字底, 說明文件
        
    Example:
        data = load_newscommentary_v14()        
    '''
    module_path=dirname(__file__)
    output_dir = join(module_path,'text/wmt2019/newscommentary_v14')
    download_dir = output_dir
    en_vocab_file = os.path.join(output_dir, "en_vocab")
    zh_vocab_file = os.path.join(output_dir, "zh_vocab")
    config = tfds.translate.wmt.WmtConfig(
        version=tfds.core.Version('0.0.3', experiments={tfds.core.Experiment.S3: False}),
        language_pair=("zh", "en"),
        subsets={tfds.Split.TRAIN: ["newscommentary_v14"] }
    )
    builder = tfds.builder("wmt_translate", config=config)
    builder.download_and_prepare(download_dir=download_dir)
    
    ##FIXME
    #train_perc, val_prec= 20, 1
    drop_prec = 100 - train_perc - val_prec
    split = tfds.Split.TRAIN.subsplit([train_perc, val_prec, drop_prec])
    examples = builder.as_dataset(split=split, as_supervised=True)
    train_examples, val_examples, _ = examples
    try:
        subword_encoder_en = tfds.features.text.SubwordTextEncoder.load_from_file(en_vocab_file)
        print(f"載入已建立的字典： {en_vocab_file}")
    except:
        print("沒有已建立的字典，從頭建立。")
        subword_encoder_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for en, _ in train_examples), target_vocab_size=2**13) # 有需要可以調整字典大小

        subword_encoder_en.save_to_file(en_vocab_file)

    print(f"字典大小：{subword_encoder_en.vocab_size}")
    print(f"前 10 個 subwords：{subword_encoder_en.subwords[:10]}")

    try:
        subword_encoder_zh = tfds.features.text.SubwordTextEncoder.load_from_file(zh_vocab_file)
        print(f"載入已建立的字典： {zh_vocab_file}")
    except:
        print("沒有已建立的字典，從頭建立。")
        subword_encoder_zh = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (zh.numpy() for _, zh in train_examples), target_vocab_size=2**13, 
            max_subword_length=1) # 每一個中文字就是字典裡的一個單位
        subword_encoder_zh.save_to_file(zh_vocab_file)

    print(f"字典大小：{subword_encoder_zh.vocab_size}")
    print(f"前 10 個 subwords：{subword_encoder_zh.subwords[:10]}")
    
    def encode(en_t, zh_t):
        '''
        因為字典的索引從 0 開始，我們可以使用 subword_encoder_en.vocab_size 這個值作為 BOS 的索引值
        用 subword_encoder_en.vocab_size + 1 作為 EOS 的索引值
        '''
        en_indices = [subword_encoder_en.vocab_size] + subword_encoder_en.encode(en_t.numpy()) + [subword_encoder_en.vocab_size + 1]
        zh_indices = [subword_encoder_zh.vocab_size] + subword_encoder_zh.encode(zh_t.numpy()) + [subword_encoder_zh.vocab_size + 1]
        return en_indices, zh_indices

    def tf_encode(en_t, zh_t):
        '''
        在 `tf_encode` 函式裡頭的 `en_t` 與 `zh_t` 都不是 Eager Tensors, 要到 `tf.py_funtion` 裡頭才是
        另外因為索引都是整數，所以使用 `tf.int64`
        '''
        return tf.py_function(encode, [en_t, zh_t], [tf.int64, tf.int64])

    def filter_max_length(en, zh, max_length=MAX_LENGTH):
        return tf.logical_and(tf.size(en) <= max_length, tf.size(zh) <= max_length)
    
    print('preprocess...')
    BUFFER_SIZE = 15000
    # 訓練集
    train_dataset = (train_examples  # 輸出：(英文句子, 中文句子)
                     .map(tf_encode) # 輸出：(英文索引序列, 中文索引序列)
                     .filter(filter_max_length) # 同上，且序列長度都不超過 40
                     .cache() # 加快讀取數據
                     .shuffle(BUFFER_SIZE) # 將例子洗牌確保隨機性
                     .padded_batch(BATCH_SIZE, padded_shapes=([-1], [-1])) # 將 batch 裡的序列都 pad 到一樣長度
                     .prefetch(tf.data.experimental.AUTOTUNE)) # 加速
    # 驗證集
    val_dataset = (val_examples
                   .map(tf_encode)
                   .filter(filter_max_length)
                   .padded_batch(BATCH_SIZE, padded_shapes=([-1], [-1])))    
    
    with open(join(module_path,'descr', 'wmt2019.rst')) as rst_file:
        fdescr = rst_file.read()
        
    bunch = Bunch(train=train_dataset, val=val_dataset, subword_encoder_zh=subword_encoder_zh, subword_encoder_en=subword_encoder_en, DESCR=fdescr)
    return bunch


def load_iaicnc():
    '''
    2018 Foxconn IAI CNC競賽的數據
    .. versionadded:: 20191014

    Parameters:

    Returns:
        data : Bunch, dictionary like data
        - 訓練數據, 競賽測試數據, 欄位名稱
        
    Example:
        data = load_iaicnc()        
    '''    
    def getsensordata(path):
        dflist=[]
        trainflist = os.listdir(path)
        trainflist.sort()
        print(path,len(trainflist))
        for i, f in tqdm(enumerate(trainflist)):
            if ('Train_B' in path):
                if i>=225 and i<=256:
                    continue
                elif i>=11 and i<=20:
                    continue
            tmp = pd.read_csv(os.path.join(path,f), header=None)
            tmp['id']=i
            dflist.append(tmp)
        data =pd.concat(dflist)
        data = data.reset_index(drop=True)
        data.columns = ['x','y','z','xg','yg','zg','v','id']
        return data       
    
    #module_path =''
    cachename ='bunch.pkl'
    module_path=dirname(__file__)
    with open(join(module_path,'descr', 'iai_cnc.rst')) as rst_file:
        fdescr = rst_file.read()
        
    if os.path.exists(join(module_path, f'data/iai_cnc/{cachename}')):
        print('讀取暫存數據',cachename)
        bunch = joblib.load(join(module_path, f'data/iai_cnc/{cachename}'))
        pass
    else:
        print('重新讀取數據')
        wearcols=['flute_1','flute_2','flute_3']
        path = join(module_path, 'data/iai_cnc/Train_A/Train_A_wear.csv')
        wear_A = pd.read_csv(path, usecols=wearcols)
        wear_A['flute_mean']=wear_A[wearcols].mean(axis=1) 
        wear_A['flute_max']=wear_A[wearcols].max(axis=1)
        wear_A['flute_min']=wear_A[wearcols].min(axis=1)

        path = join(module_path, 'data/iai_cnc/Train_B/Train_B_wear.csv')
        wear_B = pd.read_csv(path, usecols=wearcols)
        wear_B = wear_B.drop(list(range(225,257)))
        wear_B = wear_B.drop(list(range(11,21)))
        wear_B['flute_mean']=wear_B[wearcols].mean(axis=1) 
        wear_B['flute_max']=wear_B[wearcols].max(axis=1)
        wear_B['flute_min']=wear_B[wearcols].min(axis=1)

        print('wear_A',wear_A.shape)
        print('wear_B',wear_B.shape)

        trA = getsensordata(join(module_path, 'data/iai_cnc/Train_A/Train_A'))
        trB = getsensordata(join(module_path, 'data/iai_cnc/Train_B/Train_B'))
        ts = getsensordata(join(module_path, 'data/iai_cnc/Test/Test'))

        bunch = Bunch(trA=trA, trB=trB, ts=ts, wear_A=wear_A, wear_B=wear_B, data_names=trA.columns, wear_names=wear_B.columns, DESCR=fdescr)            
        joblib.dump(bunch, join(module_path, f'data/iai_cnc/{cachename}'))   
    return bunch


def load_ADS_generator():
    """
    專案：絕緣片瑕疵檢測
    
    .. versionadded:: 20191015

    Parameters:

    Returns:
        data : Bunch, dictionary like data 
            - data_generator, test_data, descr 
        
    Example:
        data = load_ADS_generator()
        tr_fen = data.dataGenerator
        funs = Bunch(fillmeangray=fillmeangray, imgaugmentation=imgaugmentation, extractDefect=extractDefect)
        batch = next(tr_fen(4, 'tr', funs=funs))
    """       
    module_path=dirname(__file__)
    def data_generator(batch_size, dtype='tr', funs=None, preprocess=True):
        '''data generator for fit_generator'''
        extractDefect=funs.extractDefect
        fillmeangray=funs.fillmeangray
        imgaugmentation=funs.imgaugmentation       
        IMG_HEIGHT, IMG_WIDTH = (224,224)
        imgpaths = []
        path = join(module_path, 'images/Insulatingpatch/training/*')
        imgpaths = glob.glob(path)
        imgpaths_tr, imgpaths_val = train_test_split(imgpaths, test_size=0.2, random_state=40)        
        imgpaths = imgpaths_tr if dtype=='tr' else imgpaths_val
        preprocess_input = sm.get_preprocessing('densenet169')
        n, i = len(imgpaths), 0
        while True:
            batch_img = []
            batch_mask = []
            for b in range(batch_size):
                if dtype =='tr':
                    imgpaths = shuffle(imgpaths) if i==0 else imgpaths #全部看完一次之後打亂順序
                img = cv2.imread(os.path.join(imgpaths[i], 'img.png'))
                mask = cv2.imread(os.path.join(imgpaths[i], 'label.png'))
                #剪出Defect圖像, Mask, 位置
                mask_gray = cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY)
                ret,mask_binary = cv2.threshold(mask_gray, 5, 255, cv2.THRESH_BINARY )
                mask = cv2.cvtColor(mask_binary,cv2.COLOR_GRAY2RGB)                                
                defectImg, defectMask, (bottom, top, left, right) = extractDefect(img, mask)
                defectHeight, defectWidth, _ = defectMask.shape
                #平移defect位址,  #FIXME: 可以再加上defect影像大小縮放
                augy = random.choice(range((img.shape[0]-defectHeight)))
                augx = random.choice(range((img.shape[1]-defectWidth)))
                img_aug=img
                mask_aug=mask
                if dtype =='tr':
                    img_aug[bottom:top, left:right] = img_aug[augy:augy+defectHeight, augx:augx+defectWidth]
                    img_aug[augy:augy+defectHeight, augx:augx+defectWidth] = defectImg
                    mask_aug=np.zeros_like(mask)
                    mask_aug[augy:augy+defectHeight, augx:augx+defectWidth] = defectMask
                img_aug = fillmeangray(img_aug) #空白處填充平均灰色
                img_aug = imgaugmentation(img_aug) if dtype =='tr' else Image.fromarray(img_aug) #影像輕微變化
                #Resize
                img_aug = img_aug.resize((IMG_HEIGHT, IMG_WIDTH), Image.NEAREST)
                img_aug = np.array(img_aug)
                img_aug = preprocess_input(img_aug) if preprocess else img_aug
                mask_aug = Image.fromarray(mask_aug).resize((IMG_HEIGHT, IMG_WIDTH), Image.NEAREST)
                mask_aug = np.array(mask_aug)
                mask_zero = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
                mask_aug = np.maximum(mask_zero, mask_aug)            
                batch_img.append(img_aug)
                mask_aug = mask_aug[:,:,:1]
                mask_aug = mask_aug//255
                batch_mask.append(mask_aug)    
                i = (i+1) % n
            batch_img=np.array(batch_img)
            batch_mask=np.array(batch_mask)
            yield batch_img,batch_mask

    path = join(module_path, 'images/Insulatingpatch/training/*')            
    paths = glob.glob(path)
    trimgQty = len(paths)
    
    path = join(module_path, 'images/Insulatingpatch/testing/*')
    paths = glob.glob(path)
    testImgs = []
    for n, path in tqdm(enumerate(paths), total=len(paths)):
        img = cv2.imread(path)
        #img = fillmeangray(img)    
        #img = Image.fromarray(img).resize((IMG_HEIGHT, IMG_WIDTH), Image.NEAREST)
        testImgs.append(img)

    with open(join(module_path,'descr', 'ads.rst')) as rst_file:
        fdescr = rst_file.read()            
            
    bunch = Bunch(dataGenerator=data_generator, testImgs=testImgs, trimgQty=trimgQty, DESCR=fdescr)
    return bunch


def load_cnc():
    """
    專案：刀具壽命預測, 刀具全生命週期數據
    
    .. versionadded:: 20191017

    Parameters:

    Returns:
        data : Bunch, dictionary like data 
            - plc data, vibration sensor data, plc column names, sensor column names descr 
        
    Example:
        data = load_cnc()
    """            
    module_path=dirname(__file__)
    if os.path.exists(join(module_path, 'data/cnc/GL-A2-09/plcblocks.pkl')):
        print('讀取暫存檔')
        plcblocks = joblib.load(join(module_path, 'data/cnc/GL-A2-09/plcblocks.pkl'))
        sensorblocks = joblib.load(join(module_path, 'data/cnc/GL-A2-09/sensorblocks.pkl'))
    else: 
        print('重新讀取數據')
        #1. Load Wear measurment
        np.set_printoptions(suppress=True)
        wearmeasure_path = join(module_path, 'data/cnc/wearmeasure_GL.pkl')
        wearmearue = joblib.load(wearmeasure_path)

        #2. Load PLC Data
        plcdatapath = join(module_path, 'data/cnc/GL-A2-09/PLC/*')
        dfs = []
        for p in glob.glob(plcdatapath):
            df = pd.read_csv(p, encoding='csbig5')
            dfs.append(df)
        plcdata = pd.concat(dfs, axis=0)

        plcdata = plcdata.apply(pd.to_numeric, errors='ignore')
        plcdata['毫秒']= plcdata['毫秒'].map(lambda x: str(int(x)))
        plcdata['datetime'] = plcdata[['日期','時間','毫秒']].apply(' '.join, axis=1)
        plcdata['datetime'] = plcdata['datetime'].map(lambda x: datetime.datetime.strptime(x, '%Y/%m/%d %H:%M:%S %f'))
        plcdata = plcdata.sort_values(by='datetime')
        plcdata.reset_index(drop=True, inplace=True)
        plcdata['單節'] = plcdata['單節'].fillna('')
        plcdic ={}
        plcdic['x'], plcdic['y'], plcdic['z'], plcdic['feed'], plcdic['time'], plcdic['feedtrue'], plcdic['feedrate'], plcdic['gcode'], plcdic['sload'],plcdic['speed'],plcdic['cutv'],plcdic['speedtrue'] = zip(*plcdata[['X軸機械座標', 'Y軸機械座標', 'Z軸機械座標', '設定進給', 'datetime', '實際進給','進給率','單節','主軸負載','設定轉速','切削量','實際轉速']].values)
        for k in plcdic.keys():
            plcdic[k] = np.array(plcdic[k])

        #3. Get Sensor Data  
        sensordatapath = join(module_path, 'data/cnc/GL-A2-09/Sensor/*')
        tdmslist = []
        for subf in glob.glob(sensordatapath):
            if not os.path.isdir(subf): 
                continue
            for tdms in glob.glob(os.path.join(subf,'*')):
                if not tdms.endswith('tdms'):
                    continue
                timestr = tdms[-17:-5]
                timeobj = datetime.datetime.strptime(timestr, '%y%m%d%H%M%S')
                tdmslist.append((tdms, timeobj))

        tdmslist = sorted(tdmslist, key=lambda s: s[1])
        tdmsFlist, tdmsTlist=zip(*tdmslist)
        tdmsFlist, tdmsTlist=np.array(tdmsFlist), np.array(tdmsTlist)

        #4. Get PLC/Sensor Blocks, 將每5秒數據切割成一個數據塊放入plcblocks
        cols=['Spindle_S01']#['Spindle_S01','Spindle_S02','Spindle_S03']
        startpoint=0
        framesize=5 #每5秒一個數據塊
        plcblocks=[]
        sensorblocks = defaultdict(list)
        print(plcdic['time'].shape[0])
        with tqdm(total=plcdic['time'].shape[0]) as pbar:
            while True:
                #get PLC block
                if startpoint >= plcdic['time'].shape[0]:
                    break
                starttime = plcdic['time'][startpoint]
                endtime = starttime + datetime.timedelta(seconds=framesize) 
                blockidx = np.where(plcdic['time'][startpoint:] < endtime)[0] + startpoint
                plcblock ={}
                if len(blockidx)==0:
                    break
                for k in plcdic.keys():
                    plcblock[k] = plcdic[k][blockidx]
                startpoint = startpoint+len(blockidx)
                pbar.update(len(blockidx))
                #get corresponding sensor block
                try:
                    stridx, endidx = np.where( (tdmsTlist <= starttime))[0][-1], np.where( (tdmsTlist <= endtime))[0][-1]
                    tdmsblocklist = tdmslist[stridx:endidx+1]
                    datalist=[]
                    for f, t in tdmsblocklist:
                        data = readTDMSasDF(path = f, cols=cols)
                        delta1, delta2=(starttime-t).total_seconds(), (endtime-t).total_seconds()
                        data = data[(data.index>delta1) & (data.index<delta2)]
                        datalist.append(data)
                    tdms_df = pd.concat(datalist, ignore_index=False) if len(datalist)>0 else pd.DataFrame()      

                except IndexError as error:
                    tdms_df=pd.DataFrame()
                except Exception as exception:
                    tdms_df=pd.DataFrame()

                if len(tdms_df)==0:
                    continue

                for c in cols:
                    sensorblocks[c].append(tdms_df[c].values)
                plcblocks.append(plcblock)

        joblib.dump(plcblocks, join(module_path, 'data/cnc/GL-A2-09/plcblocks.pkl'))
        joblib.dump(sensorblocks, join(module_path, 'data/cnc/GL-A2-09/sensorblocks.pkl'))

    with open(join(module_path,'descr', 'cnc.rst')) as rst_file:
        fdescr = rst_file.read()                    
    bunch = Bunch(plc=plcblocks, sensor=sensorblocks, plcname = list(plcblocks[0].keys()), sensornames=list(sensorblocks.keys()), DESCR=fdescr)
    return bunch


def load_fcft():
    """
    專案：主軸異常偵測, 主軸熱機數據 (20180508)
    
    .. versionadded:: 20191018

    Parameters:

    Returns:
        data : Bunch, dictionary like data 
            - vibration sensor data in each rpm speed, sensor column names, descr 
        
    Example:
        data = load_fcft()
    """        
    module_path=dirname(__file__)
    if os.path.exists(join(module_path, 'data/fcft/bunch.pkl')):
        print('讀取暫存檔')
        bunch = joblib.load(join(module_path, 'data/fcft/bunch.pkl'))
    else:     
        #1. Load PLC Data
        plcdatapath = join(module_path, 'data/fcft/*') 
        dfs = []
        for subf in glob.glob(plcdatapath):
            for p in glob.glob(join(subf, 'PLC/*')):
                df = pd.read_csv(p, encoding='csbig5')
                dfs.append(df) 

        plcdata = pd.concat(dfs, axis=0)
        plcdata = plcdata.apply(pd.to_numeric, errors='ignore')
        plcdata['毫秒']= plcdata['毫秒'].map(lambda x: str(int(x)))
        plcdata['datetime'] = plcdata[['日期','時間','毫秒']].apply(' '.join, axis=1)
        plcdata['datetime'] = plcdata['datetime'].map(lambda x: datetime.datetime.strptime(x, '%Y/%m/%d %H:%M:%S %f'))
        plcdata = plcdata.sort_values(by='datetime')
        plcdata.reset_index(drop=True, inplace=True)
        plcdata['單節'] = plcdata['單節'].fillna('')

        #2. Get Sensor Data  
        print('  > 讀取sensor數據')
        sensordatapath = plcdatapath = join(module_path, 'data/fcft/*') #join(module_path, 'data/fcft/GL_A2-FCFT0430/Sensor/*')
        tdmslist = []
        df_list = []
        cols=['Spindle_S01','Spindle_S02','Spindle_S03']#['Spindle_S01','Spindle_S02','Spindle_S03','Current_IA','Current_IB','Current_IC']
        for subf in glob.glob(sensordatapath):
            for tdms in glob.glob(join(subf, 'Sensor/*')):
                if not tdms.endswith('tdms'):
                    continue
                timestr = tdms[-17:-5]
                timeobj = datetime.datetime.strptime(timestr, '%y%m%d%H%M%S')
                tdmslist.append((tdms, timeobj))
        tdmslist = sorted(tdmslist, key=lambda s: s[1])
        for f,t in tqdm(tdmslist):
            try:
                df_ = readTDMSasDF(f, cols)
                dtobj = t
                df_['time'] = list(map(lambda x: x[0] + datetime.timedelta(0,x[1]), list(zip([dtobj]*len(df_.index), df_.index))))
                df_list.append(df_)
                print(df['time'].head())
            except:
                #print(f'warning! read {f} fail' )
                pass
        sensordata = pd.concat(df_list, ignore_index=True)

        with open(join(module_path,'descr', 'fcft.rst')) as rst_file:
            fdescr = rst_file.read()                    

        bunch=Bunch( datanames=cols, DESCR=fdescr)
        for speed in plcdata['設定轉速'].unique():
            if speed==0:
                continue
            endtime = max(plcdata[plcdata['設定轉速']==speed]['datetime'])
            starttime = min(plcdata[plcdata['實際轉速']>=speed]['datetime'])
            tdata = sensordata[(sensordata['time']>=starttime) & (sensordata['time']<=endtime)]
            bunch[f'speed_{speed}rpm']=tdata
            
        #joblib.dump(bunch, join(module_path, 'data/fcft/bunch.pkl'))
    return bunch


def load_moldcase(caseid = ['case01','case02']):
    """
    專案：成型機PHM案例數據
    
    .. versionadded:: 20191018

    Parameters:
        - caseid: case id list, ['case01','case02', ...]
        目前有case01 ~ case30數據, 其中case13, case14的健康值數據遺失

    Returns:
        data : Bunch, dictionary like data 
            - spccol_mapping: PLC數據欄位的中英文對照表
            - caseinfo: 每一個case的詳細資訊
            - caseXX_plc: 案例的PLC數據
            - caseXX_HV: 案例的健康值數據
    Example:
        data = load_moldcase()
    """           
    module_path=dirname(__file__)
    spccol_mapping = json.load(open(join(module_path, 'data/mold/spccol_mapping.json'), 'r'))
    caseinfo = json.load(open(join(module_path, 'data/mold/case.json'), 'r'))
    
    with open(join(module_path,'descr', 'mold.rst')) as rst_file:
        fdescr = rst_file.read()       
    bunch=Bunch( spccol_mapping=spccol_mapping, caseinfo=caseinfo, DESCR=fdescr)
    cpath = join(module_path,'data/mold/case.json')
    for cid in caseid:
        df_plc = pd.read_csv(join(module_path, f'data/mold/casedata/{cid}_PLC.csv'))
        df_hv = pd.read_csv(join(module_path, f'data/mold/casedata/{cid}_HV.csv'))
        case = MCase(cid, cpath)
        bunch[f'{cid}_plc']=df_plc
        bunch[f'{cid}_hv']=df_hv
        bunch[f'{cid}_caseinfo'] = case
    return bunch


def load_oee():
    """
    專案：成型機PHM案例數據
    
    .. versionadded:: 20191023

    Parameters:
       
    Returns:
        data : Bunch, dictionary like data 
            - spcdata: 控制器數據
            - lightstatus: 燈號狀態
    Example:
        data = load_oee()
    """           
    module_path=dirname(__file__)
    spcdata = pd.read_csv(join(module_path, 'data/mold/spcdata.csv'))
    spcdata['dt']= spcdata['timestamp'].map(lambda x: datetime.datetime.fromtimestamp(x))
    ligutstatusdata = pd.read_csv(join(module_path, 'data/mold/lightstatusdata.csv'))
    ligutstatusdata['dt']= ligutstatusdata['timestamp'].map(lambda x: datetime.datetime.fromtimestamp(x))
    spccol_mapping = json.load(open(join(module_path, 'data/mold/spccol_mapping.json'), 'r'))
    with open(join(module_path,'descr', 'oee.rst')) as rst_file:
        fdescr = rst_file.read()       
        
    bunch=Bunch( spcdata=spcdata, ligutstatusdata=ligutstatusdata, spccol_mapping=spccol_mapping, DESCR=fdescr)
    return bunch



def load_motoranchordata(get_random_data, preprocess_true_boxes):
    """
    專案：馬達定位點
    Load and return hotmelt ROI dataset description file (train.txt)
    這個function只讀取train.txt, 並回傳內容, 實際抓取數據在img_hotmelt_ROI.ipynb裡面的generator實作
    .. versionadded:: 20191009

    Parameters:

    Returns:
        data : Bunch, dictionary like data 
            - training image generator
            - validataion image generator
        
    Example:
        data = load_hotmeltyolodata()
    """        
    def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
        '''data generator for fit_generator'''
        n = len(annotation_lines)
        i = 0
        while True:
            image_data = []
            box_data = []
            for b in range(batch_size):
                if i==0:
                    np.random.shuffle(annotation_lines)
                image, box = get_random_data(annotation_lines[i], input_shape, random=True)
                image_data.append(image)
                box_data.append(box)
                i = (i+1) % n
            image_data = np.array(image_data)
            box_data = np.array(box_data)
            y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
            yield [image_data, *y_true], np.zeros(batch_size) 
            
    module_path=dirname(__file__)
    path = join(module_path, 'images/motorAnchorpoint/train.txt')    
    
    with open(join(module_path,'descr', 'motorAnchor.rst')) as rst_file:
        fdescr = rst_file.read()   
        
    with open(path) as f:
        lines = f.readlines()    
    
    anchors_str= '10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326'
    anchors = np.array([float(x) for x in anchors_str.split(',')]).reshape(-1, 2)
    data_generator_simple = partial(data_generator,  batch_size=10, input_shape=(416,416), anchors=anchors, num_classes=1)    
    
    val_split = 0.1
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val    
    
    gen_tr = data_generator_simple(lines[:num_train])    
    gen_val = data_generator_simple(lines[num_train:])    
    
    #bunch = Bunch(generator_tr=gen_tr, generator_val=gen_val, DESCR=fdescr)  
    bunch = Bunch(generator_tr=gen_tr, generator_val=gen_val, num_train=num_train, num_val=num_val, batch_size=10, num_classes=1, DESCR=fdescr)
    return bunch 

