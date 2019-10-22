智能製造優秀案例實作練習平台
---------------------------

**Case_Practice:**

    :Case_Practice/datasets: datasets資料夾內收錄我在HH四年半經歷過的專案數據
        - Case_Practice/datasets/images: 影像類的數據都放在這個資料夾內, 包含了十字彈片檢測, 絕緣片檢測, FaceID檢測, TV偏光片檢測, COF導電粒子檢測, 植物工廠發芽率檢測
        - Case_Practice/datasets/data: 數字類的數據都放在這個資料夾內, 包含了成型機健康值, 測試良率異常集中性分析, CNC刀具壽命預測
        - Case_Practice/datasets/text: 文字類的數據都放在這個資料夾內, 包含了客訴資料分析, 中英文翻譯Transformer
        - Case_Practice/datasets/descr: 每個專案數據與模型實作的說明文件, 基本上每一個專案都會有一份說明檔, 用reStructuredText的格式編寫
        - Case_Practice/datasets/base.py: 實作以上各專案數據的數據讀取API, 處理了各種檔案讀取的細節, 使用者透過這個API可以拿到較友善的數據, 直接進行下一步的模型建立與訓練
    
    :Case_Practice/datasets/utils: base.py需要用到的一些模組化功能與客製化的模型, 會放在utils裡面, 並透過__init__.py一起匯入
        - yolov3
        - transformer
    
    :model: 儲存每個專案訓練好的keras model
    
    :tfmodel: 儲存每個專案訓練好的tensorlite model, tfmode的推論速度比keras model快非常多
    
    :SampleCode: 各專案的實作Sample Code (jupyter notebook ipynb)
    
    
**Datasets:**

    :load_facialbeauty: 顏值訓練數據
    
    :load_hotmelt: 十字彈片(全部照片一次讀出放在array內)影像數據
    
    :load_hotmelt_generator: 十字彈片(generator)影像數據
    
    :load_hotmeltyolodata: 十字彈片的ROI切割
    
    :load_germination: 植物工廠的發芽率辨識數據
    
    :load_rca: 良率異常集中性分析數據 
    
    :load_transformer: 中英文翻譯的數據newscommentary_v14
    
    :load_iaicnc: Foxconn IAI CNC競賽的數據
    
    :load_ADS_generator: 絕緣片刮痕瑕疵的數據

   
**Sample Code:**

    :img_facialbeauty.ipynb: 顏值判斷的模型訓練與推論
    
    :img_hotmelt_ROI.ipynb: 十字彈片瑕疵檢測專案的ROI擷取模型訓練與推論 (yolov3)
    
    :img_hotmelt.ipynb: 十字彈片瑕疵檢測專案的瑕疵分類模型訓練與推論 (densenet169)
    
    :img_germination.ipynb: 植物工廠的發芽率計算
    
    :img_rca.ipynb: 生產測試不良品的集中性計算
    
    :text_transformer.ipynb: 中英翻譯的Transformer實作(Self-Attention)
    
    :data_iaicnc.ipynb: 2018 Foxconn IAI CNC競賽的實作(xgboost)
    
    :img_ads.ipync: 絕緣片瑕疵檢測(UNET)
    
    

