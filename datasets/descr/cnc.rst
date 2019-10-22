CNC dataset
-----------------

**Data Set Characteristics::**

    :控制器數據欄位:
    
        - x: X軸機械座標
        - y: y軸機械座標
        - z: z軸機械座標        
        - feed: 設定進給
        - time: Datetime
        - feedtrue: 實際進幾
        - feedrate: 進幾率
        - gcode: 單節
        - sload: 主軸負載
        - speed: 設定轉速
        - cutv: 切削量
        - seedtrue: 實際轉速
        
    :震動傳感器欄位:
    
        - Spindle_S01: ?軸震動 (X or Y ro Z)
        - Spindle_S02: ?軸震動
        - Spindle_S03: ?軸震動
        

**讀取數據Sample Code**

::
    data = load_cnc()
    
**Preprocess & Featureing**

    :判斷刀具使否在加工狀態的規則:
    
        - feedtrue不為0
        - feedrate = 100
        - feed = 10000
        - z軸持續下降
    
    :傳感器震動數據特徵:
    
        - time domain vibration rms
        - frequency domain order rms (階次)
        - time-frequency domain wavelet component rms 

**Model Characteristics**

    :Model: GMM (Gaussian Mixture Model)
    
    :Model Selection: BIC (Bayesian information criterion)


專案說明：
為了研發CNC刀具磨耗預估與壽命預估, 在真實的模具製造工廠收集了15把刀具的全生命週期數據, 這份Sample Code Demo的是其中特性最好的第9把刀數據, 簡單利用RMS與頻域特徵, 就能觀察到刀具的健康值隨著使用時間合理的下降. 在模型建立方面, 採用anormaly detection的方式, 用刀具全生命週期的最一開始60分鐘當作基準值, 比較後續數據與基準值的差異計算健康值. 在數據前處理的部份, 直接擷取每5秒鐘為一個數據塊, 並利用從數據觀察的規則判斷每一個數據塊是否包含非加工數據, 只留下完全在加工的數據塊做模型的訓練與健康值的評估. 

