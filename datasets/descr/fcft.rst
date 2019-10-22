CNC FCFT dataset
-------------------

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
        - Current_IA: 三相電流
        - Current_IB: 三相電流
        - Current_IC: 三相電流
        
    :數據收集日期:
    
        - 20180430
        - 20180502
        - 20180503
        - 20180508        
    
    
**讀取數據Sample Code**

::
    data = load_fcft()    
    
    
**Preprocess & Featureing**
    
    :數據特徵: 轉速的一階次頻率
    
**Model**
    
    :PCA T2/SPE: 用正常主軸的熱機數據當作基準, 監測後續每天的熱機數據是否發生異常. 
    
專案說明：
CNC的主軸為高單價的設備組件, 所以在每天開工以前都會透過一個熱機過程來檢測主軸是否有異常, 熱機過程如下：主軸不裝刀具, 設定不同轉速, 每個轉速都運轉一段時間並收集震動傳感器數據. 在這個專案我們收集主軸熱機過程數據, 並觀察每一個轉速的一階次頻率是否有異常發生. 透過引入異常數據驗證, 當異常發生得時候, 會在特定轉速的一階次頻率發現異常. 



