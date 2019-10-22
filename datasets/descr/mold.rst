成型機案例數據
---------------------------

**Data Set Characteristics::**

    :控制器數據: 從控制器取出的螺桿位置, 油壓缸壓力, 料管溫度的數據統計值, 總共有44種數據
    
    :設備健康值數據: 透過高維度SPC監控方法(PCA T2/SPE)計算後的設備健康值, 設備運作時實時運算, 並存入sqlite
    
    :spccol_mapping: PLC數據欄位的中英文對照表
    
    :caseinfo: 每一個案例的詳細資訊



**讀取數據Sample Code**

::

    data= load_moldcase()
    c = data.case01_caseinfo
    df_ctr = data.case01_plc
    df_hv = data.case01_hv


**Model Characteristics**
    
    :設備異常監控: PCA T2/SPE
    
        - Source Code: (還沒整理)
        - 算法筆記: https://hackmd.io/@JHSU/rywYQt7RV
    
    :設備異常分類(方法一): 案例的參數趨勢分析
    
        - Source Code: data_molding_TrendAnalysis.ipynb
        - 算法筆記: https://hackmd.io/@JHSU/By3uWuwPH
    
    :設備異常分類(方法二): 分析每一個兩個案例的之間的參數變化相似性
    
        - Source Code: data_molding_DTW.ipynb
        - 算法筆記: https://hackmd.io/@JHSU/HyCnabcPH
        
    :設備異常分類(方法三): 關聯規則
    
        - Source Code: data_molding_apriori.ipynb
        - 算法筆記: https://hackmd.io/@JHSU/BJCyWchPr
        
    :設備異常分類(方法四): SVM
    
        - Source Code: data_molding_SVM.ipynb
        - 算法筆記: https://hackmd.io/@JHSU/H1YiP5eur


專案說明：
這個專案與控制器廠商合作, 從控制器中取出螺桿位置, 油壓缸壓力, 料管溫度這三項特徵數據, 但因為控制器性能限制, 無法將實時數據取出, 退而求其次, 透過控制器本身的SPC監控功能, 取出這三項特徵的44種統計數據(ex: 最大值, 最小值,...), 並透過這些數據進行設備異常監控與設備異常分類


