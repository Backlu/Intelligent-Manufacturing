COF dataset
---------------------------

**Data Set Characteristics::**

    :raw image: 原始影像檔
    
    :labeled image: 用高倍數顯微定標註導電粒子的影像檔
    
    

**讀取數據Sample Code**

::

    data=load_cofroi()


**Configutation**

    :img_id_lower: 在一個COF圖像資料夾內需要分析的圖像範圍, img_id_lower <= 圖檔名稱

    :img_id_upper: 在一個COF圖像資料夾內需要分析的圖像範圍, 圖檔名稱 <= img_id_upper

    :zone_visible_lehgth: Pin的ROI有效區域的長度(以pixel為單位)

    :pix_value_thld1: pixel灰度值的門檻值1

    :pix_value_thld2: pixel灰度值的門檻值2

    :pix_value_thld3: pixel灰度值的門檻值3

    :pin_height_thld_UB: pin高度的限制, pin_height_thld_LB <= Pin高度 

    :pin_height_thld_LB: pin高度的限制, <= Pin高度 <= pin_height_thld_UB 

    :lower_red: 人工標識出的粒子紅點之RGB範圍

    :upper_red: 人工標識出的粒子紅點之RGB範圍

    :candidate_particle_count: 候選的粒子位置(最亮的前面n顆粒子)

    :re_height: 候選粒子影像的大小(5x5)

    :re_width: 候選粒子影像的大小(5x5)


**ROI擷取步驟**

    :一. 擷取PIN:
    
        - Step 1: 確認每一根Pin的高度位置與它所在的圖檔路徑   
        
            - 作法與規則: 
            
                - 在水平方向取平均灰度值
                - pin的顏色較淺, 所以灰度值較高, 透過門檻值找到pin的y軸位置
                - 找出連續的y軸位置, 連續的一段y軸表示一根pin
                - 篩選出合格的pin (pin高度在預定義的規格內) 
                
        - Step 2: 確認每一根Pin的長度位置
        
            - 作法與規則: 
            
                - 只取第二根Pin的圖像為基準,來決定出所有Pin的長度的結束位置
                - 垂直方向取最小灰度值, 因為pin比較亮, 當灰度值低於門檻時, 表示為pin的結束位置
                - 根據預定一個pin長度回推pin的開始位置            
        
        - Step 3: 修正pin起始位置
        
            - 作法與規則: 找到PIN前方白白亮亮的地方當作修正後的起始位置(設定一個灰度值的門檻)
            
    :二. 擷取候選粒子(X)與標注粒子(Y):
        
        - Step 4 偵測在Pin上的顯微鏡已標註粒子位置
        
            - 作法與規則: 
            
                - 搜尋藍點
                - 找出藍點的輪廓
                - 找到輪廓的重心
                - 篩選面積>0的輪廓        
    
        - Step 5 算法偵測在Pin上的具有候選資格的粒子位置
        
            - 作法與規則: 

                - 前處理: 每根pin分六段調整灰度值, 讓每一段的平均灰度值都為128
                - 規則: 先取出X軸灰度值最大的前30個位置, 再分別針對這30個X軸位置, 取出Y軸灰度值最大的位置            

        - Step 6 比對候選粒子與標注粒子
            
            - 作法與規則:
            
                - 具有候選資格的粒子,將被歸類為兩類:(1)是粒子;(2)不是粒子  
                - 規則: 當候選粒子的鄰近4個pixel內有出現人工標識的粒子,則它為(1)是粒子, 反之則它為(2)不是粒子
                - 將候選粒子擴展為候選粒子小圖: 以候選粒子的(x位置,y位置)為中心,擴展2個pixel,構成候選粒子小圖(size=5x5)


    
**Model Characteristics**


**Training Hyperparameter**  



專案說明：  
COF導電粒子計算



