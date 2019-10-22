
import sys, os, pywt, scipy
import numpy as np
from scipy import signal, stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import mixture

def isworkingChk(plcblock, g_z, g_zrise):
    '''
    用plc data來判斷每一個block的數據是否是在加工狀態, 判斷的規則大多來自數據觀察
        - feedtrue不為0
        - feedrate = 100
        - feed = 10000
        - z軸持續下降
    
    g_z: z軸目前的最小值(z軸持續下降, 記錄最小值)
    g_zrise: z軸是否往上(刀具是否有抬起, 如果有的話把z軸最小值歸0)
    
    parameters: 
        - plcblcok: plc數據
    '''
    ret_dict={}
    try:
        ret_dict['fTvalid'] = (plcblock['feedtrue'].astype(float)!=0).all() 
        ret_dict['fRvalid'] = (plcblock['feedrate'].astype(float)==100).all() 
        ret_dict['fvalid'] = (plcblock['feed'].astype(float)==10000).all() 
        ret_dict['zTvalid'] = (plcblock['z'].astype(float)<=g_z).all()
        ret = np.array(list(ret_dict.values())).all()
        ret_z, g_z = (False, g_z) if (plcblock['z']>=g_z).all() else (True, min(plcblock['z']))

    except:
        ret = False

    if not ret_z:
        zrise = max(plcblock['z']) - g_z
        rising, g_zrise = (True, zrise) if zrise>=g_zrise else (False, g_zrise)
        if not rising:
            g_z=0 if g_zrise>50 else g_z 
            g_zrise=0
    return ret, g_z, g_zrise

def getrms(data):
    return np.sqrt(np.square(data).mean())

def wavelet(data, w='haar'):
    mode = pywt.Modes.smooth
    w = pywt.Wavelet(w)#选取小波函数
    a = data
    ca = []#近似分量
    cd = []#细节分量

    for i in range(4):
        (a, d) = pywt.dwt(a, w, mode)#进行5阶离散小波变换
        ca.append(a)
        cd.append(d)
    rec_a = []
    rec_d = []
    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(pywt.waverec(coeff_list, w))#重构

    for i, coeff in enumerate(cd):
        coeff_list = [None, coeff] + [None] * i
        rec_d.append(pywt.waverec(coeff_list, w))    
    return rec_a, rec_d


def getorderrms(data, num):
    speedhz=2000#/60
    fs=25600
    nperseg=2560
    f, Pxx_den = signal.welch(data, fs, window='hanning', nperseg=nperseg, scaling='density')
    rmslist=[]
    for order in range(1,num):
        f_H_idx = np.where(f>=(speedhz*order))[0][0]
        f_L_idx = np.where(f<=(speedhz*order))[0][-1]
        rms = getrms(Pxx_den[f_L_idx-3:f_H_idx+3] )
        #rms = getrms([Pxx_den[f_H_idx],Pxx_den[f_L_idx] ])
        rmslist.append(rms)
    return rmslist

def getbestGMM(data):
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 5)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_comp in n_components_range:
            gmm = mixture.GaussianMixture(n_components=n_comp,covariance_type=cv_type, max_iter=200, tol=1e-5, n_init=3)
            gmm.fit(data)
            bic.append(gmm.bic(data))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
    print('best gmm model:\n',best_gmm)   
    return best_gmm


#-------FCFT--------

def getorderrms_fcft(data, order):
    speedhz=2000#/60
    fs=25600
    nperseg=2560
    f, Pxx_den = signal.welch(data, fs, window='hanning', nperseg=nperseg, scaling='density')
    rmslist=[]
    f_H_idx = np.where(f>=(speedhz*order))[0][0]
    f_L_idx = np.where(f<=(speedhz*order))[0][-1]
    rms = getrms(Pxx_den[f_L_idx-3:f_H_idx+3] )
    return rms

