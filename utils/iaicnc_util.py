#!/usr/bin/env python
# coding: utf-8

# Copyright © 2019 Hsu Shih-Chieh

import sys, os, datetime, warnings, scipy, collections, random, pywt, time, matplotlib
import pandas as pd 
import numpy as np
from scipy import signal, stats
from scipy.stats import skew, kurtosis, norm
from sklearn.externals import joblib 
from sklearn import decomposition, mixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook
from datetime import timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import skew, kurtosis, norm
from sklearn.base import BaseEstimator, RegressorMixin

def cnc_featuring(data):
    datacols = list(datasets.data_names)
    datacols.remove('id')    
    def getnewcolname(postfix):
        return list(map(lambda x: '_'.join([x,postfix]), datacols)) 
    def signaltonoise(a):
        a = np.asanyarray(a)
        m = a.mean(0)
        sd = a.std(axis=0, ddof=0)
        return np.where(sd == 0, 0, m/sd)
    def getmainfreqmag(d):
        ret =[]
        for c in d.columns:
            speedhz=10400/60 
            fs=50000 
            nperseg=50000 #window size
            f, Pxx_den = signal.welch(d[c], fs, nperseg=nperseg, scaling='spectrum',return_onesided=True)
            main1 = np.sqrt(np.mean(np.square(Pxx_den[173:175])))#Pxx_den[174]
            main2 = np.sqrt(np.mean(np.square(Pxx_den[346:348])))#Pxx_den[347]
            main3 = np.sqrt(np.mean(np.square(Pxx_den[520])))#Pxx_den[521]
            left1 = np.sqrt(np.mean(np.square(Pxx_den[171:173])))#Pxx_den[173]
            rifht1 = np.sqrt(np.mean(np.square(Pxx_den[175:177])))#Pxx_den[175]
            left2 = np.sqrt(np.mean(np.square(Pxx_den[343:346]))) #Pxx_den[346]
            rifht2 = np.sqrt(np.mean(np.square(Pxx_den[348:350]))) #Pxx_den[348]
            left3 = np.sqrt(np.mean(np.square(Pxx_den[518])))#Pxx_den[520]
            rifht3 = np.sqrt(np.mean(np.square(Pxx_den[522]))) #Pxx_den[522]

            frms = np.sqrt(np.square(Pxx_den[:1000]).mean())
            fmean = Pxx_den[:1000].mean()
            fstd = Pxx_den[:1000].std()
            fmax = Pxx_den[:1000].max()
            fmin = Pxx_den[:1000].min()
            fmaxmin = np.abs(fmax - fmin)

            frms2 = np.sqrt(np.square(Pxx_den[1000:]).mean())
            fmean2 = Pxx_den[1000:].mean()
            fstd2 = Pxx_den[1000:].std()
            fmax2 = Pxx_den[1000:].max()
            fmin2 = Pxx_den[1000:].min()
            fmaxmin2 = np.abs(fmax2 - fmin2)
            snr = signaltonoise(Pxx_den[:2000])
            ret.extend([main1, main2, main3, left1, left2, left3, rifht1, rifht2, rifht3, frms, fmean, fstd, fmax, fmin, fmaxmin, frms2, fmean2, fstd2, fmax2, fmin2, fmaxmin2, snr.flatten()[0]])
        return ret, len(ret)//len(d.columns)    
    
    data_X = []
    idlist = list(set(data['id']))
    tempprevmean=[-2,-2,-2,-2,-2,-2]
    for idx in tqdm_notebook(idlist):
        x = data[data['id']==idx][datacols]
        d = x['y']
        #確認工況
        checklen =30000
        d_std = d[:checklen].rolling(300).std()
        d_std = d_std.fillna(method='bfill')
        d_std = d_std.fillna(method='ffill')
        d_stb = np.where([abs(d_std - np.mean(d_std[5000:])) < 1.5 * np.std(d_std[5000:])])[1]
        cut1end = d_stb[0] 
        d_std = d[-checklen:].rolling(300).std()
        d_std = d_std.fillna(method='ffill')
        d_std = d_std.fillna(method='bfill')
        d_stb = np.where([abs(d_std - np.mean(d_std[:-5000])) < 1.5 * np.std(d_std[:-5000])])[1]
        cut2start = d[:-checklen].shape[0]+d_stb[-1] 
        cut1 = x[:cut1end]
        cut2 = x[cut2start:]
        stable = x[cut1end:cut2start]
        
        #確認工況是否正常
        cut1std = round(np.std(cut1['y']),2)
        cut2std = round(np.std(cut2['y']),2)
        stbstd = round(np.std(stable['y']),2)
        gapcheck1 = (cut1.mean()['x']-tempprevmean[0]) <2 and (cut1.mean()['y']-tempprevmean[1]) <2 and (cut1.mean()['z']-tempprevmean[2]) <2 
        gapcheck2 = (cut2.mean()['x']-tempprevmean[3]) <2 and (cut2.mean()['y']-tempprevmean[4]) <2 and (cut2.mean()['z']-tempprevmean[5]) <2 
        cut1valid = len(cut1)>300 and len(cut1)<5000 and (stbstd-cut1std)>2 and gapcheck1
        cut2valid = len(cut2)>300 and len(cut2)<5000 and (stbstd-cut2std)>2 and gapcheck2
        stbvalid = len(stable)>300
        tempprevmean = [cut1.mean()['x'], cut1.mean()['y'], cut1.mean()['z'], cut2.mean()['x'], cut2.mean()['y'], cut2.mean()['z']]
        
        stable = stable.fillna(method='bfill')
        stable = stable.reset_index(drop=True)
        cut1_re= cut1.copy()
        cut2_re= cut2.copy()
        stable_re= stable.copy()
        stable_ori=stable.copy()
        
        #每400筆取統計值
        rng = pd.date_range('1/1/2011', periods=cut1_re.shape[0], freq='min')
        cut1_re = cut1_re.set_index(rng)
        cut1_re = cut1_re.resample('400min')
        
        rng = pd.date_range('1/1/2011', periods=cut2_re.shape[0], freq='min')
        cut2_re = cut2_re.set_index(rng)
        cut2_re = cut2_re.resample('400min')        
        
        rng = pd.date_range('1/1/2011', periods=stable_re.shape[0], freq='min')
        stable_re = stable_re.set_index(rng)
        stable_re = stable_re.resample('5000min')        
                
            
        culnum=7
        cut1_mean = cut1_re.mean().quantile(0.5) if cut1valid else pd.Series([np.NAN]*culnum)
        cut1_mean.index=getnewcolname('cut1mean')
        cut1_std = cut1_re.std().median() if cut1valid else pd.Series([np.NAN]*culnum)
        cut1_std.index=getnewcolname('cut1std')
        cut1_max = cut1_re.max().quantile(0.99) if cut1valid else pd.Series([np.NAN]*culnum)
        cut1_max.index=getnewcolname('cut1max')
        cut1_min = cut1_re.min().quantile(0.01) if cut1valid else pd.Series([np.NAN]*culnum)
        cut1_min.index=getnewcolname('cut1min')
        cut1_rms = cut1_re.apply(lambda x: np.sqrt(np.square(x).mean())).median() if cut1valid else pd.Series([np.NAN]*culnum)
        cut1_rms.index=getnewcolname('cut1rms')
        cut1_maxmin = cut1_re.max().quantile(0.99) - cut1_re.min().quantile(0.01) if cut1valid else pd.Series([np.NAN]*culnum)
        cut1_maxmin.index=getnewcolname('cut1maxmin')
        
        try:
            rng = pd.date_range('1/1/2011', periods=cut1.shape[0], freq='min')
            cut1 = cut1.set_index(rng)
            cut1_re = cut1.resample('10min')
            cut1re_med = cut1_re.median()
            cut1_fdiff = cut1re_med - cut1re_med.shift()            
            cut1_fdiff.iloc[0] = cut1_fdiff.iloc[1] 
            #cut1_fdiff = np.abs(cut1_fdiff)
        except:
            #print('cut1,',idx)
            pass
        
        cut1_fdiff_mean = np.mean(np.abs(cut1_fdiff)) if cut1valid else pd.Series([np.NAN]*culnum)
        cut1_fdiff_mean.index=getnewcolname('cut1fdiffmean')
        cut1_fdiff_max = np.max(cut1_fdiff) if cut1valid else pd.Series([np.NAN]*culnum)
        cut1_fdiff_max.index=getnewcolname('cut1fdiffmax')            
        cut1_fdiff_min = np.min(cut1_fdiff) if cut1valid else pd.Series([np.NAN]*culnum)
        cut1_fdiff_min.index=getnewcolname('cut1fdiffmin')                
        cut1_fdiff_std = np.std(cut1_fdiff) if cut1valid else pd.Series([np.NAN]*culnum)
        cut1_fdiff_std.index=getnewcolname('cut1fdiffstd')
        cut1_fdiff_maxmin =  np.max(cut1_fdiff) - np.min(cut1_fdiff) if cut1valid else pd.Series([np.NAN]*culnum)
        cut1_fdiff_maxmin.index=getnewcolname('cut1fdiffmaxmin')   
        cut1_fdiff_skew = pd.Series(stats.skew(cut1_fdiff)) if cut1valid else pd.Series([np.NAN]*culnum)
        cut1_fdiff_skew.index=getnewcolname('cut1fdiffskew')        
        cut1_fdiff_kurtosis = pd.Series(stats.kurtosis(cut1_fdiff)) if cut1valid else pd.Series([np.NAN]*culnum)
        cut1_fdiff_kurtosis.index=getnewcolname('cut1fdiffkurtosis')        
        
        
        cut2_mean = cut2_re.mean().quantile(0.5) if cut2valid else pd.Series([np.NAN]*culnum)
        cut2_mean.index=getnewcolname('cut2mean')
        cut2_std = cut2_re.std().median() if cut2valid else pd.Series([np.NAN]*culnum)
        cut2_std.index=getnewcolname('cut2std')
        cut2_max = cut2_re.max().quantile(0.99) if cut2valid else pd.Series([np.NAN]*culnum)
        cut2_max.index=getnewcolname('cut2max')
        cut2_min = cut2_re.min().quantile(0.01) if cut2valid else pd.Series([np.NAN]*culnum)
        cut2_min.index=getnewcolname('cut2min')
        cut2_rms = cut2_re.apply(lambda x: np.sqrt(np.square(x).mean())).median() if cut2valid else pd.Series([np.NAN]*culnum)
        cut2_rms.index=getnewcolname('cut2rms')
        cut2_maxmin = cut2_re.max().quantile(0.99) - cut2_re.min().quantile(0.01) if cut2valid else pd.Series([np.NAN]*culnum)
        cut2_maxmin.index=getnewcolname('cut2maxmin')
        
        try:
            rng = pd.date_range('1/1/2011', periods=cut2.shape[0], freq='min')
            cut2 = cut2.set_index(rng)
            cut2_re = cut2.resample('10min')
            cut2re_med = cut2_re.median()
            cut2_fdiff = cut2re_med - cut2re_med.shift()            
            cut2_fdiff.iloc[0] = cut2_fdiff.iloc[1]
        except:
            #print('cut2,',idx)
            pass
        cut2_fdiff_mean = np.mean(np.abs(cut2_fdiff)) if cut2valid else pd.Series([np.NAN]*culnum)
        cut2_fdiff_mean.index=getnewcolname('cut2fdiffmean')
        cut2_fdiff_max = np.max(cut2_fdiff) if cut2valid else pd.Series([np.NAN]*culnum)
        cut2_fdiff_max.index=getnewcolname('cut2fdiffmax')            
        cut2_fdiff_min = np.min(cut2_fdiff) if cut2valid else pd.Series([np.NAN]*culnum)
        cut2_fdiff_min.index=getnewcolname('cut2fdiffmin') 
        cut2_fdiff_std = np.std(cut2_fdiff) if cut2valid else pd.Series([np.NAN]*culnum)
        cut2_fdiff_std.index=getnewcolname('cut2fdiffstd')
        cut2_fdiff_maxmin =  np.max(cut2_fdiff) - np.min(cut2_fdiff) if cut2valid else pd.Series([np.NAN]*culnum)
        cut2_fdiff_maxmin.index=getnewcolname('cut2fdiffmaxmin')
        cut2_fdiff_skew = pd.Series(stats.skew(cut2_fdiff)) if cut2valid else pd.Series([np.NAN]*culnum)
        cut2_fdiff_skew.index=getnewcolname('cut2fdiffskew')        
        cut2_fdiff_kurtosis = pd.Series(stats.kurtosis(cut2_fdiff)) if cut2valid else pd.Series([np.NAN]*culnum)
        cut2_fdiff_kurtosis.index=getnewcolname('cut2fdiffkurtosis') 
        
        
        stb_mean = stable_re.mean().quantile(0.5) if stbvalid else pd.Series([np.NAN]*culnum)
        stb_mean.index=getnewcolname('stablemean')
        stb_std = stable_re.std().median() if stbvalid else pd.Series([np.NAN]*culnum)
        stb_std.index=getnewcolname('stablestd')
        stb_max = stable_re.max().quantile(0.99) if stbvalid else pd.Series([np.NAN]*culnum)
        stb_max.index=getnewcolname('stablemax')
        stb_min = stable_re.min().quantile(0.01) if stbvalid else pd.Series([np.NAN]*culnum)
        stb_min.index=getnewcolname('stablemin')
        stb_rms = stable_re.apply(lambda x: np.sqrt(np.square(x).mean())).median() if stbvalid else pd.Series([np.NAN]*culnum)
        stb_rms.index=getnewcolname('stablerms')
        stb_maxmin = stable_re.max().quantile(0.99) - stable_re.min().quantile(0.01) if stbvalid else pd.Series([np.NAN]*culnum)
        stb_maxmin.index=getnewcolname('stbmaxmin')
        stb_skew = stable.skew() if stbvalid else pd.Series([np.NAN]*culnum)
        stb_skew.index=getnewcolname('stableskew')     
        stb_kurtosis = stable.kurtosis() if stbvalid else pd.Series([np.NAN]*culnum)
        stb_kurtosis.index=getnewcolname('stablekurtosis')     
        
        stb_xyzmax = stb_max['x_stablemax'] + stb_max['y_stablemax'] + stb_max['z_stablemax']
        stb_xyzmin = stb_min['x_stablemin'] + stb_min['y_stablemin'] + stb_min['z_stablemin']
        stb_xyzstd = stb_std['x_stablestd'] + stb_std['y_stablestd'] + stb_std['z_stablestd']
        stb_xyzmean = stb_mean['x_stablemean'] + stb_mean['y_stablemean'] + stb_mean['z_stablemean']
        stb_loadsum = pd.Series([stb_xyzmax,stb_xyzmin,stb_xyzstd,stb_xyzmean])
        stb_loadsum.index=['stableloadmax','stableloadmin','stableloadstd','stableloadmean']
        
        try:
            rng = pd.date_range('1/1/2011', periods=stable.shape[0], freq='min')
            stable = stable.set_index(rng)
            stb_re = stable.resample('10min')
            stbre_med = stb_re.median()
            stb_fdiff = stbre_med - stbre_med.shift()            
            stb_fdiff.iloc[0] = stb_fdiff.iloc[1]  
        except:
            #print('stb,',idx)
            pass            
        stb_fdiff_mean = np.mean(np.abs(stb_fdiff)) if stbvalid else pd.Series([np.NAN]*culnum)
        stb_fdiff_mean.index=getnewcolname('stablefdiffmean')
        stb_fdiff_max = np.max(stb_fdiff) if stbvalid else pd.Series([np.NAN]*culnum)
        stb_fdiff_max.index=getnewcolname('stablefdiffmax')   
        stb_fdiff_min = np.min(stb_fdiff) if stbvalid else pd.Series([np.NAN]*culnum)
        stb_fdiff_min.index=getnewcolname('stablefdiffmin')         
        stb_fdiff_std = np.std(stb_fdiff) if stbvalid else pd.Series([np.NAN]*culnum)
        stb_fdiff_std.index=getnewcolname('stablefdiffstd')
        stb_fdiff_maxmin =  np.max(stb_fdiff) - np.min(stb_fdiff) if stbvalid else pd.Series([np.NAN]*culnum)
        stb_fdiff_maxmin.index=getnewcolname('stablefdiffmaxmin')
        stb_fdiff_skew = pd.Series(stats.skew(stb_fdiff)) if stbvalid else pd.Series([np.NAN]*culnum)
        stb_fdiff_skew.index=getnewcolname('stablefdiffskew')        
        stb_fdiff_kurtosis = pd.Series(stats.kurtosis(stb_fdiff)) if stbvalid else pd.Series([np.NAN]*culnum)
        stb_fdiff_kurtosis.index=getnewcolname('stablefdiffkurtosis') 
        
        tmp, fqty = getmainfreqmag(stable_ori)
        stb_freqoa1 = pd.Series(tmp[9::fqty])
        stb_freqoa1.index=getnewcolname('stableFreqOA1')
        
        stb_fremean = pd.Series(tmp[10::fqty])
        stb_fremean.index=getnewcolname('stableFreqMean')
        
        stb_frestd = pd.Series(tmp[11::fqty])
        stb_frestd.index=getnewcolname('stableFreqStd') 
        
        stb_fremax = pd.Series(tmp[12::fqty])
        stb_fremax.index=getnewcolname('stableFreqMax')
        
        stb_fremin = pd.Series(tmp[13::fqty])
        stb_fremin.index=getnewcolname('stableFreqMin')        
        
        stb_frep2p = pd.Series(tmp[14::fqty])
        stb_frep2p.index=getnewcolname('stableFreqP2P')                
        
        stb_freqoa2 = pd.Series(tmp[15::fqty])
        stb_freqoa2.index=getnewcolname('stableFreqOA2')

        stb_fremean2 = pd.Series(tmp[16::fqty])
        stb_fremean2.index=getnewcolname('stableFreqMean2')
        
        stb_frestd2 = pd.Series(tmp[17::fqty])
        stb_frestd2.index=getnewcolname('stableFreqStd2') 
        
        stb_fremax2 = pd.Series(tmp[18::fqty])
        stb_fremax2.index=getnewcolname('stableFreqMax2')
        
        stb_fremin2 = pd.Series(tmp[19::fqty])
        stb_fremin2.index=getnewcolname('stableFreqMin2')        
        
        stb_frep2p2 = pd.Series(tmp[20::fqty])
        stb_frep2p2.index=getnewcolname('stableFreqP2P2')    
        stb_fresnr = pd.Series(tmp[21::fqty])
        stb_fresnr.index=getnewcolname('stableFreqSNR')    

        x_aggr = pd.concat([cut1_mean, cut1_min, cut1_fdiff_min, cut1_std, cut1_max, cut1_rms, cut1_maxmin, cut1_fdiff_kurtosis, cut1_fdiff_mean, cut1_fdiff_max, cut1_fdiff_std, cut1_fdiff_maxmin, cut1_fdiff_skew ,cut2_mean, cut2_min, cut2_fdiff_min, cut2_std, cut2_max, cut2_rms, cut2_maxmin, cut2_fdiff_kurtosis, cut2_fdiff_mean, cut2_fdiff_max, cut2_fdiff_std, cut2_fdiff_maxmin, cut2_fdiff_skew, stb_loadsum ,stb_mean, stb_min, stb_fdiff_min, stb_std, stb_max, stb_rms, stb_maxmin, stb_fdiff_kurtosis, stb_fdiff_mean, stb_fdiff_max, stb_fdiff_std, stb_fdiff_maxmin,stb_fdiff_skew, stb_freqoa1, stb_fremean, stb_frestd, stb_fremax, stb_fremin, stb_frep2p, stb_freqoa2, stb_fremean2, stb_frestd2, stb_fremax2 ,stb_fremin2, stb_frep2p2, stb_fresnr], axis=0)
        data_X.append(x_aggr)
    data_X = pd.DataFrame(data_X)    
    data_X.columns=x_aggr.index
    print(data_X.shape)
    return data_X
    

def cnc_predict(model_mean, model_max):
    def moving_average(a, n=3) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n     
    def pltsize(y,x):
        plt.figure(figsize=(y,x))
    def pltutil(xlabel='', ylabel='', title='', legend=True):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        if legend:
            plt.legend()
        plt.tight_layout()
        
    pred_TX_mean = model_mean.predict(tsx.values)
    pred_TX_max = model_max.predict(tsx.values)

    #---- 頻域的特徵在200左右突然往上抖升, 所以200後的預測值嘗試用max版的model預測結果 ----
    pred_TX = pred_TX_mean.copy()
    pred_TX[:200] = pred_TX_mean[:200] 
    pred_TX[200:] = pred_TX_max[200:]
    #---- 對預測的磨耗值平滑化(moving average, window size=5)
    pred_TX_ma = pred_TX.copy()
    pred_TX_ma[4:] = moving_average(pred_TX, n=5)


    #---- 計算cut-time ----
    cutcnt_max=[]
    pred_wear = np.array(pred_TX_ma)
    for i in range(51, 201):
        maxidx = np.where(pred_wear<=i)[0]
        if len(maxidx) == 0:
            cutcnt_max.append(0)
        else:
            maxidx = maxidx[-1]
            cutcnt_max.append(maxidx)

    #---- 60~70陡升, 嘗試使用後面兩期修正lag(-2) ----
    cutcnt_max[60:70] = cutcnt_max[62:72]


    pltsize(10,4)
    plt.subplot(221)
    plt.plot(pred_TX_mean, label='model_mean')
    plt.plot(pred_TX_max, label='model_max')
    pltutil('铣切次數','刀具磨耗值','刀具磨耗預測',legend=True)    

    plt.subplot(222)
    plt.plot(pred_TX, '.', label='predict')
    plt.plot(pred_TX_ma, label='predict_MA5')
    pltutil('铣切次數', '刀具磨耗值','刀具磨耗預測(合併)', legend=True)

    plt.subplot(223)    
    plt.plot(cutcnt_max)
    pltutil('磨耗限制', '可铣切次數', '刀具壽命預測', legend=False)
    plt.show()    
    return cutcnt_max