#!/usr/bin/env python
# coding: utf-8

# Copyright © 2019 Hsu Shih-Chieh

import chardet, json, datetime, os
from nptdms import TdmsFile
import pandas as pd 


def checkencoding(fname):
    with open(fname, 'rb') as f:
        result = chardet.detect(f.read())  # or readline if the file is large
    return result['encoding']


def readTDMSasDF(path, cols):
    try:
        tdms_file = TdmsFile(path)
    except:
        return pd.DataFrame()
    #tdms_df = tdms_file.as_dataframe()
    data_dict = {}
    if len(tdms_file.objects.keys())==0:
        return pd.DataFrame()
    
    for p in tdms_file.objects.keys():
        path_list = list(filter(None, p.split('/')))
        path_list = list(map(lambda s: s.replace("'", ""), path_list ))
        if ('_'.join(path_list) not in cols) or (not path_list) :
            continue
        if len(path_list)==2:
            obj=tdms_file.object(path_list[0],path_list[1])
            data = obj.data
            data_dict['_'.join(path_list)] = data
    time = obj.time_track()
    return pd.DataFrame(data_dict, index = time)
    #return pd.DataFrame(data_dict, index = list(time_dict.values())[0])


class MCase(object):
    def __init__(self, caseid, cpath, **kwargs):
        case = json.load(open(cpath, 'r'))
        casecfg=case[caseid]
        self.strdt = datetime.datetime.strptime(casecfg['data_strdt'], '%Y-%m-%d %H:%M:%S')
        self.enddt = datetime.datetime.strptime(casecfg['data_enddt'], '%Y-%m-%d %H:%M:%S')
        self.strts, self.endts = self.strdt.timestamp(), self.enddt.timestamp()
        self.evt_str, self.evt_end = casecfg['idx_evt'], casecfg['idx_evt2']
        self.mid = casecfg['machine']
        self.cname = casecfg['cname']
        
