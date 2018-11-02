
# coding: utf-8

# In[1]:


import os
import talib as ta
import requests
import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import json
import random
import time
import trade
from multiprocessing.pool import ThreadPool
from model_1min_normal import AE_LSTM
from model_5min_normal import AE_LSTM as AE_LSTM_5min
ta.get_function_groups


span = 5
cond_count = 2184
coin_count = 6
coin_balance_count = coin_count +1 # + 1 BNB coin(When changing the coin, we use coin/BNB, because of the fee (this function is removed because of low amount of BNB volumn)).

batch_size = coin_count
load_file_len = 148
COIN = ['BTCUSDT', 'ETHUSDT', 'EOSUSDT', 'NEOUSDT', 'ADAUSDT', 'XRPUSDT', 'BNBUSDT']
c_idx_getval = {0:'BTC', 1:'ETH', 2:'EOS', 3:'NEO', 4:'ADA', 5:'XRP', 6:'BNB', 7:'USDT'}

dir_ = '../0_training/0_get_ticker/' #Ticker dir
file_dir_list = [dir_+'btcusdt/', dir_+'ethusdt/', dir_+'eosusdt/', dir_+'neousdt/', dir_+'adausdt/', dir_+'xrpusdt/'] # len == coin_count
file_num_list = [idx for idx in range(coin_count)]
line_token = "Write your line messanger token"
line_error_token = "Write your line messanger token"

pool = ThreadPool(coin_count)
TRADE_PHASE = 'REAL' #REAL or FAKE


# In[2]:


def init_read(file_dir):
    set_file_length = load_file_len * span

    for idx in range(10):
        try:
            f = open(file_dir + 'last_file_num.txt', 'r')
            last_file_num = int(f.readline())
            f.close()
            break
        except:
            time.sleep(0.05)
            
    dataframe = pd.DataFrame()
    for idx in range(set_file_length):
        for idy in range(set_file_length):
            try:
                dataframe = dataframe.append(pd.read_csv(file_dir + "{}.csv".format(last_file_num-(set_file_length-1 + idy)+idx), header=0, dtype='float32'))
                break
            except:
                print('[!]data_tiling : {}'.format(last_file_num-(set_file_length-1 + idy)+idx))
                time.sleep(0.05)
    #
    dataframe_1min = dataframe[-load_file_len:].copy()
    
    #
    dataframe_5min = pd.DataFrame()
    open_list = []
    high_list = []
    low_list = []
    close_list = []
    volume_list = []
    wp_list = []

    for idx in range(len(dataframe)//span):
        open_list.append(dataframe['open'][idx*span:(idx+1)*span].values[0])
        high_list.append(dataframe['high'][idx*span:(idx+1)*span].values.max())
        low_list.append(dataframe['low'][idx*span:(idx+1)*span].values.min())
        close_list.append(dataframe['close'][idx*span:(idx+1)*span].values[-1])
        volume_list.append(dataframe['volume'][idx*span:(idx+1)*span].values.sum())
        wp_list.append((dataframe['volume'][idx*span:(idx+1)*span]*dataframe['weigted_price'][idx*span:(idx+1)*span]).sum()/volume_list[idx])

    dataframe_5min['open'] = open_list
    dataframe_5min['high'] = high_list
    dataframe_5min['low'] = low_list
    dataframe_5min['close'] = close_list
    dataframe_5min['volume'] = volume_list
    dataframe_5min['weigted_price'] = wp_list
    
    return dataframe_1min, dataframe_5min, last_file_num, dataframe


def next_read(file_dir, dataframe_1m, dataframe, last2_file_num):
    f = open(file_dir + 'last_file_num.txt', 'r')
    last_file_num = int(f.readline())
    f.close()
    
    load_count = last_file_num - last2_file_num
    
    #1min
    dataframe_1m = dataframe_1m.reset_index(drop=True)
    dataframe_1m = dataframe_1m.drop([idx for idx in range(load_count)])
    #5min
    dataframe = dataframe.reset_index(drop=True)
    dataframe = dataframe.drop([idx for idx in range(load_count)])
    for idx in range(load_count):
        for idy in range(load_count):
            try:
                wait_read = pd.read_csv(file_dir + "{}.csv".format(last_file_num-(load_count-1 +idy) +idx), header=0, dtype='float32')
                dataframe_1m = dataframe_1m.append(wait_read)
                dataframe = dataframe.append(wait_read)
                break
            except:
                print('[!]data_tiling : {}'.format(last_file_num-(set_file_length-1 + idy)+idx))
                time.sleep(0.05)
            
    dataframe_1m = dataframe_1m.reset_index(drop=True)
    dataframe = dataframe.reset_index(drop=True)
    
    #make 5min dataframe
    dataframe_5m = pd.DataFrame()
    open_list = []
    high_list = []
    low_list = []
    close_list = []
    volume_list = []
    wp_list = []

    for idx in range(len(dataframe)//span):
        open_list.append(dataframe['open'][idx*span:(idx+1)*span].values[0])
        high_list.append(dataframe['high'][idx*span:(idx+1)*span].values.max())
        low_list.append(dataframe['low'][idx*span:(idx+1)*span].values.min())
        close_list.append(dataframe['close'][idx*span:(idx+1)*span].values[-1])
        volume_list.append(dataframe['volume'][idx*span:(idx+1)*span].values.sum())
        wp_list.append((dataframe['volume'][idx*span:(idx+1)*span]*dataframe['weigted_price'][idx*span:(idx+1)*span]).sum()/volume_list[idx])

    dataframe_5m['open'] = open_list
    dataframe_5m['high'] = high_list
    dataframe_5m['low'] = low_list
    dataframe_5m['close'] = close_list
    dataframe_5m['volume'] = volume_list
    dataframe_5m['weigted_price'] = wp_list
    
    return dataframe_1m, dataframe_5m, last_file_num, dataframe


# In[3]:


#--changes
def Changes(dataframe, inputs):
    next_ = list(dataframe[inputs])
    last_ = [None] + next_

    changes = []
    len_ = len(next_)
    for idx in range(len_):
        if idx == 0:
            changes.append(None)
            continue
        if (next_[idx] - last_[idx]) == 0.0:
            changes.append(0.0)
        
        #
        elif (next_[idx] - last_[idx]) != 0.0 and last_[idx] == 0.0:
            changes.append(((next_[idx] - last_[idx])/0.00000001) * 100)
            
        else:
            changes.append(((next_[idx] - last_[idx])/last_[idx]) * 100)

    dataframe[inputs + '_Changes'] = changes

#label
def Labeling(dataframe, inputs):
    next_price = list(dataframe[inputs])
    last_price = [None] + next_price

    change_list = []
    length = len(next_price)
    for idx in range(length):
        if idx == 0:
            change_list.append(None)
            continue
        change_list.append(((next_price[idx] - last_price[idx])/last_price[idx]) * 100)
        
    return change_list


# In[4]:


def second_making_label(dataframe, file_num):
    WP_change_list = Labeling(dataframe, 'weigted_price')
    H_change_list = Labeling(dataframe, 'high')
    L_change_list = Labeling(dataframe, 'low')

    label_list = []
    for idx, lis in enumerate(WP_change_list):
        if lis == None:
            label_list.append(None)

        elif lis > 0.30:
            label_list.append(0) 
        elif lis > 0.15 and lis <= 0.30:
            label_list.append(1)
        elif lis > 0.10 and lis <= 0.15:
            label_list.append(2)
        elif lis > 0.06 and lis <= 0.10:
            label_list.append(3)
        elif lis > 0.03 and lis <= 0.06:
            label_list.append(4)
        elif lis > 0.01 and lis <= 0.03:
            label_list.append(5)
        elif lis > 0.00 and lis <= 0.01:
            label_list.append(6)
        elif lis == 0.0:
            label_list.append(7)
        elif lis < 0.00 and lis >= -0.01:
            label_list.append(8)
        elif lis < 0.01 and lis >= -0.03:
            label_list.append(9)
        elif lis < 0.03 and lis >= -0.06:
            label_list.append(10)
        elif lis < -0.06 and lis >= -0.10:
            label_list.append(11)
        elif lis < -0.10 and lis >= -0.15:
            label_list.append(12)
        elif lis < -0.15 and lis >= -0.30:
            label_list.append(13)
        elif lis < -0.30:
            label_list.append(14)
    label_list_2 = label_list + [None]
    del(label_list_2[0])

    dataframe['Label_WP'] = WP_change_list
    dataframe['Label_H'] = H_change_list
    dataframe['Label_L'] = L_change_list

    dataframe['Label_class'] = label_list
    dataframe['Label_class_2'] = label_list_2
    
    dataframe['Label_coin'] = file_num
    
    return dataframe

def second_making_label_5min(dataframe, file_num):
    WP_change_list = Labeling(dataframe, 'weigted_price')
    H_change_list = Labeling(dataframe, 'high')
    L_change_list = Labeling(dataframe, 'low')

    label_list = []
    for idx, lis in enumerate(WP_change_list):
        if lis == None:
            label_list.append(None)

        elif lis > 0.5:
            label_list.append(0) 
        elif lis > 0.28 and lis <= 0.5:
            label_list.append(1)
        elif lis > 0.18 and lis <= 0.28:
            label_list.append(2)
        elif lis > 0.1 and lis <= 0.18:
            label_list.append(3)
        elif lis > 0.06 and lis <= 0.1:
            label_list.append(4)
        elif lis > 0.03 and lis <= 0.06:
            label_list.append(5)
        elif lis > 0.00 and lis <= 0.03:
            label_list.append(6)
        elif lis == 0.0:
            label_list.append(7)
        elif lis < 0.00 and lis >= -0.03:
            label_list.append(8)
        elif lis < 0.03 and lis >= -0.06:
            label_list.append(9)
        elif lis < 0.06 and lis >= -0.1:
            label_list.append(10)
        elif lis < -0.1 and lis >= -0.18:
            label_list.append(11)
        elif lis < -0.18 and lis >= -0.28:
            label_list.append(12)
        elif lis < -0.28 and lis >= -0.5:
            label_list.append(13)
        elif lis < -0.5:
            label_list.append(14)
    label_list_2 = label_list + [None]
    del(label_list_2[0])

    dataframe['Label_WP'] = WP_change_list
    dataframe['Label_H'] = H_change_list
    dataframe['Label_L'] = L_change_list

    dataframe['Label_class'] = label_list
    dataframe['Label_class_2'] = label_list_2
    
    dataframe['Label_coin'] = file_num

    return dataframe


# In[5]:


def third_making_indicator(dataframe):
    Open_lis = np.array(dataframe['open'], dtype='float')
    High_lis = np.array(dataframe['high'], dtype='float')
    Low_lis = np.array(dataframe['low'], dtype='float')
    Clz_lis = np.array(dataframe['weigted_price'], dtype='float')
    Vol_lis = np.array(dataframe['volume'], dtype='float')

    ##지표##
    SMA_3_C = ta.SMA(Clz_lis, timeperiod=3)
    SMA_5_H = ta.SMA(High_lis, timeperiod=5)
    SMA_5_L = ta.SMA(Low_lis, timeperiod=5)
    SMA_5_C = ta.SMA(Clz_lis, timeperiod=5)
    SMA_10_H = ta.SMA(High_lis, timeperiod=10)
    SMA_10_L = ta.SMA(Low_lis, timeperiod=10)
    SMA_10_C = ta.SMA(Clz_lis, timeperiod=10)
    RSI_2_C = ta.RSI(Clz_lis, timeperiod=2)
    RSI_3_C = ta.RSI(Clz_lis, timeperiod=3)
    RSI_5_C = ta.RSI(Clz_lis, timeperiod=5)
    RSI_7_H = ta.RSI(High_lis, timeperiod=7)
    RSI_7_L = ta.RSI(Low_lis, timeperiod=7)
    RSI_7_C = ta.RSI(Clz_lis, timeperiod=7)
    RSI_14_H = ta.RSI(High_lis, timeperiod=14)
    RSI_14_L = ta.RSI(Low_lis, timeperiod=14)
    RSI_14_C = ta.RSI(Clz_lis, timeperiod=14)
    ADX = ta.ADX(High_lis, Low_lis, Clz_lis, timeperiod=14)
    ADXR = ta.ADXR(High_lis, Low_lis, Clz_lis, timeperiod=14)
    Aroondown, Aroonup = ta.AROON(High_lis, Low_lis, timeperiod=14)
    Aroonosc = ta.AROONOSC(High_lis, Low_lis, timeperiod=14)
    BOP = ta.BOP(Open_lis, High_lis, Low_lis, Clz_lis)
    CMO = ta.CMO(Clz_lis, timeperiod=14)
    DX = ta.DX(High_lis, Low_lis, Clz_lis, timeperiod=14)
    MFI = ta.MFI(High_lis, Low_lis, Clz_lis, Vol_lis, timeperiod=14)
    MINUS_DI = ta.MINUS_DI(High_lis, Low_lis, Clz_lis, timeperiod=14)
    PLUSDI = ta.PLUS_DI(High_lis, Low_lis, Clz_lis, timeperiod=14)
    PPO = ta.PPO(Clz_lis, fastperiod=12, slowperiod=26, matype=0)
    ROC = ta.ROC(Clz_lis, timeperiod=10)
    ROCP = ta.ROCP(Clz_lis, timeperiod=10)
    ROCR = ta.ROCR(Clz_lis, timeperiod=10)
    ROCR100 = ta.ROCR100(Clz_lis, timeperiod=10)
    Slowk, Slowd = ta.STOCH(High_lis, Low_lis, Clz_lis, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    STOCHF_f, STOCHF_d = ta.STOCHF(High_lis, Low_lis, Clz_lis, fastk_period=5, fastd_period=3, fastd_matype=0)
    Fastk, Fastd = ta.STOCHRSI(Clz_lis, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    TRIX = ta.TRIX(Clz_lis, timeperiod=30)
    ULTOSC = ta.ULTOSC(High_lis, Low_lis, Clz_lis, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    WILLR = ta.WILLR(High_lis, Low_lis, Clz_lis, timeperiod=14)
    ADOSC = ta.ADOSC(High_lis, Low_lis, Clz_lis, Vol_lis, fastperiod=3, slowperiod=10)
    NATR = ta.NATR(High_lis, Low_lis, Clz_lis, timeperiod=14)
    HT_DCPERIOD = ta.HT_DCPERIOD(Clz_lis)
    sine, leadsine = ta.HT_SINE(Clz_lis)
    integer = ta.HT_TRENDMODE(Clz_lis)

    ########
    #append
    dataframe['SMA_3_C'] = SMA_3_C
    dataframe['SMA_5_H'] = SMA_5_H
    dataframe['SMA_5_L'] = SMA_5_L
    dataframe['SMA_5_C'] = SMA_5_C
    dataframe['SMA_10_H'] = SMA_10_H
    dataframe['SMA_10_L'] = SMA_10_L
    dataframe['SMA_10_C'] = SMA_10_C
    dataframe['RSI_2_C'] = (RSI_2_C / 100.) *2 - 1.
    dataframe['RSI_3_C'] = (RSI_3_C / 100.) *2 - 1.
    dataframe['RSI_5_C'] = (RSI_5_C / 100.) *2 - 1.
    dataframe['RSI_7_H'] = (RSI_7_H / 100.) *2 - 1.
    dataframe['RSI_7_L'] = (RSI_7_L / 100.) *2 - 1.
    dataframe['RSI_7_C'] = (RSI_7_C / 100.) *2 - 1.
    dataframe['RSI_14_H'] = (RSI_14_H / 100.) *2 - 1.
    dataframe['RSI_14_L'] = (RSI_14_L / 100.) *2 - 1.
    dataframe['RSI_14_C'] = (RSI_14_C / 100.) *2 - 1.
    dataframe['ADX'] = (ADX / 100.) *2 - 1.
    dataframe['ADXR'] = (ADXR / 100.) *2 - 1.
    dataframe['Aroondown'] = (Aroondown / 100.) *2 - 1
    dataframe['Aroonup'] = (Aroonup / 100.) *2 - 1
    dataframe['Aroonosc'] = Aroonosc / 100.
    dataframe['BOP'] = BOP
    dataframe['CMO'] = CMO / 100.
    dataframe['DX'] = (DX / 100.) *2 - 1
    dataframe['MFI'] = (MFI / 100.) *2 - 1
    dataframe['MINUS_DI'] = (MINUS_DI / 100.) *2 - 1
    dataframe['PLUSDI'] = (PLUSDI / 100.) *2 - 1
    dataframe['PPO'] = PPO / 10.
    dataframe['ROC'] = ROC / 10.
    dataframe['ROCP'] = ROCP * 10
    dataframe['ROCR'] = (ROCR - 1.0) * 100.
    dataframe['ROCR100'] = ((ROCR100 / 100.) - 1.0) * 100.
    dataframe['Slowk'] = (Slowk / 100.) *2 - 1
    dataframe['Slowd'] = (Slowd / 100.) *2 - 1
    dataframe['STOCHF_f'] = (STOCHF_f / 100.) *2 - 1
    dataframe['STOCHF_d'] = (STOCHF_d / 100.) *2 - 1
    dataframe['Fastk'] = (Fastk / 100.) *2 - 1
    dataframe['Fastd'] = (Fastd / 100.) *2 - 1
    dataframe['TRIX'] = TRIX * 10.
    dataframe['ULTOSC'] = (ULTOSC / 100.) *2 - 1
    dataframe['WILLR'] = (WILLR / 100.) *2 + 1
    dataframe['ADOSC'] = ADOSC / 100.
    dataframe['NATR'] = NATR *2 -1
    dataframe['HT_DCPERIOD'] = (HT_DCPERIOD / 100.) *2 -1
    dataframe['sine'] = sine
    dataframe['leadsine'] = leadsine
    dataframe['integer'] = integer
    
    return dataframe

def third_making_indicator_5min(dataframe):
    Open_lis = np.array(dataframe['open'], dtype='float')
    High_lis = np.array(dataframe['high'], dtype='float')
    Low_lis = np.array(dataframe['low'], dtype='float')
    Clz_lis = np.array(dataframe['weigted_price'], dtype='float')
    Vol_lis = np.array(dataframe['volume'], dtype='float')

    ##지표##
    SMA_3_C = ta.SMA(Clz_lis, timeperiod=3)
    SMA_5_H = ta.SMA(High_lis, timeperiod=5)
    SMA_5_L = ta.SMA(Low_lis, timeperiod=5)
    SMA_5_C = ta.SMA(Clz_lis, timeperiod=5)
    SMA_10_H = ta.SMA(High_lis, timeperiod=10)
    SMA_10_L = ta.SMA(Low_lis, timeperiod=10)
    SMA_10_C = ta.SMA(Clz_lis, timeperiod=10)
    RSI_2_C = ta.RSI(Clz_lis, timeperiod=2)
    RSI_3_C = ta.RSI(Clz_lis, timeperiod=3)
    RSI_5_C = ta.RSI(Clz_lis, timeperiod=5)
    RSI_7_H = ta.RSI(High_lis, timeperiod=7)
    RSI_7_L = ta.RSI(Low_lis, timeperiod=7)
    RSI_7_C = ta.RSI(Clz_lis, timeperiod=7)
    RSI_14_H = ta.RSI(High_lis, timeperiod=14)
    RSI_14_L = ta.RSI(Low_lis, timeperiod=14)
    RSI_14_C = ta.RSI(Clz_lis, timeperiod=14)
    ADX = ta.ADX(High_lis, Low_lis, Clz_lis, timeperiod=14)
    ADXR = ta.ADXR(High_lis, Low_lis, Clz_lis, timeperiod=14)
    Aroondown, Aroonup = ta.AROON(High_lis, Low_lis, timeperiod=14)
    Aroonosc = ta.AROONOSC(High_lis, Low_lis, timeperiod=14)
    BOP = ta.BOP(Open_lis, High_lis, Low_lis, Clz_lis)
    CMO = ta.CMO(Clz_lis, timeperiod=14)
    DX = ta.DX(High_lis, Low_lis, Clz_lis, timeperiod=14)
    MFI = ta.MFI(High_lis, Low_lis, Clz_lis, Vol_lis, timeperiod=14)
    MINUS_DI = ta.MINUS_DI(High_lis, Low_lis, Clz_lis, timeperiod=14)
    PLUSDI = ta.PLUS_DI(High_lis, Low_lis, Clz_lis, timeperiod=14)
    PPO = ta.PPO(Clz_lis, fastperiod=12, slowperiod=26, matype=0)
    ROC = ta.ROC(Clz_lis, timeperiod=10)
    ROCP = ta.ROCP(Clz_lis, timeperiod=10)
    ROCR = ta.ROCR(Clz_lis, timeperiod=10)
    ROCR100 = ta.ROCR100(Clz_lis, timeperiod=10)
    Slowk, Slowd = ta.STOCH(High_lis, Low_lis, Clz_lis, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    STOCHF_f, STOCHF_d = ta.STOCHF(High_lis, Low_lis, Clz_lis, fastk_period=5, fastd_period=3, fastd_matype=0)
    Fastk, Fastd = ta.STOCHRSI(Clz_lis, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    TRIX = ta.TRIX(Clz_lis, timeperiod=30)
    ULTOSC = ta.ULTOSC(High_lis, Low_lis, Clz_lis, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    WILLR = ta.WILLR(High_lis, Low_lis, Clz_lis, timeperiod=14)
    ADOSC = ta.ADOSC(High_lis, Low_lis, Clz_lis, Vol_lis, fastperiod=3, slowperiod=10)
    NATR = ta.NATR(High_lis, Low_lis, Clz_lis, timeperiod=14)
    HT_DCPERIOD = ta.HT_DCPERIOD(Clz_lis)
    sine, leadsine = ta.HT_SINE(Clz_lis)
    integer = ta.HT_TRENDMODE(Clz_lis)

    ########
    #append
    dataframe['SMA_3_C'] = SMA_3_C
    dataframe['SMA_5_H'] = SMA_5_H
    dataframe['SMA_5_L'] = SMA_5_L
    dataframe['SMA_5_C'] = SMA_5_C
    dataframe['SMA_10_H'] = SMA_10_H
    dataframe['SMA_10_L'] = SMA_10_L
    dataframe['SMA_10_C'] = SMA_10_C
    dataframe['RSI_2_C'] = (RSI_2_C / 100.) *2 - 1.
    dataframe['RSI_3_C'] = (RSI_3_C / 100.) *2 - 1.
    dataframe['RSI_5_C'] = (RSI_5_C / 100.) *2 - 1.
    dataframe['RSI_7_H'] = (RSI_7_H / 100.) *2 - 1.
    dataframe['RSI_7_L'] = (RSI_7_L / 100.) *2 - 1.
    dataframe['RSI_7_C'] = (RSI_7_C / 100.) *2 - 1.
    dataframe['RSI_14_H'] = (RSI_14_H / 100.) *2 - 1.
    dataframe['RSI_14_L'] = (RSI_14_L / 100.) *2 - 1.
    dataframe['RSI_14_C'] = (RSI_14_C / 100.) *2 - 1.
    dataframe['ADX'] = (ADX / 100.) *2 - 1.
    dataframe['ADXR'] = (ADXR / 100.) *2 - 1.
    dataframe['Aroondown'] = (Aroondown / 100.) *2 - 1
    dataframe['Aroonup'] = (Aroonup / 100.) *2 - 1
    dataframe['Aroonosc'] = Aroonosc / 100.
    dataframe['BOP'] = BOP
    dataframe['CMO'] = CMO / 100.
    dataframe['DX'] = (DX / 100.) *2 - 1
    dataframe['MFI'] = (MFI / 100.) *2 - 1
    dataframe['MINUS_DI'] = (MINUS_DI / 100.) *2 - 1
    dataframe['PLUSDI'] = (PLUSDI / 100.) *2 - 1
    dataframe['PPO'] = PPO / 10.
    dataframe['ROC'] = ROC / 10.
    dataframe['ROCP'] = ROCP * 10
    dataframe['ROCR'] = (ROCR - 1.0) * 100.
    dataframe['ROCR100'] = ((ROCR100 / 100.) - 1.0) * 100.
    dataframe['Slowk'] = (Slowk / 100.) *2 - 1
    dataframe['Slowd'] = (Slowd / 100.) *2 - 1
    dataframe['STOCHF_f'] = (STOCHF_f / 100.) *2 - 1
    dataframe['STOCHF_d'] = (STOCHF_d / 100.) *2 - 1
    dataframe['Fastk'] = (Fastk / 100.) *2 - 1
    dataframe['Fastd'] = (Fastd / 100.) *2 - 1
    dataframe['TRIX'] = TRIX * 10.
    dataframe['ULTOSC'] = (ULTOSC / 100.) *2 - 1
    dataframe['WILLR'] = (WILLR / 100.) *2 + 1
    dataframe['ADOSC'] = ADOSC / 100.
    dataframe['NATR'] = NATR *2 -1
    dataframe['HT_DCPERIOD'] = (HT_DCPERIOD / 100.) *2 -1
    dataframe['sine'] = sine
    dataframe['leadsine'] = leadsine
    dataframe['integer'] = integer
    
    return dataframe


# In[6]:


def fourth_calc_changes(dataframe):
    #--calcul_changes
    apply_list = ['open','high','low','close','weigted_price','volume','SMA_3_C','SMA_5_H','SMA_5_L','SMA_5_C','SMA_10_H','SMA_10_L','SMA_10_C']

    for idx, lis in enumerate(apply_list):
        Changes(dataframe, lis)
    
    return dataframe

def fourth_calc_changes_5min(dataframe):
    #--calcul_changes
    apply_list = ['open','high','low','close','weigted_price','volume','SMA_3_C','SMA_5_H','SMA_5_L','SMA_5_C','SMA_10_H','SMA_10_L','SMA_10_C']

    for idx, lis in enumerate(apply_list):
        Changes(dataframe, lis)
    
    return dataframe


# In[7]:


def fifth_drop(dataframe):
    
    dataframe = dataframe.reset_index(drop=True)
    dataframe['open'] = ((dataframe['open'] / dataframe['weigted_price']) - 1.0) * 100.
    dataframe['high'] = ((dataframe['high'] / dataframe['weigted_price']) - 1.0) * 100.
    dataframe['low'] = ((dataframe['low'] / dataframe['weigted_price']) - 1.0) * 100.
    dataframe['close'] = ((dataframe['close'] / dataframe['weigted_price']) - 1.0) * 100.
    dataframe['SMA_3_C'] = ((dataframe['SMA_3_C'] / dataframe['weigted_price']) - 1.0) * 100.
    dataframe['SMA_5_H'] = ((dataframe['SMA_5_H'] / dataframe['weigted_price']) - 1.0) * 100.
    dataframe['SMA_5_L'] = ((dataframe['SMA_5_L'] / dataframe['weigted_price']) - 1.0) * 100.
    dataframe['SMA_5_C'] = ((dataframe['SMA_5_C'] / dataframe['weigted_price']) - 1.0) * 100.
    dataframe['SMA_10_H'] = ((dataframe['SMA_10_H'] / dataframe['weigted_price']) - 1.0) * 100.
    dataframe['SMA_10_L'] = ((dataframe['SMA_10_L'] / dataframe['weigted_price']) - 1.0) * 100.
    dataframe['SMA_10_C'] = ((dataframe['SMA_10_C'] / dataframe['weigted_price']) - 1.0) * 100.
    #
    data_list_ = dataframe['volume_Changes'].values

    for idx, lis in enumerate(data_list_):
        if lis >= -10 and lis <= 10:
            data_list_[idx] = 0
        elif lis > 10:
            data_list_[idx] = 1
        elif lis < -10:
            data_list_[idx] = -1

    dataframe['volume_Changes'] = data_list_
    #
    
    dataframe = dataframe.drop(['volume','weigted_price'], axis=1)
    dataframe = dataframe.drop([idx for idx in range(88)])
    dataframe = dataframe.reset_index(drop=True)
    
    return dataframe

def fifth_drop_5min(dataframe):
    
    dataframe = dataframe.reset_index(drop=True)
    dataframe['open'] = ((dataframe['open'] / dataframe['weigted_price']) - 1.0) * 100.
    dataframe['high'] = ((dataframe['high'] / dataframe['weigted_price']) - 1.0) * 100.
    dataframe['low'] = ((dataframe['low'] / dataframe['weigted_price']) - 1.0) * 100.
    dataframe['close'] = ((dataframe['close'] / dataframe['weigted_price']) - 1.0) * 100.
    dataframe['SMA_3_C'] = ((dataframe['SMA_3_C'] / dataframe['weigted_price']) - 1.0) * 100.
    dataframe['SMA_5_H'] = ((dataframe['SMA_5_H'] / dataframe['weigted_price']) - 1.0) * 100.
    dataframe['SMA_5_L'] = ((dataframe['SMA_5_L'] / dataframe['weigted_price']) - 1.0) * 100.
    dataframe['SMA_5_C'] = ((dataframe['SMA_5_C'] / dataframe['weigted_price']) - 1.0) * 100.
    dataframe['SMA_10_H'] = ((dataframe['SMA_10_H'] / dataframe['weigted_price']) - 1.0) * 100.
    dataframe['SMA_10_L'] = ((dataframe['SMA_10_L'] / dataframe['weigted_price']) - 1.0) * 100.
    dataframe['SMA_10_C'] = ((dataframe['SMA_10_C'] / dataframe['weigted_price']) - 1.0) * 100.
    #
    data_list_ = dataframe['volume_Changes'].values

    for idx, lis in enumerate(data_list_):
        if lis >= -10 and lis <= 10:
            data_list_[idx] = 0
        elif lis > 10:
            data_list_[idx] = 1
        elif lis < -10:
            data_list_[idx] = -1

    dataframe['volume_Changes'] = data_list_
    #
    
    dataframe = dataframe.drop(['volume','weigted_price'], axis=1)
    dataframe = dataframe.drop([idx for idx in range(88)])
    dataframe = dataframe.reset_index(drop=True)
    
    return dataframe


# In[8]:


def Send_Line(token, message):
    try:
        url = "https://notify-api.line.me/api/notify"
        token = token
        headers = {"Authorization" : "Bearer "+ token}
        payload = {"message" :  message}

        r = requests.post(url ,headers = headers ,params=payload)
    except:
        pass


# In[9]:


#----- main loop-----#
#build model
class main_(object):
    def __init__(self):
        #-graph
        g_1 = tf.Graph()
        sess_1 = tf.InteractiveSession(graph=g_1)  
        self.model = AE_LSTM(sess_1, batch_size)
        g_2 = tf.Graph()
        sess_2 = tf.InteractiveSession(graph=g_2) 
        self.model_5min = AE_LSTM_5min(sess_2, batch_size)
        
        #-for dataframe
        self.queue_list_1min = [[0,0,0,0,0,0] for idx in range(6)]
        self.queue_list_5min = [[0,0,0,0,0,0] for idx in range(6)]
        self.origin_dataframe_list_5min = [[0,0,0,0,0,0] for idx in range(6)]
        self.last_file_num = [[0,0,0,0,0,0] for idx in range(6)]
        self.save_last_rate = 100.00
        self.save_now_rate = 100.00
        self.predict_change_1m = 'UP'
        self.predict_change_5m = 'UP'
        
        #-common
        self.first_price = trade.get_price()
        time.sleep(0.3)
        
        self.first_balance = trade.get_balance()
        time.sleep(0.3)

        self.first_btc_deposit_history = trade.last_deposit_return()
        time.sleep(0.3)
        
        self.first_btc_withdraw_history = trade.last_withdraw_return()
        self.folder_dir = ''
        self.trade_penalty = 0.20
        self.global_num = 0
        self.error_count = 0
        self.error_reset_start = 0
                
        ##-- for backtest
        #init
        self.what_was_last_trade = ['BUY' for idx in range(cond_count+1)] #[BUY, SELL]
        self.what_was_last_traded_coin = [0 for idx in range(cond_count+1)]
        self.Total_money = [0 for idx in range(cond_count+1)] # yen
        self.Total_coin = [[0 for i in range(coin_balance_count)] for idx in range(cond_count+1)] #[20,6]
        for idx in range(cond_count+1):
            self.Total_coin[idx][0] = (10000/self.first_price[0])
        self.Total_Deposit = [10000 for idx in range(cond_count+1)]
        self.trade_phase = [0 for idx in range(cond_count+1)] #[0:nothing, 1:buy, 2:sell]
        
        ##-- for trade
        #trade init
        self.Total_money[0] = self.first_balance[-1] #JPY # yen
        for idx in range(coin_balance_count):
            self.Total_coin[0][idx] = self.first_balance[idx] #BTC
        
        self.coin_Deposit_sum = (np.array(self.Total_coin[0])*np.array(self.first_price)).sum()
        self.first_Deposit = (self.Total_money[0] + self.coin_Deposit_sum)
        self.Total_Deposit[0] = self.first_Deposit
        
        if self.Total_money[0] >= self.coin_Deposit_sum:
            self.what_was_last_trade[0] = 'SELL' #[BUY, SELL]
        elif self.Total_money[0] < self.coin_Deposit_sum:
            self.what_was_last_trade[0] = 'BUY' #[BUY, SELL]
            
            #0:'BTC', 1:'ETH', 2:'EOS', 3:'NEO', 4:'ADA', 5:'XRP', 6:'USDT'
            self.what_was_last_traded_coin[0] = (np.array(self.Total_coin[0])*np.array(self.first_price)).argmax()
            
    def calc_count_change(self, for_calc_1m, for_calc_5m):
        
        calc_global_num = 0
        for calc_target in [for_calc_1m, for_calc_5m]:
            state_count = 0

            for lis in calc_target:
                if lis >= 0:
                    state_count += 1

            if (state_count > coin_count//2) and calc_global_num == 0:
                change_1m = 'UP'
            elif (state_count > coin_count//2) and calc_global_num == 1:
                change_5m = 'UP'

            elif (state_count == coin_count//2) and calc_global_num == 0:
                change_1m = 'SAME'
            elif (state_count == coin_count//2) and calc_global_num == 1:
                change_5m = 'SAME'

            elif (state_count < coin_count//2) and calc_global_num == 0:
                change_1m = 'DOWN'
            elif (state_count < coin_count//2) and calc_global_num == 1:
                change_5m = 'DOWN'
            
            calc_global_num += 1

        return change_1m, change_5m
                
    def REAL_get_balance(self):
        self.final_balance = trade.get_balance()

        self.Total_money[0] = self.final_balance[-1] #JPY # yen
        for idx in range(coin_balance_count):
            self.Total_coin[0][idx] = self.final_balance[idx] #coin

        self.coin_Deposit_sum = (np.array(self.Total_coin[0])*np.array(self.final_price)).sum()                
        self.Total_Deposit[0] = (self.Total_money[0] + self.coin_Deposit_sum)
        
    #==============================================
    def REAL_trade_order(self,numb,b_cond_1m,b_cond_5m,b_cond_1m_2,b_cond_5m_2,s_cond_1m,s_cond_5m,market_change_status,switch_buy,switch_sell,                         switch_change_coin,b_stay_cond_1m,b_stay_cond_5m,prd_state,switch_prd_buy,switch_prd_sell):
        #init
        if b_cond_1m == 'pass':
            b_cond_1m = -10
        if b_cond_5m == 'pass':
            b_cond_5m = 10 #*
        if b_cond_1m_2 == 'pass':
            b_cond_1m_2 = -10
        if b_cond_5m_2 == 'pass':
            b_cond_5m_2 = -10
        if s_cond_1m == 'pass':
            s_cond_1m = 10
        if s_cond_5m == 'pass':
            s_cond_5m = 10
            
        if b_stay_cond_1m == 'pass':
            b_stay_cond_1m = -10
        if b_stay_cond_5m == 'pass':
            b_stay_cond_5m = -10
            
        #find signal & fine and set max coin
        save_price_1min = []
        save_price_5min = []
        save_coin_idx = []
        for idx in range(len(self.out_price_1min)):
            if (self.out_price_1min[idx] >= b_cond_1m and self.out_price_5min[idx] <= b_cond_5m) or             (self.out_price_1min[idx] >= b_cond_1m_2 and self.out_price_5min[idx] >= b_cond_5m_2):
                save_price_1min.append(self.out_price_1min[idx])
                save_price_5min.append(self.out_price_5min[idx])
                save_coin_idx.append(idx)
                
        if len(save_coin_idx) != 0:
            max_idx = np.array(save_price_1min).argmax()
            
            max_price_1min = save_price_1min[max_idx]
            max_price_5min = save_price_5min[max_idx]
            max_coin_idx = save_coin_idx[max_idx]
        else:
            max_price_1min = self.max_price_1min
            max_price_5min = self.max_price_5min
            max_coin_idx = self.max_coin_idx
            
            
        #####################
        #==<  condition  >==#
        #####################
        
        #set min_price
        min_price_1min = self.out_price_1min[self.what_was_last_traded_coin[numb]]
        min_price_5min = self.out_price_5min[self.what_was_last_traded_coin[numb]]
        
        #--<buy condition>--#
        if ((max_price_1min >= b_cond_1m and max_price_5min <= b_cond_5m) or (max_price_1min >= b_cond_1m_2 and max_price_5min >= b_cond_5m_2)) and         ((switch_buy == 'on' and (market_change_status == 'UP' or market_change_status == 'SAME')) or switch_buy == 'off') and         ((switch_prd_buy == 'on' and (prd_state == 'UP' or prd_state == 'SAME')) or switch_prd_buy == 'off'):
            if self.what_was_last_trade[numb] == 'SELL':
                self.trade_phase[numb] = 1
            
            #★코인 갈아타기 조건
            elif self.what_was_last_trade[numb] == 'BUY' and max_coin_idx != self.what_was_last_traded_coin[numb] and             switch_change_coin == 'on':
                if self.out_price_1min[self.what_was_last_traded_coin[numb]] >= b_stay_cond_1m and                 self.out_price_5min[self.what_was_last_traded_coin[numb]] >= b_stay_cond_5m:
                    self.trade_phase[numb] = 0
                else:
                    self.trade_phase[numb] = 3
                
            else:
                self.trade_phase[numb] = 0
                
        #--<sell condition>--#
        elif min_price_1min <= s_cond_1m and min_price_5min <= s_cond_5m and         ((switch_sell == 'on' and (market_change_status == 'DOWN' or market_change_status == 'SAME')) or switch_sell == 'off') and         ((switch_prd_sell == 'on' and (prd_state == 'DOWN' or prd_state == 'SAME')) or switch_prd_sell == 'off'):
            if self.what_was_last_trade[numb] == 'BUY':
                self.trade_phase[numb] = 2
            else:
                self.trade_phase[numb] = 0
        else:
            self.trade_phase[numb] = 0
            
        #--< trade order >--#
        if self.trade_phase[numb] == 0:
            print('[!]waiting ...')

        elif self.trade_phase[numb] == 1:
            #order
            print('{}, {}'.format(((self.Total_money[numb])/self.final_price[max_coin_idx])*0.97, c_idx_getval[max_coin_idx]+'USDT'))
            result_ = trade.market_buy_order(((self.Total_money[numb])/self.final_price[max_coin_idx])*0.97,                                             c_idx_getval[max_coin_idx]+'USDT')
            self.what_was_last_trade[numb] = 'BUY'
            self.what_was_last_traded_coin[numb] = max_coin_idx
            print('[+]buy ...')

        elif self.trade_phase[numb] == 2:
            #order
            result_ = trade.market_sell_order(self.Total_coin[numb][self.what_was_last_traded_coin[numb]],                                              c_idx_getval[self.what_was_last_traded_coin[numb]]+'USDT')
            self.what_was_last_trade[numb] = 'SELL'
            print('[+]sell ...')
            
        elif self.trade_phase[numb] == 3:
            #first sell
            result_ = trade.market_sell_order(self.Total_coin[numb][self.what_was_last_traded_coin[numb]],                                              c_idx_getval[self.what_was_last_traded_coin[numb]]+'USDT')

            time.sleep(0.5)
            self.REAL_get_balance()
            
            #second buy
            result_ = trade.market_buy_order(((self.Total_money[numb])/self.final_price[max_coin_idx])*0.97,                                             c_idx_getval[max_coin_idx]+'USDT')
            
            self.what_was_last_trade[numb] = 'BUY'
            self.what_was_last_traded_coin[numb] = max_coin_idx
            print('[+]changing ...')
            
    def trade_order(self,numb,b_cond_1m,b_cond_5m,s_cond_1m,s_cond_5m,market_change_status,switch_buy,switch_sell,                    switch_change_coin,b_stay_cond_1m,b_stay_cond_5m):
        #init
        if b_cond_1m == 'pass':
            b_cond_1m = -10
        if b_cond_5m == 'pass':
            b_cond_5m = -10
        if s_cond_1m == 'pass':
            s_cond_1m = 10
        if s_cond_5m == 'pass':
            s_cond_5m = 10
            
        if b_stay_cond_1m == 'pass':
            b_stay_cond_1m = -10
        if b_stay_cond_5m == 'pass':
            b_stay_cond_5m = -10
        
        #set min_price
        min_price_1min = self.out_price_1min[self.what_was_last_traded_coin[numb]]
        min_price_5min = self.out_price_5min[self.what_was_last_traded_coin[numb]]
        
        #find signal & fine and set max coin
        save_price_1min = []
        save_price_5min = []
        save_coin_idx = []
        for idx in range(len(self.out_price_1min)):
            if self.out_price_1min[idx] >= b_cond_1m and self.out_price_5min[idx] >= b_cond_5m:
                save_price_1min.append(self.out_price_1min[idx])
                save_price_5min.append(self.out_price_5min[idx])
                save_coin_idx.append(idx)
                
        if len(save_coin_idx) != 0:
            max_idx = np.array(save_price_1min).argmax()
            
            max_price_1min = save_price_1min[max_idx]
            max_price_5min = save_price_5min[max_idx]
            max_coin_idx = save_coin_idx[max_idx]
        else:
            max_price_1min = self.max_price_1min
            max_price_5min = self.max_price_5min
            max_coin_idx = self.max_coin_idx
            
            
        #####################
        #==<  condition  >==#
        #####################
        
        #--<buy condition>--#
        if max_price_1min >= b_cond_1m and max_price_5min >= b_cond_5m and         ((switch_buy == 'on' and (market_change_status == 'UP' or market_change_status == 'SAME')) or switch_buy == 'off'):
            if self.what_was_last_trade[numb] == 'SELL':
                self.trade_phase[numb] = 1
            
            #★코인 갈아타기 조건
            elif self.what_was_last_trade[numb] == 'BUY' and max_coin_idx != self.what_was_last_traded_coin[numb] and             switch_change_coin == 'on':
                if self.out_price_1min[self.what_was_last_traded_coin[numb]] >= b_stay_cond_1m and                 self.out_price_5min[self.what_was_last_traded_coin[numb]] >= b_stay_cond_5m:
                    self.trade_phase[numb] = 0
                else:
                    self.trade_phase[numb] = 3
                
            else:
                self.trade_phase[numb] = 0
                
        #--<sell condition>--#
        elif min_price_1min <= s_cond_1m and min_price_5min <= s_cond_5m and         ((switch_sell == 'on' and (market_change_status == 'DOWN' or market_change_status == 'SAME')) or switch_sell == 'off'):
            if self.what_was_last_trade[numb] == 'BUY':
                self.trade_phase[numb] = 2
            else:
                self.trade_phase[numb] = 0
        else:
            self.trade_phase[numb] = 0
            
        #--< trade order >--#
        if self.trade_phase[numb] == 0:
            self.coin_Deposit_sum = (np.array(self.Total_coin[numb])*np.array(self.final_price)).sum()                
            self.Total_Deposit[numb] = (self.Total_money[numb] + self.coin_Deposit_sum)
            
        elif self.trade_phase[numb] == 1:
            #order
            self.Total_coin[numb][max_coin_idx] = (self.Total_money[numb]/self.final_price[max_coin_idx])*(1- (self.trade_penalty/100))
            self.Total_money[numb] = 0
            self.what_was_last_trade[numb] = 'BUY'
            self.what_was_last_traded_coin[numb] = max_coin_idx

            self.coin_Deposit_sum = (np.array(self.Total_coin[numb])*np.array(self.final_price)).sum()
            self.Total_Deposit[numb] = (self.Total_money[numb] + self.coin_Deposit_sum)

        elif self.trade_phase[numb] == 2:
            #order
            self.Total_money[numb] = (self.final_price[self.what_was_last_traded_coin[numb]]*self.Total_coin[numb][self.what_was_last_traded_coin[numb]])*(1- (self.trade_penalty/100))
            self.Total_coin[numb][self.what_was_last_traded_coin[numb]] = 0
            self.what_was_last_trade[numb] = 'SELL'
            
            self.coin_Deposit_sum = (np.array(self.Total_coin[numb])*np.array(self.final_price)).sum()
            self.Total_Deposit[numb] = (self.Total_money[numb] + self.coin_Deposit_sum)
        
        #코인갈아타기
        elif self.trade_phase[numb] == 3:
            #first sell
            self.Total_money[numb] = (self.final_price[self.what_was_last_traded_coin[numb]]*self.Total_coin[numb][self.what_was_last_traded_coin[numb]])*(1- ((self.trade_penalty)/100))
            self.Total_coin[numb][self.what_was_last_traded_coin[numb]] = 0
            
            #second buy
            self.Total_coin[numb][max_coin_idx] = (self.Total_money[numb]/self.final_price[max_coin_idx])*(1- ((self.trade_penalty)/100))
            self.Total_money[numb] = 0
            self.what_was_last_trade[numb] = 'BUY'
            self.what_was_last_traded_coin[numb] = max_coin_idx

            self.coin_Deposit_sum = (np.array(self.Total_coin[numb])*np.array(self.final_price)).sum()
            self.Total_Deposit[numb] = (self.Total_money[numb] + self.coin_Deposit_sum)

    #==============================================
    def dataframe_processing_1min(self, dataframe, file_num):
        dataframe = second_making_label(dataframe, file_num)
        dataframe = third_making_indicator(dataframe)
        dataframe = fourth_calc_changes(dataframe)
        dataframe = fifth_drop(dataframe)

        x_data, y_data = self.model.preprocessing(dataframe)
    
        return [x_data, y_data]

    def dataframe_processing_5min(self, dataframe_5min, file_num):
        dataframe_5min = second_making_label_5min(dataframe_5min, file_num)
        dataframe_5min = third_making_indicator_5min(dataframe_5min)
        dataframe_5min = fourth_calc_changes_5min(dataframe_5min)
        dataframe_5min = fifth_drop_5min(dataframe_5min)

        x_data, y_data = self.model_5min.preprocessing(dataframe_5min)

        return [x_data, y_data]
    
    #==============================================
    def main_loop(self, loop_num, init_phase):
        try:
            #---============================loop===========================---#
            #update folder_dir
            #1
            if init_phase == True:
                #load 6coin's .csv
                Threads_list = []
                for lis in file_dir_list:
                    Threads_list.append(lis + '{}_chart/'.format(loop_num))

                pooled = pool.map(init_read, Threads_list)    
                for idx in range(coin_count):
                    self.queue_list_1min[loop_num][idx] = pooled[idx][0]
                    self.queue_list_5min[loop_num][idx] = pooled[idx][1]
                    self.last_file_num[loop_num][idx] = pooled[idx][2]
                    self.origin_dataframe_list_5min[loop_num][idx] = pooled[idx][3]

            else:
                Threads_list = []
                for idx,lis in enumerate(file_dir_list):
                    Threads_list.append((lis + '{}_chart/'.format(loop_num),                                         self.queue_list_1min[loop_num][idx],                                         self.origin_dataframe_list_5min[loop_num][idx],                                         self.last_file_num[loop_num][idx]))

                pooled = pool.starmap(next_read, Threads_list)
                for idx in range(coin_count):
                    self.queue_list_1min[loop_num][idx] = pooled[idx][0]
                    self.queue_list_5min[loop_num][idx] = pooled[idx][1]
                    self.last_file_num[loop_num][idx] = pooled[idx][2]
                    self.origin_dataframe_list_5min[loop_num][idx] = pooled[idx][3]

            #dataframe dataset processing
            xy_data_wait_1min = pool.starmap(self.dataframe_processing_1min, zip(self.queue_list_1min[loop_num], file_num_list))
            xy_data_wait_5min = pool.starmap(self.dataframe_processing_5min, zip(self.queue_list_5min[loop_num], file_num_list))

            #slicing
            x_data_1min = []
            y_data_1min = []
            x_data_5min = []
            y_data_5min = []
            for idx in range(len(file_num_list)):
                x_data_1min.append([xy_data_wait_1min[idx][0]])
                y_data_1min.append([xy_data_wait_1min[idx][1]])
                x_data_5min.append([xy_data_wait_5min[idx][0]])
                y_data_5min.append([xy_data_wait_5min[idx][1]])

            x_data_1min = np.squeeze(np.array(x_data_1min))
            y_data_1min = np.squeeze(np.array(y_data_1min))
            x_data_5min = np.squeeze(np.array(x_data_5min))
            y_data_5min = np.squeeze(np.array(y_data_5min))

            #output prediction price
            self.out_price_1min = self.model.output_data(x_data_1min, y_data_1min)
            self.out_price_5min = self.model_5min.output_data(x_data_5min, y_data_5min)

            #calculate predict change
            self.now_predict_1m = self.out_price_1min
            self.now_predict_5m = self.out_price_5min

            if self.global_num != 0:
                self.predict_change_1m_wait = self.now_predict_1m - self.last_predict_1m
                self.predict_change_5m_wait = self.now_predict_5m - self.last_predict_5m

                self.predict_change_1m, self.predict_change_5m = self.calc_count_change(self.predict_change_1m_wait, self.predict_change_5m_wait)

            self.last_predict_1m = self.now_predict_1m
            self.last_predict_5m = self.now_predict_5m

            #set maximum price
            self.max_coin_idx = np.array(self.out_price_1min).argmax()
            self.max_price_1min = self.out_price_1min[self.max_coin_idx]
            self.max_price_5min = self.out_price_5min[self.max_coin_idx]

            #set market change status 1min
            market_change_status_1min, market_change_status_5min = self.calc_count_change(self.out_price_1min, self.out_price_5min)

            #print prediction
            if self.global_num % 10 == 0:
                print('=-----------------------------------------[{}]-----------------------------------------='.format(self.global_num))
                print('[PRD] ★MAX: {} ||[BTC]{:.2f}%,{:.2f}% [ETH]{:.2f}%,{:.2f}% [EOS]{:.2f}%,{:.2f}% [NEO]{:.2f}%,{:.2f}% [ADA]{:.2f}%,{:.2f}% [XRP]{:.2f}%,{:.2f}%'.format(c_idx_getval[self.max_coin_idx],                                                                                            self.out_price_1min[0],                                                                                            self.out_price_5min[0],                                                                                            self.out_price_1min[1],                                                                                            self.out_price_5min[1],                                                                                            self.out_price_1min[2],                                                                                            self.out_price_5min[2],                                                                                            self.out_price_1min[3],                                                                                            self.out_price_5min[3],                                                                                            self.out_price_1min[4],                                                                                            self.out_price_5min[4],                                                                                            self.out_price_1min[5],                                                                                            self.out_price_5min[5]))

            #---======================<메인 트레이드 코드>=====================---#
            #--1. set trade_phase
            #init
            self.trade_phase[0] = 0 #[0:anything, 1:buy, 2:sell]
            self.final_price = trade.get_price()

            if TRADE_PHASE == 'REAL':
                self.REAL_get_balance()

                if self.Total_money[0] >= self.coin_Deposit_sum:
                    self.what_was_last_trade[0] = 'SELL' #[BUY, SELL]
                elif self.Total_money[0] < self.coin_Deposit_sum:
                    self.what_was_last_trade[0] = 'BUY' #[BUY, SELL]

                    #0:'BTC', 1:'ETH', 2:'EOS', 3:'NEO', 4:'ADA', 5:'XRP', 6:'BNB'
                    self.what_was_last_traded_coin[0] = (np.array(self.Total_coin[0])*np.array(self.final_price)).argmax()

                #REALtrade order
                self.REAL_trade_order(0, 0.60,-0.15,0.60,0.15, -0.30,-0.15, market_change_status_5min, 'on','on','off','pass','pass', self.predict_change_5m,'on','on')
                #self.REAL_trade_order(0, 0.60,-0.30,-0.60,'pass',market_change_status_5min, 'on','on','on',0.00,'pass', self.predict_change_1m,'on','on')
                #self.REAL_trade_order(0, 0.50,'pass',-0.50,'pass',market_change_status_1min, 'off','off','on','pass',-0.10, self.predict_change_1m,'on','off') #2545
                #self.REAL_trade_order(0, 0.60,'pass','pass',-0.60,market_change_status_1min, 'off','off','on','pass',0.10, self.predict_change_5m,'off','on') #4806
                #self.REAL_trade_order(0, 'pass',0.60,'pass',-0.60,market_change_status_1min, 'on','on','on','pass',-0.10, self.predict_change_5m,'off','on') #4617
                #self.REAL_trade_order(0, 'pass',0.50,'pass',-0.50,market_change_status_1min, 'on','on','on','pass',0.06, self.predict_change_5m,'off','off') #11029
                #self.REAL_trade_order(0, 0.00,0.50,-0.50,'pass',market_change_status_5min, 'on','off','off','pass','pass', self.predict_change_5m,'on','off') #11029
                #self.REAL_trade_order(0, 0.00,0.50,-0.50,'pass',market_change_status_5min,'on','off','on','pass',0.06,self.predict_change_5m,'on','on') #9685

            else:
                pass
            #---========================<백테스트 코드>======================---#
            ##condition##
            cond_switch_list = [['on','on','off','pass','pass'],['on','off','off','pass','pass'],['off','off','off','pass','pass'],                                ['on','on','on',0.06,'pass'],['on','off','on',0.06,'pass'],['off','off','on',0.06,'pass'],                                ['on','on','on',0.10,'pass'],['on','off','on',0.10,'pass'],['off','off','on',0.10,'pass'],                                ['on','on','on',0.15,'pass'],['on','off','on',0.15,'pass'],['off','off','on',0.15,'pass'],                                ['on','on','on',0.20,'pass'],['on','off','on',0.20,'pass'],['off','off','on',0.20,'pass'],                                ['on','on','on','pass',0.06],['on','off','on','pass',0.06],['off','off','on','pass',0.06],                                ['on','on','on','pass',0.10],['on','off','on','pass',0.10],['off','off','on','pass',0.10],                                ['on','on','on','pass',0.15],['on','off','on','pass',0.15],['off','off','on','pass',0.15],                                ['on','on','on','pass',0.20],['on','off','on','pass',0.20],['off','off','on','pass',0.20],                                ['on','on','on',0.06,0.06],['on','off','on',0.06,0.06],['off','off','on',0.06,0.06],                                ['on','on','on',0.10,0.10],['on','off','on',0.10,0.10],['off','off','on',0.10,0.10],                                ['on','on','on',0.15,0.15],['on','off','on',0.15,0.15],['off','off','on',0.15,0.15],                                ['on','on','on',0.20,0.20],['on','off','on',0.20,0.20],['off','off','on',0.20,0.20]]

            for idx, cond in enumerate(cond_switch_list):
                #[1]numb, b_1m, b_5m, s_1m, s_5m, market cond, 
                #[2]buy_cond_switch, sell_cond_switch, change_coin_switch, change_stay_cond_1m, change_stay_cond_5m
                self.trade_order((idx*56)+1, 0.15, 'pass', -0.15, 'pass',market_change_status_1min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+2, 0.20, 'pass', -0.20, 'pass',market_change_status_1min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+3, 0.30, 'pass', -0.30, 'pass',market_change_status_1min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+4, 0.50, 'pass', -0.50, 'pass',market_change_status_1min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+5, 0.15, 0.00, -0.15, -0.00,market_change_status_1min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+6, 0.20, 0.00, -0.20, -0.00,market_change_status_1min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+7, 0.30, 0.00, -0.30, -0.00,market_change_status_1min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+8, 0.50, 0.00, -0.50, -0.00,market_change_status_1min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+9, 'pass', 0.15, 'pass', -0.15,market_change_status_1min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+10, 'pass', 0.20, 'pass', -0.20,market_change_status_1min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+11, 'pass', 0.30, 'pass', -0.30,market_change_status_1min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+12, 'pass', 0.50, 'pass', -0.50,market_change_status_1min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+13, 0.00, 0.15, -0.00, -0.15,market_change_status_1min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+14, 0.00, 0.20, -0.00, -0.20,market_change_status_1min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+15, 0.00, 0.30, -0.00, -0.30,market_change_status_1min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+16, 0.00, 0.50, -0.00, -0.50,market_change_status_1min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+17, 0.00, 0.30, -0.15, 'pass',market_change_status_1min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+18, 0.00, 0.30, -0.20, 'pass',market_change_status_1min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+19, 0.00, 0.30, -0.30, 'pass',market_change_status_1min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+20, 0.00, 0.30, -0.15, -0.00,market_change_status_1min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+21, 0.00, 0.30, -0.20, -0.00,market_change_status_1min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+22, 0.00, 0.30, -0.30, -0.00,market_change_status_1min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+23, 0.00, 0.50, -0.15, 'pass',market_change_status_1min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+24, 0.00, 0.50, -0.30, 'pass',market_change_status_1min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+25, 0.00, 0.50, -0.50, 'pass',market_change_status_1min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+26, 0.00, 0.50, -0.15, -0.00,market_change_status_1min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+27, 0.00, 0.50, -0.30, -0.00,market_change_status_1min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+28, 0.00, 0.50, -0.50, -0.00,market_change_status_1min,cond[0],cond[1],cond[2],cond[3],cond[4])
                #
                self.trade_order((idx*56)+29, 0.15, 'pass', -0.15, 'pass',market_change_status_5min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+30, 0.20, 'pass', -0.20, 'pass',market_change_status_5min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+31, 0.30, 'pass', -0.30, 'pass',market_change_status_5min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+32, 0.50, 'pass', -0.50, 'pass',market_change_status_5min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+33, 0.15, 0.00, -0.15, -0.00,market_change_status_5min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+34, 0.20, 0.00, -0.20, -0.00,market_change_status_5min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+35, 0.30, 0.00, -0.30, -0.00,market_change_status_5min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+36, 0.50, 0.00, -0.50, -0.00,market_change_status_5min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+37, 'pass', 0.15, 'pass', -0.15,market_change_status_5min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+38, 'pass', 0.20, 'pass', -0.20,market_change_status_5min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+39, 'pass', 0.30, 'pass', -0.30,market_change_status_5min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+40, 'pass', 0.50, 'pass', -0.50,market_change_status_5min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+41, 0.00, 0.15, -0.00, -0.15,market_change_status_5min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+42, 0.00, 0.20, -0.00, -0.20,market_change_status_5min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+43, 0.00, 0.30, -0.00, -0.30,market_change_status_5min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+44, 0.00, 0.50, -0.00, -0.50,market_change_status_5min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+45, 0.00, 0.30, -0.15, 'pass',market_change_status_5min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+46, 0.00, 0.30, -0.20, 'pass',market_change_status_5min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+47, 0.00, 0.30, -0.30, 'pass',market_change_status_5min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+48, 0.00, 0.30, -0.15, -0.00,market_change_status_5min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+49, 0.00, 0.30, -0.20, -0.00,market_change_status_5min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+50, 0.00, 0.30, -0.30, -0.00,market_change_status_5min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+51, 0.00, 0.50, -0.15, 'pass',market_change_status_5min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+52, 0.00, 0.50, -0.30, 'pass',market_change_status_5min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+53, 0.00, 0.50, -0.50, 'pass',market_change_status_5min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+54, 0.00, 0.50, -0.15, -0.00,market_change_status_5min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+55, 0.00, 0.50, -0.30, -0.00,market_change_status_5min,cond[0],cond[1],cond[2],cond[3],cond[4])
                self.trade_order((idx*56)+56, 0.00, 0.50, -0.50, -0.00,market_change_status_5min,cond[0],cond[1],cond[2],cond[3],cond[4])


            #For print, check the max idx
            traded_condition_np_list = np.array(self.Total_Deposit)
            traded_condition_max = traded_condition_np_list.max()
            traded_condition_max_idx = traded_condition_np_list.argmax()

            #*Sudden deposit increase update #갑작이 10% 이상 자산변동이 있을때 업데이트를함
            self.save_now_rate = (self.Total_Deposit[0]/self.first_Deposit*100)

            if self.save_now_rate/self.save_last_rate > 1.10:
                btc_deposit_history = trade.last_deposit_return()
                if btc_deposit_history != self.first_btc_deposit_history:
                    self.first_Deposit = self.first_Deposit + (btc_deposit_history*self.final_price[0])

            elif self.save_now_rate/self.save_last_rate < 0.90:
                btc_withdraw_history = trade.last_withdraw_return()
                if btc_withdraw_history != self.first_btc_withdraw_history:
                    self.first_Deposit = self.first_Deposit - (btc_withdraw_history*self.final_price[0])

            self.save_last_rate = self.save_now_rate

            ##print
            if self.global_num % 10 == 0:
                if TRADE_PHASE == 'REAL':
                    print('[_]Real_Trade: {:.2f}%, [BTC]:{:.2f}%, [ETH]:{:.2f}%, [EOS]:{:.2f}%, [NEO]:{:.2f}%, [ADA]:{:.2f}%, [XRP]:{:.2f}%'.format((self.Total_Deposit[0]/self.first_Deposit*100),                                                                                                                                      (self.final_price[0]/self.first_price[0]*100),                                                                                                                                      (self.final_price[1]/self.first_price[1]*100),                                                                                                                                      (self.final_price[2]/self.first_price[2]*100),                                                                                                                                      (self.final_price[3]/self.first_price[3]*100),                                                                                                                                      (self.final_price[4]/self.first_price[4]*100),                                                                                                                                      (self.final_price[5]/self.first_price[5]*100)))
                else:
                    print('[_]Origin_Rate: [BTC]:{:.2f}%, [ETH]:{:.2f}%, [EOS]:{:.2f}%, [NEO]:{:.2f}%, [ADA]:{:.2f}%, [XRP]:{:.2f}%'.format((self.final_price[0]/self.first_price[0]*100),                                                                                                                          (self.final_price[1]/self.first_price[1]*100),                                                                                                                          (self.final_price[2]/self.first_price[2]*100),                                                                                                                          (self.final_price[3]/self.first_price[3]*100),                                                                                                                          (self.final_price[4]/self.first_price[4]*100),                                                                                                                          (self.final_price[5]/self.first_price[5]*100)))            

                if self.what_was_last_trade[traded_condition_max_idx] == 'BUY':
                    print('★[MAX:{:.2f}, NUM:{}, Have_Coin:{}]'.format(traded_condition_max/100, traded_condition_max_idx, c_idx_getval[self.what_was_last_traded_coin[traded_condition_max_idx]]))

                elif self.what_was_last_trade[traded_condition_max_idx] == 'SELL':
                    print('★[MAX:{:.2f}, NUM:{}, Sold_Coin:{}]'.format(traded_condition_max/100, traded_condition_max_idx, c_idx_getval[self.what_was_last_traded_coin[traded_condition_max_idx]]))

            if self.global_num == 0:
                message = '[+]start_from_here[+]\n[+]start_from_here[+]'

                Send_Line(line_token,message)

            if self.global_num % 6 == 0:
                if self.what_was_last_trade[0] == 'BUY':
                    append_message_coin = '+'+ c_idx_getval[self.what_was_last_traded_coin[0]]
                else:
                    append_message_coin = '-'+ c_idx_getval[self.what_was_last_traded_coin[0]]

                message = '\n[*Trade]{:.2f}% [{}]\n[BTC]{:.2f} [{:.2f},{:.2f}]\n[ETH]{:.2f} [{:.2f},{:.2f}]\n[EOS]{:.2f} [{:.2f},{:.2f}]\n[NEO]{:.2f} [{:.2f},{:.2f}]\n[ADA]{:.2f} [{:.2f},{:.2f}]\n[XRP]{:.2f} [{:.2f},{:.2f}]'.format(                    (self.Total_Deposit[0]/self.first_Deposit*100),                     append_message_coin,                     (self.final_price[0]/self.first_price[0]*100),self.out_price_1min[0],self.out_price_5min[0],                     (self.final_price[1]/self.first_price[1]*100),self.out_price_1min[1],self.out_price_5min[1],                     (self.final_price[2]/self.first_price[2]*100),self.out_price_1min[2],self.out_price_5min[2],                     (self.final_price[3]/self.first_price[3]*100),self.out_price_1min[3],self.out_price_5min[3],                     (self.final_price[4]/self.first_price[4]*100),self.out_price_1min[4],self.out_price_5min[4],                     (self.final_price[5]/self.first_price[5]*100),self.out_price_1min[5],self.out_price_5min[5])

                Send_Line(line_token,message)

            #update global_num
            self.global_num += 1
            if self.error_count != 0 and self.error_reset_start == 0:
                self.error_reset_start = self.global_num
            if self.error_count != 0 and ((self.global_num - self.error_reset_start) != 0 and (self.global_num - self.error_reset_start) % 10 == 0):
                self.error_count = 0
                self.error_reset_start = 0
                
            #---=============================<loop>==========================---#
        except:
            Error_message = '\n[!]Error Happened'
            Send_Line(line_error_token, Error_message)
            print(Error_message)
            
            self.error_count += 1
            self.error_reset_start = 0
            
            if self.error_count % 5 == 0:
                time.sleep(300)
                self.error_count = 0
            else:
                time.sleep(60)

main_ = main_()


# In[10]:


#--start loop
#finding when the file added
init_phase = True
last_dir_len_list = [[0 for idx in range(6)] for idx in range(6)]
now_dir_len_list = [0 for idx in range(6)]

for idx in range(6):
    for idy, lis in enumerate(file_dir_list):
        last_dir_len_list[idx][idy] = len(os.listdir(lis+'{}_chart/'.format(idx)))
    
while True:
    for loop_num in range(6):
        while True:
            now_dir_len_list = [0 for idx in range(6)]
            for idy,lis in enumerate(file_dir_list):
                now_dir_len_list[idy] = len(os.listdir(lis+'{}_chart/'.format(loop_num)))

            #general condition
            state_count = 0
            for idy in range(6):
                if now_dir_len_list[idy] > last_dir_len_list[loop_num][idy]:
                    state_count += 1
                
            if state_count == 6:
                time.sleep(0.05)
                break

            else:
                time.sleep(0.05)

        #start
        main_.main_loop(loop_num, init_phase)
        last_dir_len_list[loop_num] = now_dir_len_list

        if loop_num == 5:
            init_phase = False
#=-----------------------------------=#
"""[*From Trader*] 
[*Trade]96.57% [+BTC]
[BTC]86.27 [-0.19,0.02]
[ETH]64.26 [-0.22,0.08]
[EOS]70.45 [-0.19,0.14]
[NEO]80.54 [-0.22,0.17]
[ADA]70.28 [0.01,0.05]
[XRP]79.22 [-0.06,-0.04]"""

