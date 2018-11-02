
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

check_start = 56500 #base = 0
check_untilhere = 112200

backtest_dir = './backtest_dataset/'

span = 5
cond_count = 1344 # change how much use condition # backtest condition list
coin_count = 6
coin_balance_count = coin_count +1 #because of BNB coin should be not included.

batch_size = 600
load_file_len = 148
COIN = ['BTCUSDT', 'ETHUSDT', 'EOSUSDT', 'NEOUSDT', 'ADAUSDT', 'XRPUSDT', 'BNBUSDT']
c_idx_getval = {0:'BTC', 1:'ETH', 2:'EOS', 3:'NEO', 4:'ADA', 5:'XRP', 6:'BNB', 7:'USDT'}
dir_ = '../0_training/0_get_ticker/'
file_dir_list = [dir_+'btcusdt/', dir_+'ethusdt/', dir_+'eosusdt/', dir_+'neousdt/', dir_+'adausdt/', dir_+'xrpusdt/'] # len == coin_count
file_num_list = [idx for idx in range(coin_count)]

pool = ThreadPool(coin_count)

#for save
save_span = 100
load = True

if os.path.isfile("./save_back.txt"):
    pass
else:
    load = False


# In[ ]:


def init_read(file_dir, check_start, check_untilhere):
    set_file_length = load_file_len * span
    return_list = [0 for idx in range(span)]

    dataframe = pd.DataFrame()
    for idx in range(check_untilhere - check_start + load_file_len*5 + (span-1)):
        for idy in range(check_untilhere - check_start + load_file_len*5 + (span-1)):
            try:
                dataframe = dataframe.append(pd.read_csv(file_dir + "{}.csv".format((check_start- load_file_len*5) + 1 + idx - idy), header=0, dtype='float32'))
                break
            except:
                print('[!]data_tiling')
                time.sleep(0.05)
                
        if idx % 1000 == 0 and 'btcusdt' in file_dir:
            print('[+] load 5min [{}/{}]'.format(idx, check_untilhere - check_start + load_file_len*5 + (span-1)))
    #
    dataframe_1min = dataframe[load_file_len*4:-(span-1)].copy()
    
    #5개씩 윈도윙
    for idk in range(span):
        new_dataframe = pd.DataFrame()
        open_list = []
        high_list = []
        low_list = []
        close_list = []
        volume_list = []
        wp_list = []

        wait_dataframe = dataframe[idk:-(span-idk)].reset_index(drop=True)

        for idx in range(len(wait_dataframe)//span):
            open_list.append(wait_dataframe['open'][idx*span:(idx+1)*span].values[0])
            high_list.append(wait_dataframe['high'][idx*span:(idx+1)*span].values.max())
            low_list.append(wait_dataframe['low'][idx*span:(idx+1)*span].values.min())
            close_list.append(wait_dataframe['close'][idx*span:(idx+1)*span].values[-1])
            volume_list.append(wait_dataframe['volume'][idx*span:(idx+1)*span].values.sum())
            wp_list.append((wait_dataframe['volume'][idx*span:(idx+1)*span]*wait_dataframe['weigted_price'][idx*span:(idx+1)*span]).sum()/volume_list[idx])

        new_dataframe['open'] = open_list
        new_dataframe['high'] = high_list
        new_dataframe['low'] = low_list
        new_dataframe['close'] = close_list
        new_dataframe['volume'] = volume_list
        new_dataframe['weigted_price'] = wp_list

        return_list[idk] = new_dataframe.values
    
    return dataframe_1min.values, return_list


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


#----- main loop-----#
#build model
class main_(object):
    def __init__(self, check_start):
        #-graph
        g_1 = tf.Graph()
        sess_1 = tf.InteractiveSession(graph=g_1)  
        self.model = AE_LSTM(sess_1, batch_size)
        g_2 = tf.Graph()
        sess_2 = tf.InteractiveSession(graph=g_2) 
        self.model_5min = AE_LSTM_5min(sess_2, batch_size)
        
        #-for dataframe
        self.output_1m = [0 for idx in range(coin_count)]
        self.output_5m = [0 for idx in range(coin_count)]
        self.predict_change_1m = 'UP'
        self.predict_change_5m = 'UP'
        
        #-common
        self.first_price = [0 for idx in range(len(file_dir_list))]
        for idx,lis in enumerate(file_dir_list):
            self.first_price[idx] = pd.read_csv(lis + '0_chart/{}.csv'.format(check_start), header=0, dtype='float32')['close'][0]
            
        self.folder_dir = ''
        self.trade_penalty = 0.20
        self.global_num = 0
                
        ##-- for backtest
        #init
        self.what_was_last_trade = ['BUY' for idx in range(cond_count+1)] #[BUY, SELL]
        self.what_was_last_traded_coin = [0 for idx in range(cond_count+1)]
        self.Total_money = [0 for idx in range(cond_count+1)] # yen
        self.Total_coin = [[0 for i in range(coin_balance_count)] for idx in range(cond_count+1)] #[condition,6]
        for idx in range(cond_count+1):
            self.Total_coin[idx][0] = (10000/self.first_price[0])
        self.Total_Deposit = [10000 for idx in range(cond_count+1)]
        
        self.Total_Deposit[0] = 0
        self.Total_coin[0][0] = 0
        
        self.trade_phase = [0 for idx in range(cond_count+1)] #[0:nothing, 1:buy, 2:sell]
        
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
    
        #==============================================
    def trade_order(self,numb,b_cond_1m,b_cond_5m,s_cond_1m,s_cond_5m,market_change_status,switch_buy,switch_sell,                    switch_change_coin,b_stay_cond_1m,b_stay_cond_5m,prd_state,switch_prd_buy,switch_prd_sell,                    switch_1m_force,switch_5m_force,force_change, force_thresh):
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
            max_idx = np.array(save_price_5min).argmax()
            
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
        if max_price_1min >= b_cond_1m and max_price_5min >= b_cond_5m and         ((switch_buy == 'on' and (market_change_status == 'UP')) or switch_buy == 'off') and         ((switch_prd_buy == 'on' and (prd_state == 'UP')) or switch_prd_buy == 'off'):
            if self.what_was_last_trade[numb] == 'SELL':
                self.trade_phase[numb] = 1
            
            #★BNB coin 경유/ 코인 갈아타기 조건
            elif self.what_was_last_trade[numb] == 'BUY' and max_coin_idx != self.what_was_last_traded_coin[numb] and             switch_change_coin == 'on':
                if (self.out_price_1min[self.what_was_last_traded_coin[numb]] >= b_stay_cond_1m and                 self.out_price_5min[self.what_was_last_traded_coin[numb]] >= b_stay_cond_5m):
                    self.trade_phase[numb] = 0
                else:
                    self.trade_phase[numb] = 3
                
            else:
                self.trade_phase[numb] = 0
                
        #--<sell condition>--#
        elif min_price_1min <= s_cond_1m and min_price_5min <= s_cond_5m and         ((switch_sell == 'on' and (market_change_status == 'DOWN')) or switch_sell == 'off') and         ((switch_prd_sell == 'on' and (prd_state == 'DOWN')) or switch_prd_sell == 'off'):
            if self.what_was_last_trade[numb] == 'BUY':
                self.trade_phase[numb] = 2
            else:
                self.trade_phase[numb] = 0
        else:
            self.trade_phase[numb] = 0
         
        ###############
        ##force trade##
        ###############
        if switch_1m_force == 'on' and self.max_price_1min_F >= force_thresh:
            if force_change == 'on' and switch_change_coin == 'on' and self.what_was_last_trade[numb] == 'BUY' and             self.max_coin_1m_idx != self.what_was_last_traded_coin[numb]:
                self.trade_phase[numb] = 3
            else:
                self.trade_phase[numb] = 0
            max_coin_idx = self.max_coin_1m_idx
            
        elif switch_5m_force =='on' and self.max_price_5min_F >= force_thresh:
            if force_change == 'on' and switch_change_coin == 'on' and self.what_was_last_trade[numb] == 'BUY' and             self.max_coin_5m_idx != self.what_was_last_traded_coin[numb]:
                self.trade_phase[numb] = 3
            else:
                self.trade_phase[numb] = 0
            max_coin_idx = self.max_coin_5m_idx
            
        elif (switch_1m_force == 'on' and min_price_1min <= -1*force_thresh) or (switch_5m_force =='on' and min_price_5min <= -1*force_thresh):
            self.trade_phase[numb] = 2
            
        #####################    
        #--< trade order >--#
        #####################
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
        
        #bnb 경유
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
    def main_loop(self, check_start, windowing_num):
        #---============================loop===========================---#
        #load
        load_x_1m = np.load(backtest_dir+'{}_x_1min.npy'.format(check_start+windowing_num))
        load_y_1m = np.load(backtest_dir+'{}_y_1min.npy'.format(check_start+windowing_num))
        load_x_5m = np.load(backtest_dir+'{}_x_5min.npy'.format(check_start+windowing_num))
        load_y_5m = np.load(backtest_dir+'{}_y_5min.npy'.format(check_start+windowing_num))
        
        load_final_price =np.load(backtest_dir+'{}_price.npy'.format(check_start+windowing_num))
        
        for idx in range(coin_count):
            self.output_1m[idx] = self.model.output_data(load_x_1m[idx], load_y_1m[idx]) #[6,600]
            self.output_5m[idx] = self.model_5min.output_data(load_x_5m[idx], load_y_5m[idx]) #[6,600]
        
        self.output_1m = np.array(self.output_1m)
        self.output_5m = np.array(self.output_5m)
        self.output_1m = self.output_1m.transpose(1,0) #[600,6]
        self.output_5m = self.output_5m.transpose(1,0) #[600,6]
        
        for idb in range(batch_size):
            self.final_price = load_final_price[idb]
            
            #output prediction price
            self.out_price_1min = self.output_1m[idb]
            self.out_price_5min = self.output_5m[idb]
        
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
            self.max_coin_idx = np.array(self.out_price_5min).argmax()
            self.max_price_1min = self.out_price_1min[self.max_coin_idx]
            self.max_price_5min = self.out_price_5min[self.max_coin_idx]
            
            self.max_coin_1m_idx = np.array(self.out_price_1min).argmax()
            self.max_coin_5m_idx = np.array(self.out_price_5min).argmax()
            self.max_price_1min_F = self.out_price_1min[self.max_coin_1m_idx]
            self.max_price_5min_F = self.out_price_5min[self.max_coin_5m_idx]

            #set market change status 1min
            market_change_status_1min, market_change_status_5min = self.calc_count_change(self.out_price_1min, self.out_price_5min)

            #print prediction
            if self.global_num % 10 == 0:
                print('=-----------------------------------------[{}]-----------------------------------------='.format(self.global_num))
                print('[PRD] ★MAX: {} ||[BTC]{:.2f}%,{:.2f}% [ETH]{:.2f}%,{:.2f}% [EOS]{:.2f}%,{:.2f}% [NEO]{:.2f}%,{:.2f}% [ADA]{:.2f}%,{:.2f}% [XRP]{:.2f}%,{:.2f}%'.format(c_idx_getval[self.max_coin_idx],                                                                                            self.out_price_1min[0],                                                                                            self.out_price_5min[0],                                                                                            self.out_price_1min[1],                                                                                            self.out_price_5min[1],                                                                                            self.out_price_1min[2],                                                                                            self.out_price_5min[2],                                                                                            self.out_price_1min[3],                                                                                            self.out_price_5min[3],                                                                                            self.out_price_1min[4],                                                                                            self.out_price_5min[4],                                                                                            self.out_price_1min[5],                                                                                            self.out_price_5min[5]))

            #---========================<백테스트 코드>======================---#
            ##condition##
            cond_switch_list_wait = [['on','on','off','pass','pass'],['off','on','off','pass','pass'],['on','off','off','pass','pass'],['off','off','off','pass','pass'],                                     ['on','on','on',-0.00,'pass'],['off','on','on',-0.00,'pass'],['on','off','on',-0.00,'pass'],['off','off','on',-0.00,'pass'],                                     ['on','on','on',-0.15,'pass'],['off','on','on',-0.15,'pass'],['on','off','on',-0.15,'pass'],['off','off','on',-0.15,'pass'],                                     ['on','on','on','pass',-0.00],['off','on','on','pass',-0.00],['on','off','on','pass',-0.00],['off','off','on','pass',-0.00],                                     ['on','on','on','pass',-0.15],['off','on','on','pass',-0.15],['on','off','on','pass',-0.15],['off','off','on','pass',-0.15],                                     ['on','on','on','pass','pass'],['off','on','on','pass','pass'],['on','off','on','pass','pass'],['off','off','on','pass','pass']]

            append_switch = [[self.predict_change_1m,'off','off'],                             [self.predict_change_1m,'on','on'],[self.predict_change_1m,'on','off'],[self.predict_change_1m,'off','on'],                             [self.predict_change_5m,'on','on'],[self.predict_change_5m,'on','off'],[self.predict_change_5m,'off','on']]
            
            cond_switch_list = []
            for append_lis in append_switch:
                for cond_lis in cond_switch_list_wait:
                    cond_switch_list.append(cond_lis + append_lis)
                
            for idx, cond in enumerate(cond_switch_list):
                #[1]numb, b_1m, b_5m, s_1m, s_5m, market cond, 
                #[2]buy_cond_switch, sell_cond_switch, change_coin_switch, change_stay_cond_1m, change_stay_cond_5m
                #[3]prd_state,switch_prd_buy,switch_prd_sell

                self.trade_order((idx*8)+1, 0.60,0.00,-0.60,0.00,market_change_status_1min,cond[0],cond[1],cond[2],cond[3],cond[4],cond[5],cond[6],cond[7],'on','on', 'on',0.8)
                self.trade_order((idx*8)+2, 0.60,0.00,-0.60,0.00,market_change_status_1min,cond[0],cond[1],cond[2],cond[3],cond[4],cond[5],cond[6],cond[7],'on','off', 'on',0.8)
                self.trade_order((idx*8)+3, 0.60,0.00,-0.60,0.00,market_change_status_1min,cond[0],cond[1],cond[2],cond[3],cond[4],cond[5],cond[6],cond[7],'off','on', 'on',0.8)
                self.trade_order((idx*8)+4, 0.60,0.00,-0.60,0.00,market_change_status_1min,cond[0],cond[1],cond[2],cond[3],cond[4],cond[5],cond[6],cond[7],'off','off', 'on',0.8)
                
                self.trade_order((idx*8)+5, 0.60,0.00,-0.60,0.00,market_change_status_1min,cond[0],cond[1],cond[2],cond[3],cond[4],cond[5],cond[6],cond[7],'on','on', 'off',0.8)
                self.trade_order((idx*8)+6, 0.60,0.00,-0.60,0.00,market_change_status_1min,cond[0],cond[1],cond[2],cond[3],cond[4],cond[5],cond[6],cond[7],'on','off', 'off',0.8)
                self.trade_order((idx*8)+7, 0.60,0.00,-0.60,0.00,market_change_status_1min,cond[0],cond[1],cond[2],cond[3],cond[4],cond[5],cond[6],cond[7],'off','on', 'off',0.8)
                self.trade_order((idx*8)+8, 0.60,0.00,-0.60,0.00,market_change_status_1min,cond[0],cond[1],cond[2],cond[3],cond[4],cond[5],cond[6],cond[7],'off','off', 'off',0.8)
                
            #For print, check the max idx
            traded_condition_np_list = np.array(self.Total_Deposit)
            traded_condition_max = traded_condition_np_list.max()
            traded_condition_max_idx = traded_condition_np_list.argmax()

            ##print
            if self.global_num % 10 == 0:

                print('[_]Origin_Rate: [BTC]:{:.2f}%, [ETH]:{:.2f}%, [EOS]:{:.2f}%, [NEO]:{:.2f}%, [ADA]:{:.2f}%, [XRP]:{:.2f}%'.format((self.final_price[0]/self.first_price[0]*100),                                                                                                                      (self.final_price[1]/self.first_price[1]*100),                                                                                                                      (self.final_price[2]/self.first_price[2]*100),                                                                                                                      (self.final_price[3]/self.first_price[3]*100),                                                                                                                      (self.final_price[4]/self.first_price[4]*100),                                                                                                                      (self.final_price[5]/self.first_price[5]*100)))            

                if self.what_was_last_trade[traded_condition_max_idx] == 'BUY':
                    print('★[MAX:{:.2f}, NUM:{}, Have_Coin:{}]'.format(traded_condition_max/100, traded_condition_max_idx, c_idx_getval[self.what_was_last_traded_coin[traded_condition_max_idx]]))

                elif self.what_was_last_trade[traded_condition_max_idx] == 'SELL':
                    print('★[MAX:{:.2f}, NUM:{}, Sold_Coin:{}]'.format(traded_condition_max/100, traded_condition_max_idx, c_idx_getval[self.what_was_last_traded_coin[traded_condition_max_idx]]))

            self.global_num += 1
            #---=============================<loop>==========================---#


# In[ ]:


def save_point(save_list):
    for idx in range(len(save_list)):
        save_list[idx] = str(save_list[idx])
    
    f = open('./save_back.txt', 'w')
    f.write('\n'.join(save_list))
    f.close()
    
def load_point():
    f = open('./save_back.txt', 'r')
    load_list = f.read().splitlines()
    f.close()
    
    return load_list

def load_init(main_, load_list):
    #price load
    load_list[1] = load_list[1].split('[')[1].split(']')[0].split(', ')
    for idx in range(coin_count):
        main_.first_price[idx] = float(load_list[1][idx])
    
    #1
    for idx in range(cond_count):        
        main_.what_was_last_trade[idx+1] = str(load_list[(idx*5)+2])
        main_.what_was_last_traded_coin[idx+1] = int(load_list[(idx*5)+3])
        main_.Total_money[idx+1] = float(load_list[(idx*5)+4])
        
        load_list[(idx*5)+5] = load_list[(idx*5)+5].split('[')[1].split(']')[0].split(', ')
        for idy in range(coin_count):
            main_.Total_coin[idx+1][idy] = float(load_list[(idx*5)+5][idy])
        main_.Total_Deposit[idx+1] = float(load_list[(idx*5)+6])


# In[ ]:


#--start loop
#load
if load == True:
    load_list = load_point()
    check_start = int(load_list[0])
    print('[+]loaded save_point[+]')
    print('From_start_here : {}'.format(check_start))

else:
    print('[!]loaded failed save_point[!]')
    
#main
main_ = main_(check_start)

#load_init
if load == True:
    load_init(main_, load_list)
    print('[+]loaded save_point[+]')

else:
    print('[!]loaded failed save_point[!]')

for idx in range(check_untilhere - check_start):
    if idx % 100 == 0:
        main_.main_loop(check_start, idx)
        
        main_.output_1m = [0 for idx in range(coin_count)]
        main_.output_5m = [0 for idx in range(coin_count)]
        
    if idx % save_span == 0:
        save_list = [check_start+(idx//100)*100 + 100,main_.first_price]
        for idx in range(cond_count):
            save_list.append(main_.what_was_last_trade[idx+1])
            save_list.append(main_.what_was_last_traded_coin[idx+1])
            save_list.append(main_.Total_money[idx+1])
            save_list.append(main_.Total_coin[idx+1])
            save_list.append(main_.Total_Deposit[idx+1])
        
        save_point(save_list)

