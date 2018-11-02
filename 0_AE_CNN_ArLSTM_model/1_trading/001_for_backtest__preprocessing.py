
# coding: utf-8

# In[1]:


import os
import glob
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

#
check_start = 27300 #base = 0
check_untilhere = 117700

backtest_dir = './backtest_dataset/'

#make dir
if os.path.isdir(backtest_dir):
    pass
else:
    os.makedirs(backtest_dir)

#load last file number, then change check_start
dataset_list = glob.glob(backtest_dir+'*_price.npy')
if len(dataset_list) != 0:
    for idx,lis in enumerate(dataset_list):
        dataset_list[idx] = int(lis.split('\\')[1].split('_')[0])
    check_start = max(dataset_list) + 100

#
span = 5
cond_count = 2184
coin_count = 6
loop_count = 6
save_loop = 100

coin_balance_count = coin_count +1 #because of BNB coin should be not included.
loop_num = 0

batch_size = coin_count
load_file_len = 148
COIN = ['BTCUSDT', 'ETHUSDT', 'EOSUSDT', 'NEOUSDT', 'ADAUSDT', 'XRPUSDT', 'BNBUSDT']
c_idx_getval = {0:'BTC', 1:'ETH', 2:'EOS', 3:'NEO', 4:'ADA', 5:'XRP', 6:'BNB', 7:'USDT'}
dir_ = '../0_training/0_get_ticker/'
file_dir_list = [dir_+'btcusdt/', dir_+'ethusdt/', dir_+'eosusdt/', dir_+'neousdt/', dir_+'adausdt/', dir_+'xrpusdt/'] # len == coin_count
file_num_list = [idx for idx in range(coin_count)]

main_pool = ThreadPool(loop_count)
pool = ThreadPool(coin_count)


# In[2]:


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
    
    #windowing the file
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
        self.queue_list_1min = [[0,0,0,0,0,0] for idx in range(6)]
        self.queue_list_5min = [[0,0,0,0,0,0] for idx in range(6)]
        
        #-common
        self.first_price = [0 for idx in range(len(file_dir_list))]
        for idx,lis in enumerate(file_dir_list):
            self.first_price[idx] = pd.read_csv(lis + '0_chart/{}.csv'.format(check_start), header=0, dtype='float32')['close'][0]
            
        self.folder_dir = ''
        self.trade_penalty = 0.13
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
        self.trade_phase = [0 for idx in range(cond_count+1)] #[0:nothing, 1:buy, 2:sell]
        
        #for save
        self.save_x_1m = []
        self.save_y_1m = []
        self.save_x_5m = []
        self.save_y_5m = []
        self.save_final_price = []

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
    
    def processing(self, check_start, check_untilhere):
        #load 6coin's .csv
        for idk in range(coin_count):
            Threads_list = []
            for lis in file_dir_list:
                Threads_list.append((lis + '{}_chart/'.format(idk), check_start, check_untilhere))

            pooled = pool.starmap(init_read, Threads_list)    
            for idx in range(coin_count):
                                #loop_num #coin
                self.queue_list_1min[idk][idx] = pooled[idx][0]
                self.queue_list_5min[idk][idx] = pooled[idx][1]
                
    def data_save(self, file_num):
        self.save_x_1m = np.array(self.save_x_1m).transpose(1,0,2,3)
        self.save_y_1m = np.array(self.save_y_1m).transpose(1,0,2,3)
        self.save_x_5m = np.array(self.save_x_5m).transpose(1,0,2,3)
        self.save_y_5m = np.array(self.save_y_5m).transpose(1,0,2,3)
        self.save_final_price = np.array(self.save_final_price)
        print(self.save_x_1m.shape)
        
        #np save
        np.save(backtest_dir+'{}_x_1min.npy'.format(file_num - save_loop), self.save_x_1m)
        np.save(backtest_dir+'{}_y_1min.npy'.format(file_num - save_loop), self.save_y_1m)
        np.save(backtest_dir+'{}_x_5min.npy'.format(file_num - save_loop), self.save_x_5m)
        np.save(backtest_dir+'{}_y_5min.npy'.format(file_num - save_loop), self.save_y_5m)
        np.save(backtest_dir+'{}_price.npy'.format(file_num - save_loop), self.save_final_price)
        
        print('[+]saved : {}'.format(file_num))
        
        #init
        self.save_x_1m = []
        self.save_y_1m = []
        self.save_x_5m = []
        self.save_y_5m = []
        self.save_final_price = []
    
    #==============================================
    def main_loop(self, loop_num, check_start, windowing_num, min_loop_num):
        #---============================loop===========================---#
        #update folder_dir

        feed_1min = []
        feed_5min = []
        for idx in range(coin_count):
            feed_1min.append(pd.DataFrame(self.queue_list_1min[loop_num][idx][0+windowing_num:load_file_len+windowing_num], columns=['open','high','low','close','volume','weigted_price'], dtype='float32'))
            feed_5min.append(pd.DataFrame(self.queue_list_5min[loop_num][idx][min_loop_num][0+(windowing_num//span):load_file_len+(windowing_num//span)], columns=['open','high','low','close','volume','weigted_price'], dtype='float32'))

        #dataframe dataset processing
        xy_data_wait_1min = pool.starmap(self.dataframe_processing_1min, zip(feed_1min, file_num_list))
        xy_data_wait_5min = pool.starmap(self.dataframe_processing_5min, zip(feed_5min, file_num_list))

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

        #for data return
        x_data_1min = np.squeeze(np.array(x_data_1min)) #[6,60,64]
        y_data_1min = np.squeeze(np.array(y_data_1min))
        x_data_5min = np.squeeze(np.array(x_data_5min))
        y_data_5min = np.squeeze(np.array(y_data_5min))
        return_final_price = []
        for idx in range(coin_count):
             return_final_price.append(self.queue_list_1min[loop_num][idx][load_file_len+windowing_num -1][3])
        return_final_price.append(0)
        
        
        return [x_data_1min, y_data_1min, x_data_5min, y_data_5min, return_final_price]
        #---=============================<loop>==========================---#


# In[9]:


#--start loop
#main
min_loop_num = 0
main_ = main_(check_start)
main_.processing(check_start, check_untilhere)

for idx in range(check_untilhere - check_start):
    main_thread = []
    for idy in range(loop_count):
        main_thread.append((idy, check_start, idx, min_loop_num))
    main_pooled = main_pool.starmap(main_.main_loop, main_thread)
    
    for idy in range(loop_count):
        #append for save
        main_.save_x_1m.append(main_pooled[idy][0])
        main_.save_y_1m.append(main_pooled[idy][1])
        main_.save_x_5m.append(main_pooled[idy][2])
        main_.save_y_5m.append(main_pooled[idy][3])
        main_.save_final_price.append(main_pooled[idy][4])
        
    if (idx+1) % save_loop == 0:
        main_.data_save(check_start+idx+1)
        
    #update
    min_loop_num = (min_loop_num+1) % span
    
    if (idx+1) % 10 == 0 :
        print('now : {}'.format(check_start+idx+1))

