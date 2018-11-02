
# coding: utf-8

# In[1]:


import talib as ta
import pandas as pd
import numpy as np
import os
ta.get_function_groups

#Hyper parametor
span = 1
processing_file_numb = 6*6 #6*7
skip_file_numb = 0
windowing_file_len = 149
skip_process = True #[True, False]

#auto search the last number, and then save in continue.
file_dir = './' #save dir
try:
    file_list = []
    file_list = os.listdir(file_dir+'dataset/')
except:
    pass

#
if os.path.isdir(file_dir+"ckpt"):
    pass
else:
    os.makedirs(file_dir+"ckpt")
if os.path.isdir(file_dir+"dataset"):
    pass
else:
    os.makedirs(file_dir+"dataset")


# In[2]:


def windowing(dataframe):
    new_dataframe = pd.DataFrame()
    open_list = []
    high_list = []
    low_list = []
    close_list = []
    volume_list = []
    wp_list = []

    if len(dataframe) % span != 0:
        dataframe = dataframe[0:len(dataframe)-(len(dataframe)%span)]
        
    for idx in range(len(dataframe)//span):
        open_list.append(dataframe['open'][idx*span:(idx+1)*span].values[0])
        high_list.append(dataframe['high'][idx*span:(idx+1)*span].values.max())
        low_list.append(dataframe['low'][idx*span:(idx+1)*span].values.min())
        close_list.append(dataframe['close'][idx*span:(idx+1)*span].values[-1])
        volume_list.append(dataframe['volume'][idx*span:(idx+1)*span].values.sum())
        wp_list.append((dataframe['volume'][idx*span:(idx+1)*span]*dataframe['weigted_price'][idx*span:(idx+1)*span]).sum()/volume_list[idx])
        
        if idx % 1000 == 0:
            print('{}/{}'.format(idx,len(dataframe)//span))

    new_dataframe['open'] = open_list
    new_dataframe['high'] = high_list
    new_dataframe['low'] = low_list
    new_dataframe['close'] = close_list
    new_dataframe['volume'] = volume_list
    new_dataframe['weigted_price'] = wp_list

    return new_dataframe

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


# In[3]:


def first_input_time(file_name):
    dataframe = pd.read_csv("./{}.csv".format(file_name), header=0, dtype = 'float32')
    dataframe = dataframe.reset_index(drop=True)
    
    return dataframe

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
    
    #coin_name_label
    if file_num < 6*1:
        dataframe['Label_coin'] = 0 #BTC_Q
    elif file_num < 6*2:
        dataframe['Label_coin'] = 0 #BTC_B
    elif file_num < 6*3:
        dataframe['Label_coin'] = 1 #ETH
    elif file_num < 6*4:
        dataframe['Label_coin'] = 2 #EOS
    elif file_num < 6*5:
        dataframe['Label_coin'] = 3 #NEO
    elif file_num < 6*6:
        dataframe['Label_coin'] = 4 #ADA
    elif file_num < 6*7:
        dataframe['Label_coin'] = 5 #XRP
    
    return dataframe


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

def fourth_calc_changes(dataframe):
    #--calcul_changes
    apply_list = ['open','high','low','close','weigted_price','volume','SMA_3_C','SMA_5_H','SMA_5_L','SMA_5_C','SMA_10_H','SMA_10_L','SMA_10_C']
    
    for idx, lis in enumerate(apply_list):
        Changes(dataframe, lis)
    
    return dataframe

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
    dataframe = dataframe.drop([len(dataframe)-1])
    dataframe = dataframe.drop([idx for idx in range(88)])
    dataframe = dataframe.reset_index(drop=True)
    
    return dataframe


# In[4]:


for i in range(processing_file_numb):
    if i < skip_file_numb:
        continue
    dataframe = first_input_time(i)

    for idx in range(span):
        new_dataframe = dataframe[idx:-1].reset_index(drop=True)
        if span != 1:
            new_dataframe = windowing(new_dataframe)
        new_dataframe = second_making_label(new_dataframe, i)
        
        for idy in range((len(new_dataframe)-windowing_file_len)//1000):
            new_file_list = []
            for idw in range(len(file_list)):
                if 'output_{}_{}_'.format(i,idx) in file_list[idw]:
                    new_file_list.append(int(file_list[idw].split('output_{}_{}_'.format(i,idx))[1].split('.npy')[0]))
            try:
                skip_process_numb = max(new_file_list)
            except:
                skip_process_numb = 0
            
            if idy <= skip_process_numb and skip_process == True:
                continue
                
            dataset_list = []
            for idz in range(1000):
                copy_dataframe = new_dataframe[(idy*1000)+idz:(idy*1000)+windowing_file_len+idz].copy().reset_index(drop=True)
                check_ = copy_dataframe.loc[:,'open':'weigted_price'].values
                
                if 0 in check_:
                    continue
                    
                new_dataframe_2 = third_making_indicator(copy_dataframe)
                new_dataframe_2 = fourth_calc_changes(new_dataframe_2)
                new_dataframe_2 = fifth_drop(new_dataframe_2)
                dataset_list.append(new_dataframe_2.values)
                
                del(copy_dataframe)
            
            np.save(file_dir + 'dataset/output_{}_{}_{}.npy'.format(i,idx,idy), np.array(dataset_list))
            print('saved [{}/{}] ({}/{}) {}/{}'.format(i+1, processing_file_numb, idx+1, span, idy, len(new_dataframe)//1000))
            
            #for_check
            if i == skip_file_numb and idx == 0 and idy == 0:
                new_dataframe_2.to_csv(file_dir + 'dataset/0_for_check.csv', index=False)

