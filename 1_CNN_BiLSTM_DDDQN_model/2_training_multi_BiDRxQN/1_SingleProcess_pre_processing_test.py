
# coding: utf-8

# In[1]:


import talib as ta
import pandas as pd
import numpy as np
import os
import glob
ta.get_function_groups

#Hyper parametor
span = 5
S_stride = 'off'

processing_file_list = glob.glob('./csv_test/*')
processing_file_num = len(processing_file_list)

#item count
item_list = []
for item in processing_file_list:
    item_list.append(item.split('\\')[1].split('_')[0])
item_list = list(set(item_list))
item_list.sort()
multi_count = len(item_list)
train_data_count = len(processing_file_list)//multi_count

#item list to item dictionary
item_dict = {}
for idx,item in enumerate(item_list):
    item_dict[item] = idx

#auto search the last number, and then save in continue.
file_dir = './'
dir_list = ["csv_test","dataset_csv_test","dataset_csv_test","dataset_indicator_test","dataset_label_test"]

for _dir_ in dir_list:
    if os.path.isdir(file_dir+_dir_):
        pass
    else:
        os.makedirs(file_dir+_dir_)


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
        
        if idx == (len(dataframe)//span) - 1:
            print('[+] Calculated span')

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
            changes.append(100)
            
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


def first_input(file_name):
    dataframe = pd.read_csv(file_name, header=0, dtype = 'float32')
    dataframe = dataframe.reset_index(drop=True)
    
    return dataframe

def second_labeling(dataframe, item):
    WP_change_list = Labeling(dataframe, 'weigted_price')
    O_change_list = Labeling(dataframe, 'open')
    H_change_list = Labeling(dataframe, 'high')
    L_change_list = Labeling(dataframe, 'low')
    C_change_list = Labeling(dataframe, 'close')
    
    label_dataframe = pd.DataFrame()
    label_dataframe['number'] = [item_dict[item] for _ in range(len(WP_change_list))]
    label_dataframe['weigted_price'] = WP_change_list
    label_dataframe['high'] = H_change_list
    label_dataframe['low'] = L_change_list
    
    return label_dataframe

def third_making_indicator(dataframe):
    Open_lis = np.array(dataframe['open'], dtype='float')
    High_lis = np.array(dataframe['high'], dtype='float')
    Low_lis = np.array(dataframe['low'], dtype='float')
    Clz_lis = np.array(dataframe['weigted_price'], dtype='float')
    Vol_lis = np.array(dataframe['volume'], dtype='float')

    ##지표##
    SMA_3_H = ta.SMA(High_lis, timeperiod=3)
    SMA_3_L = ta.SMA(Low_lis, timeperiod=3)
    SMA_3_C = ta.SMA(Clz_lis, timeperiod=3)
    SMA_5_H = ta.SMA(High_lis, timeperiod=5)
    SMA_5_L = ta.SMA(Low_lis, timeperiod=5)
    SMA_5_C = ta.SMA(Clz_lis, timeperiod=5)
    SMA_7_H = ta.SMA(High_lis, timeperiod=7)
    SMA_7_L = ta.SMA(Low_lis, timeperiod=7)
    SMA_7_C = ta.SMA(Clz_lis, timeperiod=7)
    RSI_3_H = ta.RSI(High_lis, timeperiod=3)
    RSI_3_L = ta.RSI(Low_lis, timeperiod=3)
    RSI_3_C = ta.RSI(Clz_lis, timeperiod=3)
    RSI_5_H = ta.RSI(High_lis, timeperiod=5)
    RSI_5_L = ta.RSI(Low_lis, timeperiod=5)
    RSI_5_C = ta.RSI(Clz_lis, timeperiod=5)
    RSI_7_H = ta.RSI(High_lis, timeperiod=7)
    RSI_7_L = ta.RSI(Low_lis, timeperiod=7)
    RSI_7_C = ta.RSI(Clz_lis, timeperiod=7)
    ADX = ta.ADX(High_lis, Low_lis, Clz_lis, timeperiod=7)
    ADXR = ta.ADXR(High_lis, Low_lis, Clz_lis, timeperiod=7)
    ADX_2 = ta.ADX(High_lis, Low_lis, Clz_lis, timeperiod=5)
    ADXR_2 = ta.ADXR(High_lis, Low_lis, Clz_lis, timeperiod=5)
    Aroondown, Aroonup = ta.AROON(High_lis, Low_lis, timeperiod=14)
    Aroondown_2, Aroonup_2 = ta.AROON(High_lis, Low_lis, timeperiod=7)
    Aroonosc = ta.AROONOSC(High_lis, Low_lis, timeperiod=14)
    Aroonosc_2 = ta.AROONOSC(High_lis, Low_lis, timeperiod=7)
    BOP = ta.BOP(Open_lis, High_lis, Low_lis, Clz_lis)
    CMO = ta.CMO(Clz_lis, timeperiod=7)
    DX = ta.DX(High_lis, Low_lis, Clz_lis, timeperiod=7)
    MINUS_DI = ta.MINUS_DI(High_lis, Low_lis, Clz_lis, timeperiod=7)
    PLUSDI = ta.PLUS_DI(High_lis, Low_lis, Clz_lis, timeperiod=7)
    PPO = ta.PPO(Clz_lis, fastperiod=12, slowperiod=26, matype=0)
    PPO_2 = ta.PPO(Clz_lis, fastperiod=6, slowperiod=13, matype=0)
    ROCR100 = ta.ROCR100(Clz_lis, timeperiod=10)
    ROCR100_2 = ta.ROCR100(Clz_lis, timeperiod=5)
    STOCHF_f, _ = ta.STOCHF(High_lis, Low_lis, Clz_lis, fastk_period=3, fastd_period=2, fastd_matype=0)
    Fastk, _ = ta.STOCHRSI(Clz_lis, timeperiod=7, fastk_period=5, fastd_period=3, fastd_matype=0)
    TRIX = ta.TRIX(Clz_lis, timeperiod=7)
    ULTOSC = ta.ULTOSC(High_lis, Low_lis, Clz_lis, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    WILLR = ta.WILLR(High_lis, Low_lis, Clz_lis, timeperiod=14)
    WILLR_2 = ta.WILLR(High_lis, Low_lis, Clz_lis, timeperiod=7)
    NATR = ta.NATR(High_lis, Low_lis, Clz_lis, timeperiod=7)
    integer = ta.HT_TRENDMODE(Clz_lis)

    #append
    dataframe['SMA_3_H'] = SMA_3_H
    dataframe['SMA_3_L'] = SMA_3_L
    dataframe['SMA_3_C'] = SMA_3_C
    dataframe['SMA_5_H'] = SMA_5_H
    dataframe['SMA_5_L'] = SMA_5_L
    dataframe['SMA_5_C'] = SMA_5_C
    dataframe['SMA_7_H'] = SMA_7_H
    dataframe['SMA_7_L'] = SMA_7_L
    dataframe['SMA_7_C'] = SMA_7_C
    dataframe['RSI_3_H'] = (RSI_3_H / 100.) *2 - 1.
    dataframe['RSI_3_L'] = (RSI_3_L / 100.) *2 - 1.
    dataframe['RSI_3_C'] = (RSI_3_C / 100.) *2 - 1.
    dataframe['RSI_5_H'] = (RSI_5_H / 100.) *2 - 1.
    dataframe['RSI_5_L'] = (RSI_5_L / 100.) *2 - 1.
    dataframe['RSI_5_C'] = (RSI_5_C / 100.) *2 - 1.
    dataframe['RSI_7_H'] = (RSI_7_H / 100.) *2 - 1.
    dataframe['RSI_7_L'] = (RSI_7_L / 100.) *2 - 1.
    dataframe['RSI_7_C'] = (RSI_7_C / 100.) *2 - 1.
    dataframe['ADX'] = (ADX / 100.) *2 - 1.
    dataframe['ADX_2'] = (ADX_2 / 100.) *2 - 1.
    dataframe['ADXR'] = (ADXR / 100.) *2 - 1.
    dataframe['ADXR_2'] = (ADXR_2 / 100.) *2 - 1.
    dataframe['Aroondown'] = (Aroondown / 100.) *2 - 1
    dataframe['Aroondown_2'] = (Aroondown_2 / 100.) *2 - 1
    dataframe['Aroonup'] = (Aroonup / 100.) *2 - 1
    dataframe['Aroonup_2'] = (Aroonup_2 / 100.) *2 - 1
    dataframe['Aroonosc'] = Aroonosc / 100.
    dataframe['Aroonosc_2'] = Aroonosc_2 / 100.
    dataframe['BOP'] = BOP
    dataframe['CMO'] = CMO / 100.
    dataframe['DX'] = (DX / 100.) *2 - 1
    dataframe['MINUS_DI'] = (MINUS_DI / 100.) *2 - 1
    dataframe['PLUSDI'] = (PLUSDI / 100.) *2 - 1
    dataframe['PPO'] = PPO
    dataframe['PPO_2'] = PPO_2
    dataframe['ROCR100'] = ((ROCR100 / 100.) - 1.0) * 100.
    dataframe['STOCHF_f'] = (STOCHF_f / 100.) *2 - 1
    dataframe['Fastk'] = (Fastk / 100.) *2 - 1
    dataframe['TRIX'] = TRIX * 10.
    dataframe['ULTOSC'] = (ULTOSC / 100.) *2 - 1
    dataframe['WILLR'] = (WILLR / 100.) *2 + 1
    dataframe['WILLR_2'] = (WILLR_2 / 100.) *2 + 1
    dataframe['NATR'] = NATR *2 -1
    dataframe['integer'] = integer
    
    return dataframe

def fourth_calc_changes(dataframe):
    #--calcul_changes
    apply_list = ['open','high','low','close','weigted_price','volume','SMA_3_H','SMA_3_L','SMA_3_C','SMA_5_H','SMA_5_L','SMA_5_C','SMA_7_H','SMA_7_L','SMA_7_C']
    
    for idx, lis in enumerate(apply_list):
        Changes(dataframe, lis)
    
    return dataframe

def fifth_drop(dataframe):
    
    dataframe = dataframe.reset_index(drop=True)
    dataframe['open'] = ((dataframe['open'] / dataframe['weigted_price']) - 1.0) * 100.
    dataframe['high'] = ((dataframe['high'] / dataframe['weigted_price']) - 1.0) * 100.
    dataframe['low'] = ((dataframe['low'] / dataframe['weigted_price']) - 1.0) * 100.
    dataframe['close'] = ((dataframe['close'] / dataframe['weigted_price']) - 1.0) * 100.
    dataframe['SMA_3_H'] = ((dataframe['SMA_3_H'] / dataframe['weigted_price']) - 1.0) * 100.
    dataframe['SMA_3_L'] = ((dataframe['SMA_3_L'] / dataframe['weigted_price']) - 1.0) * 100.
    dataframe['SMA_3_C'] = ((dataframe['SMA_3_C'] / dataframe['weigted_price']) - 1.0) * 100.
    dataframe['SMA_5_H'] = ((dataframe['SMA_5_H'] / dataframe['weigted_price']) - 1.0) * 100.
    dataframe['SMA_5_L'] = ((dataframe['SMA_5_L'] / dataframe['weigted_price']) - 1.0) * 100.
    dataframe['SMA_5_C'] = ((dataframe['SMA_5_C'] / dataframe['weigted_price']) - 1.0) * 100.
    dataframe['SMA_7_H'] = ((dataframe['SMA_7_H'] / dataframe['weigted_price']) - 1.0) * 100.
    dataframe['SMA_7_L'] = ((dataframe['SMA_7_L'] / dataframe['weigted_price']) - 1.0) * 100.
    dataframe['SMA_7_C'] = ((dataframe['SMA_7_C'] / dataframe['weigted_price']) - 1.0) * 100.
    
    #volume_changes to -1~1
    data_list_1 = dataframe['volume_Changes'].copy().values
    data_list_2 = dataframe['volume_Changes'].copy().values
    data_list_1 = data_list_1/1000.

    for idx, lis in enumerate(data_list_1):
        if lis >= 1:
            data_list_1[idx] = 1
        elif lis <= -1:
            data_list_1[idx] = -1
            
    for idx, lis in enumerate(data_list_2):
        if lis >= -10 and lis <= 10:
            data_list_2[idx] = 0
        elif lis > 10:
            data_list_2[idx] = 1
        elif lis < -10:
            data_list_2[idx] = -1
    
    dataframe['volume_Changes'] = data_list_1
    dataframe['volume_Changes_2'] = data_list_2
    
    dataframe = dataframe.drop(['volume','weigted_price'], axis=1)
    dataframe = dataframe.drop([idx for idx in range(300)])
    dataframe = dataframe.reset_index(drop=True)
    
    return dataframe


# In[4]:


for i,lis in enumerate(processing_file_list):
    item = lis.split('\\')[1].split('_')[0]
    
    dataframe = first_input(lis)
    
    if S_stride == 'off':
        for idx in range(span):
            dataframe_2 = dataframe[idx:len(dataframe)-span+idx].reset_index(drop=True)
            
            if span != 1:
                dataframe_2 = windowing(dataframe_2)
            
            dataframe_2_save = dataframe_2.copy().drop([idx for idx in range(300)]).reset_index(drop=True)
            dataframe_2_save = dataframe_2_save.drop([len(dataframe_2_save)-1]).reset_index(drop=True)
            dataframe_2_save.to_csv(file_dir + 'dataset_csv_test/{}_{}_{}.csv'.format(item,i%train_data_count,idx), index=False)
            print('[{}] {}'.format(i, dataframe_2_save.values.shape))

            label_dataframe = second_labeling(dataframe_2,item)
            label_dataframe = label_dataframe.drop([idx for idx in range(301)]).reset_index(drop=True)
            label_dataframe.to_csv(file_dir + 'dataset_label_test/{}_{}_{}.csv'.format(item,i%train_data_count,idx), index=False)
            print('[{}] {}'.format(i, label_dataframe.values.shape))

            dataframe_2 = third_making_indicator(dataframe_2)
            dataframe_2 = fourth_calc_changes(dataframe_2)
            dataframe_2 = fifth_drop(dataframe_2)

            dataframe_2 = dataframe_2.drop([len(dataframe_2)-1]).reset_index(drop=True)
            dataframe_2.to_csv(file_dir + 'dataset_indicator_test/{}_{}_{}.csv'.format(item,i%train_data_count,idx), index=False)
            print('[{}] {}'.format(i, dataframe_2.values.shape))

    elif S_stride == 'on':
        dataframe_csv = []
        dataframe_label = []
        dataframe_indicator = []
        for idx in range(span):
            dataframe_2 = dataframe[idx:len(dataframe)-span+idx].reset_index(drop=True)
            
            if span != 1:
                dataframe_2 = windowing(dataframe_2)
                
            dataframe_csv_column = dataframe_2.columns
            dataframe_csv.append(dataframe_2.values[300:-1])

            label_dataframe = second_labeling(dataframe_2,item)
            
            dataframe_label_column = label_dataframe.columns
            dataframe_label.append(label_dataframe.values[301:])

            dataframe_2 = third_making_indicator(dataframe_2)
            dataframe_2 = fourth_calc_changes(dataframe_2)
            dataframe_2 = fifth_drop(dataframe_2)

            dataframe_indicator_column = dataframe_2.columns
            dataframe_indicator.append(dataframe_2.values[:-1])
        
        #
        dataframe_csv = np.stack(np.array(dataframe_csv),axis=1)
        dataframe_csv = np.reshape(dataframe_csv,[-1,dataframe_csv.shape[2]])
        
        dataframe_label = np.stack(np.array(dataframe_label),axis=1)
        dataframe_label = np.reshape(dataframe_label,[-1,dataframe_label.shape[2]])
        
        dataframe_indicator = np.stack(np.array(dataframe_indicator),axis=1)
        dataframe_indicator = np.reshape(dataframe_indicator,[-1,dataframe_indicator.shape[2]])
        
        dataframe_csv = pd.DataFrame(dataframe_csv, columns=dataframe_csv_column)
        dataframe_label = pd.DataFrame(dataframe_label, columns=dataframe_label_column)
        dataframe_indicator = pd.DataFrame(dataframe_indicator, columns=dataframe_indicator_column)
        
        dataframe_csv.to_csv(file_dir + 'dataset_csv_test/{}_{}_{}.csv'.format(item,i%train_data_count,0), index=False)
        print('[{}] {}'.format(i, dataframe_csv.values.shape))
        
        dataframe_label.to_csv(file_dir + 'dataset_label_test/{}_{}_{}.csv'.format(item,i%train_data_count,0), index=False)
        print('[{}] {}'.format(i, dataframe_label.values.shape))
        
        dataframe_indicator.to_csv(file_dir + 'dataset_indicator_test/{}_{}_{}.csv'.format(item,i%train_data_count,0), index=False)
        print('[{}] {}'.format(i, dataframe_indicator.values.shape))

