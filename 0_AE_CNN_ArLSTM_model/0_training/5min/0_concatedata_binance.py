
# coding: utf-8

# In[1]:


import pandas as pd
import glob
import time
from multiprocessing.pool import ThreadPool

skip_chart_phase = False
from_here = 55700 #False
until_here = 111400 #False

#make dir list
dir_ = './0_get_ticker/' #[!] path of ticker data
file_dir_list = [dir_+'btcusdt/', dir_+'ethusdt/', dir_+'eosusdt/', dir_+'neousdt/', dir_+'adausdt/', dir_+'xrpusdt/']

#counting file_len
file_list = [0 for idx in range(len(file_dir_list))]
file_len = [0 for idx in range(len(file_dir_list))]

if until_here == False and from_here == False:
    for idx, lis in enumerate(file_dir_list):
        file_list[idx] = glob.glob(lis + '0_chart/*.csv')
        file_len[idx] = len(file_list[idx]) - 1
else:
    for idx in range(len(file_dir_list)):
        file_len[idx] = until_here - from_here
    

def concatnating_data(dir__, number, file_len):
    for i in range(6):
        if skip_chart_phase == True and i in [1,2,3,4]:
            continue
            
        dataframe = pd.DataFrame()
        set_loop_change = 0

        for idx in range(file_len):
            for idy in range(file_len):
                try:
                    dataframe = dataframe.append(pd.read_csv(dir__+'{}_chart/'.format(i)+"{}.csv".format(from_here + (idx+1-idy)), header=0, dtype='float32'))
                    break
                except:
                    print("read again")
                    time.sleep(0.05)

            if idx % 1000 == 0 and number == 0:
                print('[{}/{}] {}/{}'.format(i+1,6,idx,file_len))

        dataframe.to_csv('{}.csv'.format(i+ (number+1)*6), index=False)
    
    return 0
    
Thread_list =[]
for j, lis in enumerate(file_dir_list):
    Thread_list.append((lis,j,file_len[j]))

pool = ThreadPool(len(file_dir_list))
pooled = pool.starmap(concatnating_data, Thread_list)

