
# coding: utf-8

# In[1]:


import requests
import pandas as pd
import datetime
import shutil
import time
import os
from binance.client import Client
import threading
client = Client('api_key', 'api_secret')

#[!]setting this[!] 
save_continue = True

#save continue setting
set_from_start_here = 1 #normal = 1, number of what u want to start to make the file.

if save_continue == True:
    try:
        file_list = os.listdir('./xrpusdt/5_chart/')
        file_list.remove('last_file_num.txt')

        for idx in range(len(file_list)):
            file_list[idx] = int(file_list[idx].split('.csv')[0])
        set_from_start_here = max(file_list)
    except:
        set_from_start_here = 1
        
if set_from_start_here == 1:
    set_time_point = str(int(time.time()) - 1000000) #server's saved price length -> 6000*100 (server time != PC, but set 1000000, 600000 < 1000000)
else:
    calc_min_time = min(int(os.path.getctime('./xrpusdt/5_chart/'+str(set_from_start_here-1)+'.csv')), int(os.path.getmtime('./xrpusdt/5_chart/'+str(set_from_start_here-1)+'.csv')))
    set_time_point = str(calc_min_time + 60)

#common
logDir = "./"
exchange = "binance"
coin = ['btcusdt', 'ethusdt', 'eosusdt', 'neousdt', 'adausdt', 'xrpusdt']
COIN = ['BTCUSDT', 'ETHUSDT', 'EOSUSDT', 'NEOUSDT', 'ADAUSDT', 'XRPUSDT']
span = 60 # sec
term_get_price = 1.2 # sec  // limit -> 10 per sec


# In[2]:


make_list = ['./adausdt','./btcusdt','./eosusdt','./ethusdt','./neousdt','./xrpusdt']
if os.path.isdir("./adausdt"):
    pass
else:
    for idx, lis in enumerate(make_list):
        os.makedirs(lis)
        for idy in range(6):
            os.makedirs(lis + '/{}_chart'.format(idy))
        os.makedirs(lis + '/0_chart/ticker')


# In[3]:


def crypto_historical_chart(exc, coin_, sec, frm, model_list):
    url = 'https://api.cryptowat.ch/markets/{}/{}/ohlc?periods={}&after={}'.format(exc, coin_, sec, frm)
    page = requests.get(url)
    data = page.json()['result'][str(span)]

    #
    candle_1m = pd.DataFrame(data, columns=['id','open','high','low','close','volume','weigted_price'])
    candle_1m = candle_1m.drop([len(candle_1m)-1]).drop(['id'], axis=1)
    candle_1m['weigted_price'] = candle_1m['weigted_price']/candle_1m['volume']

    #set start number of after crypto.
    len_of_crypto_chart = len(candle_1m)
    
    #start point update
    for idx in range(len(model_list)):
        model_list[idx].start_from = set_from_start_here + len_of_crypto_chart -1
        model_list[idx].save_step = 0 + model_list[idx].start_from*6

    #make the chart of crypto/
    for idx in range(len_of_crypto_chart):
        if candle_1m[idx:1+idx]['open'].values[0] == 0:
            for idy in range(6):
                shutil.copy('./'+ coin_ +'/{}_chart/{}.csv'.format(idy,idx + (set_from_start_here-1)), './'+ coin_ +'/{}_chart/{}.csv'.format(idy,idx+1 + (set_from_start_here-1)))
        else:
            for idy in range(6):
                candle_1m[idx:1+idx].to_csv('./'+ coin_ +'/{}_chart/{}.csv'.format(idy,idx+1 + (set_from_start_here-1)), index=False)


# In[4]:


#first
class model(threading.Thread):
    def __init__(self, coin_, COIN_):
        threading.Thread.__init__(self) 
        self.coin_ = coin_
        self.COIN_ = COIN_
        self.first_request = client.get_recent_trades(symbol='{}'.format(self.COIN_), limit=1)[0]
        self.index_num = self.first_request['id']
        self.start_time = int(str(self.first_request['time'])[:-3])
        self.start_from = 5999
        self.save_step = 0 + self.start_from*6
        self.save_phase = 0
        self.error_step = 0
        
        self.dataframe_save_0 = pd.DataFrame()
        self.dataframe_save_1 = pd.DataFrame()
        self.dataframe_save_2 = pd.DataFrame()
        self.dataframe_save_3 = pd.DataFrame()
        self.dataframe_save_4 = pd.DataFrame()
        self.dataframe_save_5 = pd.DataFrame()
        print('[+] start')
        
    def minute_price_historical(self):
        try:
            trades = client.get_recent_trades(symbol='{}'.format(self.COIN_))
            for idx in range(len(trades)):
                if trades[idx]['isBuyerMaker'] == True:
                    trades[idx]['isBuyerMaker'] = 'buy'
                else:
                    trades[idx]['isBuyerMaker'] = 'sell'
                trades[idx]['time'] = str(trades[idx]['time'])[:-3]

            df = pd.DataFrame(trades)
            df = df.drop(['isBestMatch'], axis=1)
            df = df.rename(index=str, columns={"isBuyerMaker": "taker_side", "qty": "quantity", "time": "created_at"})
        except:
            df = pd.DataFrame()
            df['created_at'] = [0]
            df['id'] = [0]
            df['price'] = [0]
            df['quantity'] = [0]
            df['taker_side'] = ['']

        return df
    
    def make_chart(self, folder_dir, forchart):
        chartframe = pd.DataFrame()
        self.file_num = self.save_step//6

        try:
            #main
            rate_wait = forchart['price'].astype(float)
            amount_wait = forchart['quantity'].astype(float)
            
            #main
            chartframe['open'] = [rate_wait.values[0]]
            chartframe['high'] = [rate_wait.max()]
            chartframe['low'] = [rate_wait.min()]
            chartframe['close'] = [rate_wait.values[-1]]
            chartframe['volume'] = [amount_wait.sum()]
            chartframe['weigted_price'] = [(rate_wait*amount_wait).sum()/chartframe['volume'][0]]
            
            for idx in range(10):
                try:
                    f = open(folder_dir + 'last_file_num.txt', 'w')
                    f.write(str(self.file_num))
                    f.close()
                    
                    chartframe.to_csv(folder_dir + '{}.csv'.format(self.file_num), index=False)
                    if folder_dir == './'+self.coin_+'/0_chart/':
                        forchart.to_csv(folder_dir + 'ticker/' + '{}.csv'.format(self.file_num), index=False)
                    
                    break
                except:
                    time.sleep(0.01)

        except:
            f = open(folder_dir + 'last_file_num.txt', 'w')
            f.write(str(self.file_num))
            f.close()
            
            shutil.copy(folder_dir + '{}.csv'.format(self.file_num-1), folder_dir + '{}.csv'.format(self.file_num))
            if folder_dir == './'+self.coin_+'/0_chart/':
                shutil.copy(folder_dir + 'ticker/' + '{}.csv'.format(self.file_num-1), folder_dir + 'ticker/' + '{}.csv'.format(self.file_num))
            
    def except_(self, folder_dir):
        f = open(folder_dir + 'last_file_num.txt', 'w')
        f.write(str(self.file_num))
        f.close()
        
        shutil.copy(folder_dir + '{}.csv'.format(self.file_num-1), folder_dir + '{}.csv'.format(self.file_num))
        if folder_dir == './'+self.coin_+'/0_chart/':
            shutil.copy(folder_dir + 'ticker/' + '{}.csv'.format(self.file_num-1), folder_dir + 'ticker/' + '{}.csv'.format(self.file_num))
        
    #_loop
    def run(self):
        print('[!]loop_start coin : {}'.format(self.coin_))
        #-- loop ---
        while True:

            #make ticker
            df = self.minute_price_historical()
            try:
                df['id'][0] == 0
                df['price'][0] == 0
            except:
                pass
            else:
                df = df.sort_values(by=["id"], ascending=True)
                df = df.reset_index(drop=True)
                if df['id'][0] != 0:
                    for idx in range(len(df)):
                        if int(df['id'][idx]) > self.index_num and float(df['price'][idx]) > 0.0:
                            if int(df['created_at'][idx]) < self.start_time+10:
                                self.dataframe_save_0 = self.dataframe_save_0.append(df[idx:idx+1])
                                self.dataframe_save_1 = self.dataframe_save_1.append(df[idx:idx+1])
                                self.dataframe_save_2 = self.dataframe_save_2.append(df[idx:idx+1])
                                self.dataframe_save_3 = self.dataframe_save_3.append(df[idx:idx+1])
                                self.dataframe_save_4 = self.dataframe_save_4.append(df[idx:idx+1])
                                self.dataframe_save_5 = self.dataframe_save_5.append(df[idx:idx+1])

                                self.index_num = int(df['id'][idx])
                                self.error_step = 0
                            else:
                                self.save_step += 1
                                self.start_time += 10
                                self.save_phase = 1
                                break
                else:
                    self.error_step += 1
                    if self.error_step == 20:
                        self.save_step += 1
                        self.start_time += 10
                        self.save_phase = 1


            #make chart
            if self.save_step > (5 + self.start_from*6) and self.save_step%6 == 0 and self.save_phase == 1:
                try:
                    self.make_chart('./'+self.coin_+'/0_chart/', self.dataframe_save_0)
                except:
                    self.except_('./'+self.coin_+'/0_chart/')
                self.dataframe_save_0 = pd.DataFrame()

            elif self.save_step > (5 + self.start_from*6) and self.save_step%6 == 1 and self.save_phase == 1:
                try:
                    self.make_chart('./'+self.coin_+'/1_chart/', self.dataframe_save_1)
                except:
                    self.except_('./'+self.coin_+'/1_chart/')
                self.dataframe_save_1 = pd.DataFrame()

            elif self.save_step > (5 + self.start_from*6) and self.save_step%6 == 2 and self.save_phase == 1:
                try:
                    self.make_chart('./'+self.coin_+'/2_chart/', self.dataframe_save_2)
                except:
                    self.except_('./'+self.coin_+'/2_chart/')
                self.dataframe_save_2 = pd.DataFrame()

            elif self.save_step > (5 + self.start_from*6) and self.save_step%6 == 3 and self.save_phase == 1:
                try:
                    self.make_chart('./'+self.coin_+'/3_chart/', self.dataframe_save_3)
                except:
                    self.except_('./'+self.coin_+'/3_chart/')
                self.dataframe_save_3 = pd.DataFrame()

            elif self.save_step > (5 + self.start_from*6) and self.save_step%6 == 4 and self.save_phase == 1:
                try:
                    self.make_chart('./'+self.coin_+'/4_chart/', self.dataframe_save_4)
                except:
                    self.except_('./'+self.coin_+'/4_chart/')
                self.dataframe_save_4 = pd.DataFrame()

            elif self.save_step > (5 + self.start_from*6) and self.save_step%6 == 5 and self.save_phase == 1:
                try:
                    self.make_chart('./'+self.coin_+'/5_chart/', self.dataframe_save_5)
                except:
                    self.except_('./'+self.coin_+'/5_chart/')
                self.dataframe_save_5 = pd.DataFrame()

            self.save_phase = 0
            time.sleep(term_get_price)
            #-----------



#+--<start loop>--+#
first_request = client.get_recent_trades(symbol='BTCUSDT', limit=1)[0]
start_time = int(str(first_request['time'])[:-3])

for idy in range(60):
    if start_time%60 == ((60 - idy)% 60):
        print('[+] wait_{} sec'.format(idy))
        time.sleep(idy)
        print('[+] start_time : {}'.format(start_time + idy))
        break

#==-<Trheading>-==#
#init
if __name__ == '__main__':
    model_list = []
    procs = []
    
    for idx in range(len(coin)):
        model_list.append(model(coin[idx], COIN[idx]))
        time.sleep(0.05)

    #getting ticker from cryptowatch
    for coin_ in coin:
        proc = threading.Thread(target=crypto_historical_chart, args=(exchange, coin_, str(span), set_time_point,model_list,))
        procs.append(proc)
        proc.start()
        
    for proc in procs:
        proc.join()
        
    #getting ticker from exchange
    for idx,lis in enumerate(model_list):
        lis.start()
        time.sleep(0.05)

