
# coding: utf-8

# In[113]:


import numpy as np
import os
import tensorflow as tf
import pandas as pd
import random
import glob

class stock_state(object):
    def __init__(self, phase,cond,len_cond,max_epLength,\
                 load_stock_frame,processed_frame,load_label_frame,load_stock_len,\
                 S_trade_hist='on',S_money_rate_hist='on',S_percent_hist='on'):
        self.calc_period = 60
        self.fee = 0.2 #%
        self.reward_threshold = 1.0 #%
        self.cond = cond #[percentage list]
        self.len_cond = len_cond
        self.train_period = max_epLength+1
        
        #Read files
        self.load_stock_frame = load_stock_frame
        self.processed_frame = processed_frame
        self.load_label_frame = load_label_frame
        self.load_stock_len = load_stock_len
        
        #Switch
        self.S_trade_hist = S_trade_hist
        self.S_money_rate_hist = S_money_rate_hist
        self.S_percent_hist = S_percent_hist
        
    def reset(self,item_state,file_state,start_point):
        #reset state & random state        
        self.start_point = start_point
        self.stock_frame = self.load_stock_frame[item_state][file_state][self.start_point:self.start_point+self.train_period].reset_index(drop=True)
        self.stock_frame_train = self.processed_frame[item_state][file_state][self.start_point-self.calc_period:self.start_point+self.train_period].reset_index(drop=True)
        self.label_frame = self.load_label_frame[item_state][file_state][self.start_point:self.start_point+self.train_period].reset_index(drop=True)

        #init parameters
        self.now_point = 0
        self.reward = 0
        self.money = 100
        self.stock = 0
        self.action_money = 100
        self.total_money = 100
        
        #init vecters
        self.trade_hist = [list(np.eye(self.len_cond, dtype=np.int8)[(self.len_cond-1)//2])]*self.calc_period #onehot [look(init)]
        self.money_rate_hist = [[1.0]]*self.calc_period
        self.percent_hist = [[0.0]]*self.calc_period
        
        #for return
        return_state = self.stock_frame_train[self.now_point:self.now_point+self.calc_period].values
        return_label = self.label_frame[self.now_point:self.now_point+1].values

        return_history = []
        if self.S_trade_hist == 'on':
            return_history.append(np.array(self.trade_hist))
        if self.S_money_rate_hist == 'on':
            return_history.append(np.array(self.money_rate_hist))
        if self.S_percent_hist == 'on':
            return_history.append(np.array(self.percent_hist))

        try:
            return_history = np.hstack(return_history)
        except:
            return_history = np.reshape(np.array([[0.0]]*self.calc_period),[-1,1])
        
        return return_state, return_history, return_label
        
    def calc_reward(self): 
        if self.total_money >= self.action_money*(1 + (self.reward_threshold/100)):
            self.reward = self.reward + 1
            self.action_money = self.total_money

        elif self.total_money <= self.action_money*(1 - (self.reward_threshold/100)):
            self.reward = self.reward - 1
            self.action_money = self.total_money

        else:
            self.reward = self.reward + 0
            
    def buy(self, percent):
        if self.money == 0:
            self.total_money = self.money + (self.stock*self.stock_frame['weigted_price'][self.now_point])
            
        else:
            self.stock = self.stock + (self.money/self.stock_frame['weigted_price'][self.now_point])*percent*(1-(self.fee/100))
            self.money = self.money*(1-percent)
            
            self.total_money = self.money + (self.stock*self.stock_frame['weigted_price'][self.now_point])
            self.buy_price = self.stock_frame['weigted_price'][self.now_point]

        #for undate vectors
        #self.len_cond -1 -> to remain buy,sell & to reject look
        for idx in range((self.len_cond-1)//2):
            if percent == self.cond[idx]:
                now_trade_hist = list(np.eye(self.len_cond, dtype=np.int8)[idx])
        
        return now_trade_hist
    
    def look(self,percent):
        self.total_money = self.money + (self.stock*self.stock_frame['weigted_price'][self.now_point])

        #for undate vectors
        now_trade_hist = list(np.eye(self.len_cond, dtype=np.int8)[(self.len_cond-1)//2])
        
        return now_trade_hist
    
    def sell(self, percent):
        if self.stock == 0:
            self.total_money = self.money + (self.stock*self.stock_frame['weigted_price'][self.now_point])

        else:
            self.money = self.money + (self.stock*self.stock_frame['weigted_price'][self.now_point])*percent*(1-(self.fee/100))
            self.stock = self.stock*(1-percent)
            
            self.total_money = self.money + (self.stock*self.stock_frame['weigted_price'][self.now_point])
            self.sell_price = self.stock_frame['weigted_price'][self.now_point]

        #for undate vectors
        for idx in range((self.len_cond-1)//2):
            if percent == self.cond[((self.len_cond-1)//2)+1+idx]:
                now_trade_hist = list(np.eye(self.len_cond, dtype=np.int8)[((self.len_cond-1)//2)+1+idx])
        
        return now_trade_hist
            
    def step(self, action):
        #init
        self.reward = 0
        
        #buy and sell the stock
        for idx in range(self.len_cond):
            if action == idx:
                percent = self.cond[idx]
                if idx < (self.len_cond-1)//2:
                    now_trade_hist = self.buy(percent)
                elif idx == (self.len_cond-1)//2:
                    now_trade_hist = self.look(percent)
                elif idx > (self.len_cond-1)//2:
                    now_trade_hist = self.sell(percent)
                
                break
                
        #calc reward
        self.calc_reward()
        
        #update vectors
        now_money_rate_hist = self.money/self.total_money        
        
        del(self.trade_hist[0])
        del(self.money_rate_hist[0])
        del(self.percent_hist[0])
        self.trade_hist.append(now_trade_hist)
        self.money_rate_hist.append([now_money_rate_hist])
        self.percent_hist.append([percent])
        
        #step next
        self.now_point += 1
        
        #for return
        return_state = self.stock_frame_train[self.now_point:self.now_point+self.calc_period].values
        return_label = self.label_frame[self.now_point:self.now_point+1].values

        return_history = []
        if self.S_trade_hist == 'on':
            return_history.append(np.array(self.trade_hist))
        if self.S_money_rate_hist == 'on':
            return_history.append(np.array(self.money_rate_hist))
        if self.S_percent_hist == 'on':
            return_history.append(np.array(self.percent_hist))
          
        #End reward
        if self.now_point == self.train_period-1:
            if self.total_money > 100:
                self.reward += self.total_money - 100
            elif self.total_money == 100:
                self.reward += -10
            elif self.total_money < 100:
                self.reward += self.total_money - 100
            
        try:
            return_history = np.hstack(return_history)
        except:
            return_history = np.reshape(np.array([[0.0]]*self.calc_period),[-1,1])
        
        return return_state, return_history, return_label, self.reward, self.total_money

