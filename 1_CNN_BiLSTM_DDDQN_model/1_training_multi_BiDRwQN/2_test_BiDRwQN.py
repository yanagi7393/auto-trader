
# coding: utf-8

# In[1]:


import numpy as np
import random
import tensorflow as tf
import os
import csv
import itertools
import tensorflow.contrib.layers as layers
from tensorflow.contrib.layers import flatten
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.framework import arg_scope
from helper import *
from densenet import *
from stock_state_1 import stock_state
import glob
import pandas as pd


# In[2]:


#SWITCH
    #Train
multi_count = 'auto' #['auto' or int(num)]
S_batch_norm = 'on'
S_item_shuffle = 'off' #shuffle multi item sequence. #if use shuffled dataset, please 'on'
S_multi_start_point = 'off' #shuffle multi environment start point. #if use lengths of datasets are diiferrent, please 'on'

    #loss
S_predict_price = 'on' #['on' or off]
    
    #Average_reward
S_avg_r = 'off' # please set off on a base // on, off

    # History vector Hyperparameters
S_trade_hist='on'
S_money_rate_hist='on'
S_percent_hist='on'


# In[3]:


# Percentage Hyperparameters
cond = [1.0,0,1.0] #[percentage list] #percentage to range(0～1)
len_cond = len(cond)

hist_vec_len = 0 #len_cond + 1init vector + 2history vectors
if S_trade_hist == 'on':
    hist_vec_len += len_cond
if S_money_rate_hist == 'on':
    hist_vec_len += 1
if S_percent_hist == 'on':
    hist_vec_len += 1
if hist_vec_len == 0:
    hist_vec_len = 1
    
# Hyperparameters
growth_k = 12 #filters
nb_block = 2 # how many (dense block + Transition Layer) ?
output_size = 128
input_size = 64
input_width = 8 #root(input_size)

#Setting the testing parameters
phase = 'test' # 'train' or 'test'
path = "./drqn" #The path to save/load our model to/from.
batch_size = 1 #total batch size
split_batch = 1 #how much use trace
seq_len = 60 #how much use sequence length for now action
label_len = 3
e = 0.0 #The chance of chosing a random action
max_epLength = 1440 #The max length of episode.
num_episodes = 10000 #How many episodes of game environment to train network with.
load_model = True #Whether to load a saved model.
h_size = output_size+hist_vec_len #The size of hidden layer init -> 128+5 = 133
nb_lstm_layers = 2

#for predict price calc
if S_predict_price == 'on':
    label_penalty = 1.0
elif S_predict_price == 'off':
    label_penalty = 0.0


# In[4]:


#--GPU setting
gpuConfig = tf.ConfigProto(
    device_count={'GPU':1},
    gpu_options=tf.GPUOptions(
        allow_growth = True,
        #per_process_gpu_memory_fraction=0.40,
        visible_device_list="0"))


# ### Read files

# In[5]:


############################
##=----read file list----=##
############################
file_list = glob.glob('./dataset_csv_{}/*.csv'.format(phase))
file_list.sort()

#set item_list
item_list = []
for item in file_list:
    item_list.append(item.split('\\')[1].split('_')[0])
item_list = list(set(item_list))
item_list.sort()
if multi_count == 'auto':
    multi_count = len(item_list)

#how many files are exist in each item.
train_data_count = len(file_list)//multi_count

############################
##=-read multi file list-=##
############################
if multi_count == 'auto':
    multi_file_list = [[] for idx in range(multi_count)]
    for idx in range(multi_count):
        multi_file_list[idx] = glob.glob('./dataset_csv_{}/{}*'.format(phase,item_list[idx]))
else:
    multi_file_list = [[] for idx in range(multi_count)]
    for idx in range(multi_count):
        multi_file_list[idx] = file_list[idx*train_data_count:(idx+1)*train_data_count]

#csv read
    #init
load_stock_frame = [[] for idx in range(multi_count)] #[multi_count]
processed_frame = [[] for idx in range(multi_count)] #[multi_count]
load_label_frame = [[] for idx in range(multi_count)] #[multi_count]
load_stock_len = [[] for idx in range(multi_count)] #[multi_count]

    #read    #[multi_count, train_data_count]
for idx in range(multi_count):
    for idy, file in enumerate(multi_file_list[idx]):
        load_stock_frame[idx].append(pd.read_csv(file, header=0, dtype='float32').reset_index(drop=True))
        processed_frame[idx].append(pd.read_csv('./dataset_indicator_{}\\'.format(phase) + file.split('\\')[1], header=0, dtype='float32'))
        load_label_frame[idx].append(pd.read_csv('./dataset_label_{}\\'.format(phase) + file.split('\\')[1], header=0, dtype='float32'))
        load_stock_len[idx].append(len(load_stock_frame[idx][idy]))

        print(file)


# ### Implementing the network itself

# In[6]:


class Qnetwork():
    def __init__(self,h_size,rnn_cell_fw1,rnn_cell_bw1,rnn_cell_fw2,rnn_cell_bw2,myScope):
        #prams
        self.seq_len = 60
        
        #tf.placeholder
        self.training_flag = tf.placeholder(tf.bool)
        self.batch_size = tf.placeholder(dtype=tf.int32,shape=[])
        self.scalarInput = tf.placeholder(shape=[None,input_size], dtype=tf.float32)
        self.hist_Input = tf.placeholder(shape=[None,hist_vec_len], dtype=tf.float32)
        
        self.labels = tf.placeholder(shape=[None,label_len], dtype=tf.float32) #[batch*multi_count,3]
    
        #Densnet preprocess
        self.imageIn = tf.reshape(self.scalarInput,shape=[-1,input_width,input_width,1])
        
        #densnet output -> input LSTM
        self.densnet_output = DenseNet(self.imageIn, output_size, nb_block, growth_k, self.training_flag, S_batch_norm, myScope).model_output #[batch*multi_count*seq_len,128]
        self.densnet_output = tf.concat([self.densnet_output, self.hist_Input], axis=1) #[batch*multi_count*seq_len,133]
        self.convFlat = tf.reshape(self.densnet_output,[self.batch_size*multi_count,self.seq_len,h_size]) #[batch*multi_count, seq_len, 133]
        
        #split
        self.convFlat1 = layers.fully_connected(self.convFlat, h_size, activation_fn=None)
        self.convFlat2 = layers.fully_connected(self.convFlat, h_size, activation_fn=None)
        
        #1
        (self.output_fw1, self.output_bw1), _ = tf.nn.bidirectional_dynamic_rnn(                                    inputs=self.convFlat1,cell_fw=rnn_cell_fw1,cell_bw=rnn_cell_bw1,                                    dtype=tf.float32,scope=myScope+'_rnn1')
        
        self.rnn1 = tf.concat([self.output_fw1[:,-1,:], self.output_bw1[:,-1,:]], axis=1)
        self.rnn1 = layers.fully_connected(self.rnn1, h_size, activation_fn=None) #[batch*multi_count, 133]

        #2
        (self.output_fw2, self.output_bw2), _ = tf.nn.bidirectional_dynamic_rnn(                                    inputs=self.convFlat2,cell_fw=rnn_cell_fw2,cell_bw=rnn_cell_bw2,                                    dtype=tf.float32,scope=myScope+'_rnn2')
        
        self.rnn2 = tf.concat([self.output_fw2[:,-1,:], self.output_bw2[:,-1,:]], axis=1)
        self.rnn2 = layers.fully_connected(self.rnn2, h_size, activation_fn=None) #[batch*multi_count, 133]
        
        #priceNet
        self.rnn_price = layers.fully_connected(self.rnn2, h_size, activation_fn=tf.nn.relu)
        self.rnn_price = layers.fully_connected(self.rnn_price, 3, activation_fn=None) #[batch*multi_count, 3]

        #split into separate Value and Advantage streams
        self.rnn = tf.concat([self.rnn1, self.rnn2], axis=1)
        self.rnn = layers.fully_connected(self.rnn, h_size*2, activation_fn=None)
        
        self.streamA, self.streamV = tf.split(self.rnn, 2, 1) #[batch*multi_count, 133]
        self.AW = tf.Variable(tf.random_normal([h_size, len_cond]))
        self.VW = tf.Variable(tf.random_normal([h_size, 1]))
        self.Advantage = tf.matmul(self.streamA,self.AW) #[batch*multi_count,len_cond]
        self.Value = tf.matmul(self.streamV,self.VW) #[batch*multi_count, 1]
        
        #combine final Q-values.
        self.Qout = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keep_dims=True)) #[batch*multi_count,len_cond]
        self.predict = tf.argmax(self.Qout,1) #[batch*multi_count]

        #for loss.
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32) #[batch*multi_count]
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32) #[batch*multi_count]
        self.actions_onehot = tf.one_hot(self.actions,len_cond,dtype=tf.float32) #[batch*multi_count,len_cond]
        
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1) #[batch*multi_count,len_cond]
        
        self.td_error = tf.square(self.targetQ - self.Q)
        self.lb_error = tf.reduce_mean(tf.square(self.rnn_price - self.labels))

        l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        self.loss = tf.reduce_mean(self.td_error) + (self.lb_error)*label_penalty + (l2_loss * 1e-4)
        
        self.trainer = tf.contrib.opt.NadamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)


# ### Experience Replay

# In[7]:


class experience_buffer():
    def __init__(self, buffer_size = 500):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self,experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1+len(self.buffer))-self.buffer_size] = []
        self.buffer.append(experience)
            
    def sample(self,batch_size):
        sampled_episodes = random.sample(self.buffer, batch_size//split_batch)
        
        sampledTraces = []
        for episode in sampled_episodes:
            point = np.random.randint(0,len(episode)+1-split_batch)
            sampledTraces.append(episode[point:point+split_batch])
        sampledTraces = np.array(sampledTraces)
        return np.reshape(sampledTraces,[batch_size,6])


# ### Testing the network

# In[8]:


env_list = [0]*multi_count
for idx in range(multi_count):
    env_list[idx] = stock_state(phase,                    cond,len_cond,max_epLength,                    load_stock_frame,processed_frame,load_label_frame,load_stock_len,                    S_trade_hist,S_money_rate_hist,S_percent_hist)


# In[1]:


tf.reset_default_graph()
item_states = [idx for idx in range(multi_count)]

def create_one_cell():
    with tf.variable_scope('LSTM'):
        cell = tf.contrib.rnn.LSTMCell(num_units=h_size, state_is_tuple=True, activation=tf.tanh)
        return cell

cell_fw1 = tf.contrib.rnn.MultiRNNCell([create_one_cell() for _ in range(nb_lstm_layers)], state_is_tuple=True)
cell_bw1 = tf.contrib.rnn.MultiRNNCell([create_one_cell() for _ in range(nb_lstm_layers)], state_is_tuple=True)
cell_fw2 = tf.contrib.rnn.MultiRNNCell([create_one_cell() for _ in range(nb_lstm_layers)], state_is_tuple=True)
cell_bw2 = tf.contrib.rnn.MultiRNNCell([create_one_cell() for _ in range(nb_lstm_layers)], state_is_tuple=True)
mainQN = Qnetwork(h_size,cell_fw1,cell_bw1,cell_fw2,cell_bw2,'main')

init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=5)

#create lists to contain total rewards and steps per episode
rList = []

#Make a path for our model to be saved in.
if not os.path.exists(path):
    os.makedirs(path)
    
with tf.Session(config = gpuConfig) as sess:
    if load_model == True:
        print ('Loading Model...')
        
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("[+] restored model %s" % path)
        else:
            print("[!] fail to restore model %s" % path)

    for i in range(num_episodes):
        episodeBuffer = []
        
        #Reset random item & reset start point
        item_states = shuffle_item(item_states, S_item_shuffle)
        file_state = random.randrange(0,train_data_count)
        start_point = random.randrange(seq_len, (load_stock_len[item_states[0]][file_state]-(max_epLength+1))-2)

        #Reset environment and get first new observation
        s = []
        h = []
        lb = []

        for idx, env in enumerate(env_list):
            start_point = shuffle_start_point(start_point,seq_len,((load_stock_len[item_states[idx]][file_state]-(max_epLength+1))-2),S_multi_start_point)
            
            s_, h_, lb_ = env.reset(item_states[idx], file_state, start_point) #[multi_count,seq_len,64 / multi_count,seq_len,8]
            s.append(s_)
            h.append(h_)
            lb.append(lb_)
            
        s = np.vstack(s) #[multi_count*seq_len,64]
        h = np.vstack(h) #[multi_count*seq_len,8]
        lb = np.vstack(lb) #[multi_count,4]
        lb = lb[:,1:] #[multi_count,3]
        rAll = 0
        j = 0

        #The Q-Network
        while j < max_epLength: #If the agent takes longer than 200 moves to reach either of the blocks, end the trial.
            j+=1
            
            #Choose an action by greedily (with e chance of random action) from the Q-network
            if np.random.rand(1) < e:
                a = multi_random_action(multi_count,len_cond)
            else:
                a = sess.run(mainQN.predict,                    feed_dict={mainQN.scalarInput:s, mainQN.hist_Input:h, mainQN.labels:lb,                               mainQN.batch_size:1, mainQN.training_flag:False})
                
            #Get new observation
            s1 = []
            h1 = []
            lb1 = []
            r = []
            score = []
            for idy, env in enumerate(env_list):
                s1_, h1_, lb1_, r_, score_ = env.step(a[idy]) #[multi_count,seq_len,64 / multi_count,seq_len,8]
                s1.append(s1_)
                h1.append(h1_)
                lb1.append(lb1_)
                r.append(r_)
                score.append(score_)
                
            s1 = np.vstack(s1) #[multi_count*seq_len,64]
            h1 = np.vstack(h1) #[multi_count*seq_len,8]
            lb1 = np.vstack(lb1) #[multi_count,4]
            lb1 = lb1[:,1:] #[multi_count,3]
            score = np.array(score).mean()
            
            if S_avg_r == 'on':
                r = np.array(r).mean()
                r = [r for _ in range(multi_count)]
            elif S_avg_r == 'off':
                pass
            
            episodeBuffer.append(np.reshape(np.array([s,h,a,r,s1,h1]),[1,6]))
            #print('[{}] {:.2f}, {:.2f}, {:.2f} / {:.2f}, {:.2f}, {:.2f}'.format(j,a[0],a[1],a[2],r[0],r[1],r[2]))

            rAll += np.array(r).mean()
                
            s = s1
            h = h1
            lb = lb1

        #Add the episode to the experience buffer
        bufferArray = np.array(episodeBuffer)
        rList.append(rAll)

        if i % 1 == 0:
            print('[{}/{}] Reward: {:.2f}, Score: {:.2f}'.format(i,num_episodes,rAll,score))
            
print ("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")

