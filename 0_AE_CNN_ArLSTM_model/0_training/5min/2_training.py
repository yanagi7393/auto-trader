
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import glob
import os
from threading import Thread as th
from queue import Queue

#
layers = tf.contrib.layers
arg_scope = tf.contrib.framework.arg_scope


#--GPU setting
gpuConfig = tf.ConfigProto(
    device_count={'GPU':1},
    gpu_options=tf.GPUOptions(
        #per_process_gpu_memory_fraction=0.55,
        visible_device_list="0"))

#--Hyper parameter
phase = 'train' #[train, eval]
train_data_rate = 1.0 # how much train in dataset

#
model_dir = './ckpt/'
dataset_dir = './dataset/'
epoch = 1000
save_cycle = 500 / train_data_rate
print_cycle = 10

#regularizer
regularizer_switch = 'off'
regularizer_beta = 0.00001

penalty_sum_loss = 1.0
penalty_price_net = 1.0 #1.0 -> 0.1
penalty_prd = 1.0 # 0.1 -> 1.0

batch_size = 10200 #set LSTM_batch_size * 60
LSTM_batch_size = 170
load_file_len = 1000 #do not change / we saved 1000 length in 1 file
batch_stack_num = 200 #--------------< set the batch_stack_number >-

#CNN
raw_image_size = 8
image_size = raw_image_size*2
class_num = 15
coin_count = 6

#LSTM
Num_LSTM_layer = 3 #2 ~ 3
seq_length = (batch_size // LSTM_batch_size) - 1
lstm_size = 512
learning_rate = 0.0001

#
random_index = [idx for idx in range(load_file_len * batch_stack_num)]
if phase == 'train':
    random.shuffle(random_index)

#
if os.path.isdir("./ckpt"):
    pass
else:
    os.makedirs("./ckpt")
if os.path.isdir("./dataset"):
    pass
else:
    os.makedirs("./dataset")


# In[2]:


file_list = glob.glob(dataset_dir + '*.npy')

#random
file_random_index = [idx for idx in range(len(file_list))]
random.shuffle(file_random_index)

def read_data(file_numb):
    read_file = np.load(file_list[file_random_index[(file_numb*batch_stack_num) + 0]])

    x_data = np.concatenate([read_file[:,:,:4], read_file[:,:,10:]], axis=2)
    y_data = read_file[:,:,4:10]
    
    for idx in range(batch_stack_num - 1):
        read_file = np.load(file_list[file_random_index[(file_numb*batch_stack_num) + idx+1]])

        x_data_wait = np.concatenate([read_file[:,:,:4], read_file[:,:,10:]], axis=2)
        y_data_wait = read_file[:,:,4:10]
        
        x_data = np.append(x_data, x_data_wait, axis=0)
        y_data = np.append(y_data, y_data_wait, axis=0)

        if idx % 10 == 0:
            print('load [{}/{}]'.format(idx, batch_stack_num - 1))
    #
    random_index = [idx for idx in range(len(y_data))]
    if phase == 'train':
        random.shuffle(random_index)
        
    return x_data, y_data

def read_data_th(q, file_numb):
    read_file = np.load(file_list[file_random_index[(file_numb*batch_stack_num) + 0]])

    x_data = np.concatenate([read_file[:,:,:4], read_file[:,:,10:]], axis=2)
    y_data = read_file[:,:,4:10]
    
    for idx in range(batch_stack_num - 1):
        read_file = np.load(file_list[file_random_index[(file_numb*batch_stack_num) + idx+1]])

        x_data_wait = np.concatenate([read_file[:,:,:4], read_file[:,:,10:]], axis=2)
        y_data_wait = read_file[:,:,4:10]
        
        x_data = np.append(x_data, x_data_wait, axis=0)
        y_data = np.append(y_data, y_data_wait, axis=0)

        if idx % 10 == 0:
            print('load [{}/{}]'.format(idx, batch_stack_num - 1))
    #
    random_index = [idx for idx in range(len(y_data))]
    if phase == 'train':
        random.shuffle(random_index)
    
    q.put(x_data)
    q.put(y_data)


# In[3]:


#--Placeholder

#CNN
ori_inputs = tf.placeholder(tf.float32, [batch_size, image_size, image_size, 1])

ori_class_lb = tf.placeholder(tf.float32, [batch_size, class_num])
ori_price_lb = tf.placeholder(tf.float32, [batch_size])
prd_class_lb = tf.placeholder(tf.float32, [LSTM_batch_size, class_num])
prd_price_lb = tf.placeholder(tf.float32, [LSTM_batch_size])

coin_lb = tf.placeholder(tf.float32, [batch_size, coin_count])

#Dropout
keep_prob = tf.placeholder(tf.float32) #0.8
LSTM_keep_prob = tf.placeholder(tf.float32) #0.8
is_training = tf.placeholder(tf.bool)

def restore_model(saver, model_dir):
    ckpt = tf.train.get_checkpoint_state(model_dir)

    if ckpt:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("[+] restored model %s" % model_dir)
    else:
        print("[!] fail to restore model %s" % model_dir)


# In[4]:


#--Hyper parameter0
layers = tf.contrib.layers
arg_scope = tf.contrib.framework.arg_scope
batch_norm_params = {'decay': 0.999,
                   'epsilon': 0.001,
                   'is_training': is_training,
                   'scope': 'batch_norm'}
#model
def conv_encoder(inputs, scope=None, reuse=False):
    with tf.variable_scope('encoder'):
        if reuse == True:
            tf.get_variable_scope().reuse_variables()
        with arg_scope([layers.conv2d], activation_fn=tf.nn.leaky_relu,
                        kernel_size=[4, 4],
                        stride=[2, 2],
                        padding='SAME',
                        #normalizer_fn=layers.batch_norm,
                        #normalizer_params=batch_norm_params,
                        weights_initializer=layers.xavier_initializer(),
                        #weights_regularizer=layers.l2_regularizer(0.0001, scope='l2_decay')
                      ):
            net = layers.conv2d(inputs, num_outputs=32*1, normalizer_fn=None, scope = 'conv1')
            # 16x16 x 1 -> 8x8 x (32*2)
            net = layers.conv2d(net, num_outputs=32*2, scope = 'conv2')
            # -> 4x4 x (32*3)
            net = layers.conv2d(net, num_outputs=32*4, kernel_size=[3,3], stride=[1,1], scope = 'conv3')
            # -> 4x4 x (32*4)
            latent_vec = layers.conv2d(net, num_outputs=lstm_size,
                                       stride=[1,1], 
                                       padding='VALID',
                                       scope = 'latent_vec')                      
            # -> 1x1 x lstm_size
            
    return latent_vec

def conv_decoder(inputs, scope=None, reuse=False):
    with tf.variable_scope('decoder'):
        if reuse == True:
            tf.get_variable_scope().reuse_variables()
        with arg_scope([layers.conv2d_transpose], activation_fn=tf.nn.relu,
                        kernel_size=[4, 4],
                        stride=[2, 2],
                        padding='SAME',
                        #normalizer_fn=layers.batch_norm,
                        #normalizer_params=batch_norm_params,
                        weights_initializer=layers.xavier_initializer(),
                        #weights_regularizer=layers.l2_regularizer(0.0001, scope='l2_decay')
                      ):
            net = layers.conv2d_transpose(inputs, num_outputs=32 *4, padding= 'VALID', scope = 'deconv1')
            # -> 4x4 x (32*4)
            net = layers.conv2d_transpose(net, num_outputs=32 *2, kernel_size = [3,3], stride=[1,1], 
                                          scope = 'deconv2')
            # -> 4x4 x (32*3)
            net = layers.conv2d_transpose(net, num_outputs=32 *1, scope = 'deconv3')
            # -> 8x8 x (32*2)
            net = layers.conv2d_transpose(net, num_outputs=1, 
                                          activation_fn=None,
                                          normalizer_fn=None,
                                          biases_initializer=None,
                                          scope = 'deconv4')
            # -> 16x16 x 1
 
    return net

def PriceNet(inputs ,reuse=False):
    with tf.variable_scope("CNN"):
        if reuse == True:
            tf.get_variable_scope().reuse_variables()
            
        with arg_scope([layers.conv2d],
                        activation_fn=tf.nn.relu,
                        kernel_size=[1, 1],
                        stride= [1,1],
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        #weights_regularizer=layers.l2_regularizer(0.0001, scope='l2_decay')
                      ):
            net_2 = layers.dropout(inputs, is_training=is_training, keep_prob=keep_prob, scope='dropout1')
            net_2 = layers.conv2d(net_2, lstm_size+coin_count,  scope='fc2')
            net_2 = layers.dropout(net_2, is_training=is_training, keep_prob=keep_prob, scope='dropout2')
            net_2 = layers.conv2d(net_2, lstm_size+coin_count,  scope='fc3')

            class_net = tf.squeeze(layers.conv2d(net_2, class_num, activation_fn=None,scope='fc6_class')) #1: Class
            price_net =tf.squeeze(layers.conv2d(net_2, 1, activation_fn=None,scope='fc6_price')) #1: PRICE

            return class_net, price_net
        
def create_one_cell():
    with tf.variable_scope('LSTM'):
        cell = tf.contrib.rnn.LSTMCell(num_units=lstm_size+coin_count, state_is_tuple=True, activation=tf.tanh)
        
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=LSTM_keep_prob)

        return cell

def LSTM(inputs):
    with tf.variable_scope('LSTM'):
        cell = tf.contrib.rnn.MultiRNNCell([create_one_cell() for _ in range(Num_LSTM_layer)], state_is_tuple=True)

        outputs, _states = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
        Y_pred = layers.fully_connected(outputs[:, -1, :], lstm_size+coin_count, activation_fn=None)

        return Y_pred

##################
##  Draw graph  ##
##################
#=-----<AE>
encoder_out = conv_encoder(ori_inputs) # [batch,1,1,lstm_size]
decoder_out = conv_decoder(encoder_out)

#=-----<LSTM>
encoder_out_SQ = tf.squeeze(encoder_out) #[batch, lstm_size]
encoder_out_SQ = tf.concat([encoder_out_SQ, coin_lb], axis=1)

LSTM_label = tf.stack([encoder_out_SQ[(batch_size//LSTM_batch_size)*(idx+1) - 1,:] 
                       for idx in range(LSTM_batch_size)], axis=0)#seq=29, ->[batch, lstm_size]
LSTM_in = tf.stack([encoder_out_SQ[(batch_size//LSTM_batch_size)*idx:(batch_size//LSTM_batch_size)*(idx+1) - 1,:] 
                    for idx in range(LSTM_batch_size)], axis=0)#seq=29, ->[batch,29,lstm_size]

LSTM_out = LSTM(LSTM_in)#seq=29, ->[batch,lstm_size]

#=-----<LSTM predict -> AE>
predict_decorder_label = tf.stack([ori_inputs[(batch_size//LSTM_batch_size)*(idx+1) - 1,:,:,:] 
                       for idx in range(LSTM_batch_size)], axis=0)
predict_encoder_out = tf.reshape(LSTM_out, [LSTM_batch_size,1,1,-1])

#=-----<RE Autoencoder>
re_encoder_out = conv_encoder(decoder_out, reuse=True)

#=-----<input latent_vec -> output class&price>
ori_class, ori_price = PriceNet(tf.reshape(encoder_out_SQ, [batch_size,1,1,-1]))
prd_class, prd_price = PriceNet(predict_encoder_out, reuse=True)


##################
##     Loss     ##
##################
#=-----<loss>
AE_loss = tf.reduce_mean(tf.square(ori_inputs - decoder_out))
latent_AE_loss = tf.reduce_mean(tf.square(encoder_out - re_encoder_out))

LSTM_loss = tf.reduce_mean(tf.square(LSTM_out - LSTM_label))
LSTM_vec_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.nn.softmax(LSTM_label), logits=LSTM_out))

ori_class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ori_class_lb, logits=ori_class))
ori_price_loss = tf.reduce_mean(tf.square(ori_price - ori_price_lb))

prd_class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=prd_class_lb, logits=prd_class))
prd_price_loss = tf.reduce_mean(tf.square(prd_price - prd_price_lb))

#=-----<final loss>
if regularizer_switch == 'on':
    tv = tf.trainable_variables()
    regularizer_cost = tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])

    loss = ((AE_loss + latent_AE_loss + LSTM_loss + LSTM_vec_loss/2)/10. +             (((ori_class_loss)*1.0+(prd_class_loss)*penalty_prd) +              ((ori_price_loss)*1.0+(prd_price_loss)*penalty_prd)*10)*penalty_price_net)*penalty_sum_loss + regularizer_cost*regularizer_beta

else:
    loss = ((AE_loss + latent_AE_loss + LSTM_loss + LSTM_vec_loss/2)/10. +             (((ori_class_loss)*1.0+(prd_class_loss)*penalty_prd) +              ((ori_price_loss)*1.0+(prd_price_loss)*penalty_prd)*10)*penalty_price_net)*penalty_sum_loss

##################
##  set__train  ##
##################
#=-----<predict>
ori_predict = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(ori_class, 1), tf.argmax(ori_class_lb, 1)), tf.float32))
prd_predict = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prd_class, 1), tf.argmax(prd_class_lb, 1)), tf.float32))

#re_ori_predict = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(re_ori_class, 1), tf.argmax(ori_class_lb, 1)), tf.float32))
#re_prd_predict = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(re_prd_class, 1), tf.argmax(prd_class_lb, 1)), tf.float32))

#=-----<optimizer>
optimizer = tf.contrib.opt.NadamOptimizer(learning_rate)
train = optimizer.minimize(loss)

vars_ = tf.global_variables()
saver = tf.train.Saver(vars_, max_to_keep=3)

sess = tf.InteractiveSession(config = gpuConfig)
#sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

restore_model(saver, model_dir)


# In[5]:


#init
global_step = 0

if phase == 'train':
    
    for idx in range(epoch):
        for idz in range(len(file_list) // batch_stack_num):
            #file read
            if idx == 0 and idz == 0 :
                x_data, y_data = read_data(idz)
                
                q = Queue()
                p = th(target=read_data_th, args=(q,idz,))
                p.start()
            else:
                x_data = q.get()
                y_data = q.get()
                
                q.queue.clear()
                p = th(target=read_data_th, args=(q,idz,))
                p.start()
            #
            train_cycle = (len(y_data)//LSTM_batch_size) -1 #set train_cycle
            for i in range(train_cycle):
                train_condition = np.random.choice([True, False], 1, p=[train_data_rate, 1-train_data_rate])[0]
                if train_condition == True:
                    # <data batch>
                    batch_x = np.concatenate([x_data[random_index[(i*LSTM_batch_size) + idy]] for idy in range(LSTM_batch_size)], axis=0) # [480, 49]
                    batch_y = np.concatenate([y_data[random_index[(i*LSTM_batch_size) + idy]] for idy in range(LSTM_batch_size)], axis=0) # [480, 2]

                    # <input>
                    batch_ori_inputs = np.reshape(np.concatenate([batch_x for idy in range((image_size//raw_image_size)**2)],axis=1), [batch_size,image_size,image_size,1])

                    # <label>
                    batch_ori_class_lb = np.eye(class_num, dtype=np.int8)[batch_y[:,3].astype(int)] #[batch,class_num]
                    batch_ori_price_lb = batch_y[:,0] #[batch,]

                    batch_prd_class_lb = np.stack([batch_ori_class_lb[(batch_size//LSTM_batch_size)*(idy+1) -1,:] 
                                                 for idy in range(LSTM_batch_size)], axis=0) #[4, class_num]
                    batch_prd_price_lb = np.stack([batch_ori_price_lb[(batch_size//LSTM_batch_size)*(idy+1) -1] 
                                                 for idy in range(LSTM_batch_size)], axis=0) #[4, 1]
                    
                    batch_coin_lb = np.eye(coin_count, dtype=np.int8)[batch_y[:,5].astype(int)] #[batch,class_num]


                    #===============#
                    #==start train==#
                    #===============#
                    #--predict real loss
                                        #train code
                    _, = sess.run([train], feed_dict = {ori_inputs:batch_ori_inputs,
                                                                    ori_class_lb:batch_ori_class_lb,
                                                                    prd_class_lb:batch_prd_class_lb,
                                                                    ori_price_lb:batch_ori_price_lb,
                                                                    prd_price_lb:batch_prd_price_lb,
                                                                    coin_lb:batch_coin_lb,
                                                                    keep_prob:0.8,
                                                                    LSTM_keep_prob:1.0,
                                                                    is_training:True})
                    
                    sum_loss, o_class, o_price, p_class, p_price = sess.run([loss, ori_predict, ori_price_loss, 
                                prd_predict, prd_price_loss], feed_dict = {ori_inputs:batch_ori_inputs,
                                                                    ori_class_lb:batch_ori_class_lb,
                                                                    prd_class_lb:batch_prd_class_lb,
                                                                    ori_price_lb:batch_ori_price_lb,
                                                                    prd_price_lb:batch_prd_price_lb,
                                                                    coin_lb:batch_coin_lb,
                                                                    keep_prob:1.0,
                                                                    LSTM_keep_prob:1.0,
                                                                    is_training:False})
                else:
                    pass
                    
                #save
                global_step += 1
                if global_step % save_cycle == save_cycle - 1:
                    saver.save(sess, model_dir + 'model_{}.ckpt'.format(phase), global_step = global_step)
                    print('saved : {} checkpoint'.format(i))

                #print
                if global_step % print_cycle == 0 and train_condition == True and                 sum_loss <= 1000 and o_price <= 10000 and p_price <= 10000:
                    print('[train]{} :{}[{}/{}]'.format(idx,idz, i+1, train_cycle))
                    print('[+]loss: {:.4f}, O_class: {:.4f}, P_class: {:.4f}'.format(sum_loss, o_class, p_class))
                    print('[+]              O_price: {:.4f}, P_price: {:.4f}'.format(o_price, p_price))
                    
            random.shuffle(random_index)
            
            #penalty_prd update
            #penalty_prd = penalty_prd*5.0
            #penalty_price_net = penalty_price_net/5.0
            #if penalty_prd >= 1.0:
            #    penalty_prd = 1.0
            #if penalty_price_net <= 0.1:
            #    penalty_price_net = 0.1
                
        random.shuffle(file_random_index)

