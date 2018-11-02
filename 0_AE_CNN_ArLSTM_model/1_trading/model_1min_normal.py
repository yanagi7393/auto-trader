import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import random

class AE_LSTM(object):
    def __init__(self, sess, batch_size):
        #--
        self.layers = tf.contrib.layers
        self.arg_scope = tf.contrib.framework.arg_scope
        
        #--Hyper parameter
        self.phase = 'output' #[train, output]
        self.model_dir = '../0_training/1min/ckpt/'
        self.epoch = 1000
        self.save_cycle = 60
        self.print_cycle = 10
        self.learning_rate = 0.0001
        self.penalty = 1
        
        self.global_step = 0
        self.random_index = []

        #CNN
        self.image_size = 16
        self.raw_image_size = 8
        self.class_num = 15
        self.coin_count = 6
        
        #condition
        if self.phase == 'train':
            self.csv_dir = 'output.csv'
            self.batch_size = 6000
            self.LSTM_batch_size = 100
        elif self.phase == 'output':
            self.csv_dir = ''
            self.batch_size = batch_size * 60
            self.LSTM_batch_size = batch_size
            
        #penalty
        self.penalty_sum_loss = 1.0
        self.penalty_price_net = 1.0 #1.0 -> 0.1
        self.penalty_prd = 1.0 # 0.1 -> 1.0

        #LSTM
        self.Num_LSTM_layer = 3 #2 ~ 3
        self.seq_length = (self.batch_size // self.LSTM_batch_size) - 1
        self.lstm_size = 512
        self.sess = sess
        self.build_model()
    
    def restore_model(self, saver):
        ckpt = tf.train.get_checkpoint_state(self.model_dir)

        if ckpt:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("[+] restored model %s" % self.model_dir)
        else:
            print("[!] fail to restore model %s" % self.model_dir)

    def preprocessing(self, dataframe):
        #--data presessing
        #-60 sequence processing
        x_data = []
        y_data = []
        
        label_lists = dataframe[['Label_WP','Label_H','Label_L','Label_class','Label_class_2','Label_coin']].values
        new_frame = dataframe.drop(['Label_WP','Label_H','Label_L','Label_class','Label_class_2','Label_coin'],\
                                   axis = 1).values

        #condition
        data_len = len(new_frame) - (self.batch_size // self.LSTM_batch_size) + 1
        for idx in range(data_len):
            x_data.append(new_frame[idx:idx + (self.batch_size // self.LSTM_batch_size)])
            y_data.append(label_lists[idx:idx + (self.batch_size // self.LSTM_batch_size)])
                
        #condition        
        if self.phase == 'train':
            self.random_index = [idx for idx in range(len(y_data))]
            random.shuffle(self.random_index)

        return x_data, y_data

    #--= model =--#
    def conv_encoder(self, inputs, scope=None, reuse=False):
        with tf.variable_scope('encoder'):
            if reuse == True:
                tf.get_variable_scope().reuse_variables()
                
            with self.arg_scope([self.layers.conv2d], activation_fn=tf.nn.leaky_relu,
                            kernel_size=[4, 4],
                            stride=[2, 2],
                            padding='SAME',
                            #normalizer_fn=self.layers.batch_norm,
                            #normalizer_params=self.batch_norm_params,
                            weights_initializer=self.layers.xavier_initializer(),
                            #weights_regularizer=self.layers.l2_regularizer(0.0001, scope='l2_decay')
                          ):
                net = self.layers.conv2d(inputs, num_outputs=32*1, normalizer_fn=None, scope = 'conv1')
                # 16x16 x 1 -> 8x8 x (32*1)
                net = self.layers.conv2d(net, num_outputs=32*2, scope = 'conv2')
                # -> 4x4 x (32*2)
                net = self.layers.conv2d(net, num_outputs=32*4, kernel_size=[3,3], stride=[1,1], scope = 'conv3')
                # -> 4x4 x (32*4)
                latent_vec = self.layers.conv2d(net, num_outputs=self.lstm_size,
                                           stride=[1,1], 
                                           padding='VALID',
                                           scope = 'latent_vec')                           
                # -> 1x1 x lstm_size

                return latent_vec

    def conv_decoder(self, inputs, scope=None, reuse=False):
        with tf.variable_scope('decoder'):
            if reuse == True:
                tf.get_variable_scope().reuse_variables()
                
            with self.arg_scope([self.layers.conv2d_transpose], activation_fn=tf.nn.relu,
                            kernel_size=[4, 4],
                            stride=[2, 2],
                            padding='SAME',
                            #normalizer_fn=self.layers.batch_norm,
                            #normalizer_params=self.batch_norm_params,
                            weights_initializer=self.layers.xavier_initializer(),
                            #weights_regularizer=self.layers.l2_regularizer(0.0001, scope='l2_decay')
                          ):
                net = self.layers.conv2d_transpose(inputs, num_outputs=32 *4, padding= 'VALID', scope = 'deconv1')
                # -> 4x4 x (32*4)
                net = self.layers.conv2d_transpose(net, num_outputs=32 *2, kernel_size = [3,3], stride=[1,1],
                                              scope = 'deconv2')
                # -> 4x4 x (32*2)
                net = self.layers.conv2d_transpose(net, num_outputs=32 *1, scope = 'deconv3')
                # -> 8x8 x (32*1)
                net = self.layers.conv2d_transpose(net, num_outputs=1, 
                                              activation_fn=None,
                                              normalizer_fn=None,
                                              biases_initializer=None,
                                              scope = 'deconv4')
                # -> 16x16 x 1

                return net

    def PriceNet(self, inputs ,reuse=False):
        with tf.variable_scope("CNN"):
            if reuse == True:
                tf.get_variable_scope().reuse_variables()

            with self.arg_scope([self.layers.conv2d],
                            activation_fn=tf.nn.relu,
                            kernel_size=[1, 1],
                            stride= [1,1],
                            weights_initializer=self.layers.xavier_initializer(),
                            #weights_regularizer=self.layers.l2_regularizer(0.0001, scope='l2_decay')
                          ):
                net_2 = self.layers.dropout(inputs, is_training=self.is_training, keep_prob=self.keep_prob, scope='dropout1')
                net_2 = self.layers.conv2d(net_2, self.lstm_size+self.coin_count,  scope='fc2')
                net_2 = self.layers.dropout(net_2, is_training=self.is_training, keep_prob=self.keep_prob, scope='dropout2')
                net_2 = self.layers.conv2d(net_2, self.lstm_size+self.coin_count,  scope='fc3')

                class_net = tf.squeeze(self.layers.conv2d(net_2, self.class_num, activation_fn=None, scope='fc6_class')) #1: Class
                price_net =tf.squeeze(self.layers.conv2d(net_2, 1, activation_fn=None, scope='fc6_price')) #1: PRICE

                return class_net, price_net

    def create_one_cell(self):
        with tf.variable_scope('LSTM'):
            cell = tf.contrib.rnn.LSTMCell(num_units=self.lstm_size+self.coin_count, state_is_tuple=True, activation=tf.tanh)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.LSTM_keep_prob)

            return cell

    def LSTM(self, inputs, reuse=False):
        with tf.variable_scope('LSTM'):
            if reuse == True:
                tf.get_variable_scope().reuse_variables()

            cell = tf.contrib.rnn.MultiRNNCell([self.create_one_cell() for _ in range(self.Num_LSTM_layer)], state_is_tuple=True)

            outputs, _states = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
            Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1, :], self.lstm_size+self.coin_count, activation_fn=None)  # We use the last cell's output

            return Y_pred

    def build_model(self):
        #--Placeholder
        #CNN
        self.ori_inputs = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, 1])
        self.ori_class_lb = tf.placeholder(tf.float32, [self.batch_size, self.class_num])
        self.ori_price_lb = tf.placeholder(tf.float32, [self.batch_size])
        self.coin_lb = tf.placeholder(tf.float32, [self.batch_size, self.coin_count])
        
        #LSTM
        self.prd_class_lb = tf.placeholder(tf.float32, [self.LSTM_batch_size, self.class_num])
        self.prd_price_lb = tf.placeholder(tf.float32, [self.LSTM_batch_size])

        #Dropout
        self.keep_prob = tf.placeholder(tf.float32) #0.8
        self.LSTM_keep_prob = tf.placeholder(tf.float32) #0.8
        self.is_training = tf.placeholder(tf.bool)

        ##################
        ##  Draw graph  ##
        ##################
        #=-----<AE>
        self.encoder_out = self.conv_encoder(self.ori_inputs) # [batch,1,1,lstm_size]
        self.decoder_out = self.conv_decoder(self.encoder_out)

        #=-----<LSTM>
        self.encoder_out_SQ = tf.squeeze(self.encoder_out)
        self.encoder_out_SQ = tf.concat([self.encoder_out_SQ, self.coin_lb], axis=1)

        self.LSTM_label = tf.stack([self.encoder_out_SQ[(self.batch_size//self.LSTM_batch_size)*(idx+1) - 1,:] 
                               for idx in range(self.LSTM_batch_size)], axis=0)#seq=59, ->[batch, lstm_size]
        self.LSTM_in = tf.stack([self.encoder_out_SQ[(self.batch_size//self.LSTM_batch_size)*idx:(self.batch_size//self.LSTM_batch_size)*(idx+1) - 1,:] 
                            for idx in range(self.LSTM_batch_size)], axis=0)#seq=59, ->[batch,59,lstm_size]
        self.LSTM_out = self.LSTM(self.LSTM_in)#seq=59, ->[batch,lstm_size]
        
        #=-----<LSTM predict -> AE>
        self.predict_encoder_out = tf.reshape(self.LSTM_out, [self.LSTM_batch_size,1,1,-1])

        #=-----<RE Autoencoder>
        self.re_encoder_out = self.conv_encoder(self.decoder_out, reuse=True)

        #=-----<input latent_vec -> output class&price>
        self.ori_class, self.ori_price = self.PriceNet(tf.reshape(self.encoder_out_SQ, [self.batch_size,1,1,-1]))
        self.prd_class, self.prd_price = self.PriceNet(self.predict_encoder_out, reuse=True)


        ##################
        ##     Loss     ##
        ##################
        #=-----<loss>
        self.AE_loss = tf.reduce_mean(tf.square(self.ori_inputs - self.decoder_out))
        self.latent_AE_loss = tf.reduce_mean(tf.square(self.encoder_out - self.re_encoder_out))

        self.LSTM_loss = tf.reduce_mean(tf.square(self.LSTM_out - self.LSTM_label))

        self.ori_class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.ori_class_lb, logits=self.ori_class))
        self.ori_price_loss = tf.reduce_mean(tf.square(self.ori_price - self.ori_price_lb))

        self.prd_class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.prd_class_lb, logits=self.prd_class))
        self.prd_price_loss = tf.reduce_mean(tf.square(self.prd_price - self.prd_price_lb))

        #self.re_ori_class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.ori_class_lb, logits=self.re_ori_class))
        #self.re_ori_price_loss = tf.reduce_mean(tf.square(self.re_ori_price - self.ori_price_lb))
        #self.re_prd_class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.prd_class_lb, logits=self.re_prd_class))
        #self.re_prd_price_loss = tf.reduce_mean(tf.square(self.re_prd_price - self.prd_price_lb))

        #=-----<final loss>
        self.loss = ((self.AE_loss + self.latent_AE_loss + self.LSTM_loss)/10. + \
                (((self.ori_class_loss)*1.0+(self.prd_class_loss)*self.penalty_prd) + \
                 ((self.ori_price_loss)*1.0+(self.prd_price_loss)*self.penalty_prd)*1)*self.penalty_price_net)*self.penalty_sum_loss


        #self.loss = (self.AE_loss + self.latent_AE_loss + self.LSTM_loss + self.predict_recon_loss + \
        #        (((self.ori_class_loss+self.re_ori_class_loss)+(self.prd_class_loss+self.re_prd_class_loss)*self.penalty_prd) + \
        #         ((self.ori_price_loss+self.re_ori_price_loss)+(self.prd_price_loss+self.re_prd_price_loss)*self.penalty_prd)*10)*self.penalty_price_net)*self.penalty_sum_loss
        
        
        ##################
        ## for__output  ##
        ##################    
        self.for_output_prd_class = tf.argmax(self.prd_class,0)

        
        ##################
        ##  set__train  ##
        ##################
        #=-----<predict>
        self.ori_predict = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.ori_class, 1), tf.argmax(self.ori_class_lb, 1)), tf.float32))
        self.prd_predict = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.prd_class, 1), tf.argmax(self.prd_class_lb, 1)), tf.float32))

        #self.re_ori_predict = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.re_ori_class, 1), tf.argmax(self.ori_class_lb, 1)), tf.float32))
        #self.re_prd_predict = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.re_prd_class, 1), tf.argmax(self.prd_class_lb, 1)), tf.float32))

        #=-----<optimizer>
        optimizer = tf.contrib.opt.NadamOptimizer(self.learning_rate)
        self.train = optimizer.minimize(self.loss)

        vars_ = tf.global_variables()
        self.saver = tf.train.Saver(vars_, max_to_keep=3)
        self.sess.run(tf.global_variables_initializer())

        self.restore_model(self.saver)
    
    def output_data(self, x_data, y_data):
        #-data batch
        x_dimension_1 = x_data.shape[0]
        x_dimension_2 = x_data.shape[1]
        x_dimension_3 = x_data.shape[2]
        y_dimension_1 = y_data.shape[0]
        y_dimension_2 = y_data.shape[1]
        y_dimension_3 = y_data.shape[2]
        
        x_data = np.reshape(x_data, [x_dimension_1*x_dimension_2, x_dimension_3])
        y_data = np.reshape(y_data, [y_dimension_1*y_dimension_2, y_dimension_3])

        
        #-input
        batch_ori_inputs = np.reshape(np.concatenate([x_data for idy in range\
                                                      ((self.image_size//self.raw_image_size)**2)],axis=1),\
                                                      [x_dimension_1*x_dimension_2,self.image_size,self.image_size,1]) 
        batch_coin_lb = np.eye(self.coin_count, dtype=np.int8)[y_data[:,5].astype(int)] #[batch,class_num]
        
        #1
        #--predict real loss
        p_class, p_price = self.sess.run([self.for_output_prd_class, self.prd_price], \
                                    feed_dict = {self.ori_inputs:batch_ori_inputs,
                                                 self.coin_lb:batch_coin_lb,
                                                 self.keep_prob:1.0,
                                                 self.LSTM_keep_prob:1.0,
                                                 self.is_training:False})

        return p_price