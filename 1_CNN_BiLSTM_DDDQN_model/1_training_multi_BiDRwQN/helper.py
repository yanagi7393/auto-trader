import numpy as np
import random
import tensorflow as tf
import os
import csv
import itertools
import tensorflow.contrib.slim as slim

def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)
    total_vars = len(tf.trainable_variables())
    a = tf.trainable_variables()[0].eval(session=sess)
    b = tf.trainable_variables()[total_vars//2].eval(session=sess)
    if a.all() == b.all():
        pass
    else:
        print("Target Set Failed")
        
def multi_random_action(multi_count,len_cond):
    a = []
    for idx in range(multi_count):
        a.append(np.random.randint(0,len_cond))
                 
    return a

def shuffle_item(item_states, S_item_shuffle):
    if S_item_shuffle == 'on':
        random.shuffle(item_states)
        
    elif S_item_shuffle == 'off':
        pass
    
    return item_states

def shuffle_start_point(start_point,seq_len,end_point,S_multi_start_point):
    if S_multi_start_point == 'on':
        start_point = random.randrange(seq_len, end_point)
        
    elif S_multi_start_point == 'off':
        pass
    
    return start_point