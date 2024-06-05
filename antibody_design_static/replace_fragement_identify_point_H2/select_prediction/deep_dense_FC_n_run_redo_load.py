import keras
import keras.backend as K
import pandas as pd
from pandas import DataFrame
from keras.models import Model, load_model
from keras.layers import Input, merge, Activation, Dropout, Dense, concatenate, Concatenate, Flatten
from keras.layers.convolutional import Convolution1D
from keras.layers.pooling import AveragePooling1D, GlobalAveragePooling1D, MaxPool1D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from keras.optimizers import RMSprop


from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

#import xgboost as xgb
from sklearn import metrics

import os
import yaml
import numpy as np
import random
import sys

random.seed(0)


def randomShuffle_name(X, Y, Z):
    random.seed(0)
    idx = [t for t in range(X.shape[0])]
    random.shuffle(idx)
    X = X[idx]
    Y = Y[idx]
    print Z
    print idx
    Z = Z[idx]

    print()
    print('-' * 36)
    print('dimension of X after synthesis:', X.shape)
    print('dimension of Y after synthesis', Y.shape)
    print('label after shuffle:', '\n', DataFrame(Y).head())
    print('-' * 36)
    return X, Y, Z




def randomShuffle(X, Y):
    random.seed(0)
    idx = [t for t in range(X.shape[0])]
    random.shuffle(idx)
    X = X[idx]
    Y = Y[idx]
    print()
    print('-' * 36)
    print('dimension of X after synthesis:', X.shape)
    print('dimension of Y after synthesis', Y.shape)
    print('label after shuffle:', '\n', DataFrame(Y).head())
    print('-' * 36)
    return X, Y

def synData(X_0, Y_0, X_1, Y_1,N_0,N_1, time):

    X_0_syn = X_0
    Y_0_syn = Y_0
    N_0_syn = N_0
    for i in range(time - 1):
        X_0_syn = np.vstack( (X_0_syn, X_0) )
        Y_0_syn = np.hstack( (Y_0_syn, Y_0) )
        N_0_syn =np.hstack( (N_0_syn, N_0) )

    print('dimension of generation data of X', X_0_syn.shape)
    print('dimension of generation data of Y', Y_0_syn.shape)
    print('dimension of generation data of X with label of 1', X_1.shape)
    print('dimension of generation data of Y with label of 1', Y_1.shape)

    #synthesis dataset
    X_syn = np.vstack( (X_0_syn, X_1) )
    Y_syn = np.hstack( (Y_0_syn, Y_1) )
    N_syn = np.hstack( (N_0_syn, N_1) )

    print()
    print('dimension of X after combination', X_syn.shape)
    print('dimension of Y after combination', Y_syn.shape)
    print(DataFrame(Y_syn).head())

    #shuffle data
    X_syn, Y_syn, N_syn = randomShuffle_name(X_syn, Y_syn, N_syn)

    return X_syn, Y_syn, N_syn


input_f=sys.argv[1]
positive_dic_vec=np.load(input_f, allow_pickle=True).item()

#dic_vec_aff=np.load('dict_aff.npy', allow_pickle=True).item()



pos_num=len(positive_dic_vec.values())
print pos_num
pos = np.zeros((pos_num, 800))
all_name=[]
index=0
for  key   in   positive_dic_vec.keys():
    #print(key)
    pos[index,:] = positive_dic_vec[key]
    print len(pos[index,:])
    all_name.append(key)
    index=index+1


all_name=np.array(all_name)

train_X = pos[0:,:]
train_X = train_X.astype(np.float64)
train_name = all_name[0:]





#train_X, train_Y, train_name=randomShuffle_name(train_X, train_Y,train_name)







def preprocess_data_train(data_set,train_set):
    mean = np.mean(train_set)
    std = np.std(train_set)

    t = data_set

    t -= mean
    t /= std
    return t



def preprocess_data(data_set):
    mean = np.mean(data_set, axis=1, keepdims=True)
    std = np.std(data_set,axis=1,keepdims=True)
    np.save('mean_std.npy', [mean, std])
    #mean, std = np.load('mean_std.npy', allow_pickle=True)

    t = data_set

    t -= mean
    t /= std
    print (mean)
    print (std)
    return t



def aucJ(true_labels, predictions):
    
    fpr, tpr, thresholds = metrics.roc_curve(true_labels, predictions, pos_label=1)
    auc = metrics.auc(fpr,tpr)

    return auc

def acc(true, pred):
    
    return np.sum(true == pred) * 1.0 / len(true)

def assess(model, X, label, thre = 0.5):
    
    threshold = thre
    
    pred = model.predict(X)
    pred = pred.flatten()
    
    pred[pred > threshold] = 1
    pred[pred <= threshold] = 0
    
    auc = aucJ(label, pred)
    accuracy = acc(label, pred)
    
    print('auc: ', auc)
    print('accuracy: ', accuracy)



train_X = preprocess_data(train_X)


def densef(X_prev, X):
    
    return concatenate([X_prev, X], axis = -1)

def dfblock(X, layer_num, dropout = 0.1, shape = 200, l2_reg = 1e-4):
    
    X_prev = Dense(shape, init = 'glorot_normal', activation = 'relu', kernel_regularizer = l2(l2_reg))(X)
    X_prev = BatchNormalization(axis = 1, beta_regularizer = l2(l2_reg), gamma_regularizer = l2(l2_reg))(X_prev)
    X_prev = densef(X, X_prev)
    X_prev = Dropout(dropout)(X_prev)
    
    for i in range(layer_num - 1):
        X = Dense(shape, init = 'glorot_normal', activation = 'relu', kernel_regularizer = l2(l2_reg))(X_prev)
        X = BatchNormalization(axis = 1, beta_regularizer = l2(l2_reg), gamma_regularizer = l2(l2_reg))(X)
        X_prev = densef(X, X_prev)
        X_prev = Dropout(dropout)(X_prev)
    
    return X_prev

def noblock(X, dropout = 0.1, shape = 800, l2_reg = 1e-4):
    
    X = Dense(shape, init = 'glorot_normal', activation = 'relu', kernel_regularizer = l2(l2_reg))(X)
    X = BatchNormalization(axis = 1)(X)
    X = Dropout(dropout)(X)
    
    return X

def Dense_FCNN1(input_shape, dense_layer = 3, layer_num = 3, denshape = 200, dropout = 0.1, l2_reg = 1e-4):
    
    X_input = Input(input_shape)
    
    X = dfblock(X_input, layer_num = layer_num, dropout = dropout, shape = denshape, l2_reg = l2_reg)
    X = noblock(X, dropout, input_shape[0], l2_reg)
    
    for i in range(dense_layer - 1):
        X = dfblock(X, layer_num = layer_num, dropout = dropout, shape = denshape, l2_reg = l2_reg)
        X = noblock(X, dropout, input_shape[0], l2_reg)
    
    out = Dense(1, init = 'glorot_normal', activation = 'sigmoid')(X)
    
    model = Model(inputs = X_input, outputs = out, name = 'FCNN1')
    
    return model

model2 = Dense_FCNN1((800, ), 1, 16, 128, 0.15, 1e-4)



optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model2.compile(optimizer = optimizer , loss = "binary_crossentropy", metrics=["accuracy"])

'''
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)
'''






model2.summary()


checkpointer= ModelCheckpoint('model_n_{epoch:03d}.h5', verbose=1, save_weights_only=False,period=20)

#history=model2.fit(x = train_X, y = train_Y, epochs = 1500, batch_size = 64, validation_data = (valid_X,valid_Y),callbacks=[checkpointer])






model2 = load_model('FCdenset.h5')

#assess(model2, train_X, train_Y)
#assess(model2, test_X, test_Y)

pred = model2.predict(train_X)
pred = pred.flatten()
print (pred)

import pandas as pd

name=train_name
listA=zip(list(name),list(pred))
#for i in range(len(pred)):
#     listA[i][0]=pred[i]
#     listA[i][1]=name[i]
y=sorted(listA,key=lambda l:l[1],reverse=True)
fold=input_f.replace('_dic_contact_vec_e.npy','')
fw=open(input_f.replace('_dic_contact_vec_e.npy','_')+'sorted_out.txt','w')
for i in range(len(y)):
     print (y[i][0], y[i][1])
     fw.write(fold+"  "+str(y[i][0]) +"  "+ str(y[i][1]))
     fw.write('\n')

fw.close()








