# -*- coding: utf-8 -*-
"""initial-Data processing

The necessary files and libraries are imported here. After that data processing steps including summing vrious tarining and tesing dataset, labeleing,resizing,reshaping,formating are done here.
"""

import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Input, Conv2D, Flatten, MaxPool2D, Activation,BatchNormalization,MaxPooling2D,Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import backend as k
from keras.optimizers import Adamax
from keras.layers.advanced_activations import LeakyReLU

FIG_WIDTH=16
ROW_HEIGHT=3 
RESIZE_DIM=32

#Training Data
paths_train_a=glob.glob(os.path.join('training-a','*.png'))
paths_train_b=glob.glob(os.path.join('training-b','*.png'))
paths_train_e=glob.glob(os.path.join('training-e','*.png'))
paths_train_c=glob.glob(os.path.join('training-c','*.png'))
paths_train_d=glob.glob(os.path.join('training-d','*.png'))
paths_train_all=paths_train_a+paths_train_b+paths_train_c+paths_train_d+paths_train_e

#Testing Data
paths_test_a=glob.glob(os.path.join('testing-a','*.png'))
paths_test_b=glob.glob(os.path.join('testing-b','*.png'))
paths_test_c=glob.glob(os.path.join('testing-c','*.png'))
paths_test_d=glob.glob(os.path.join('testing-d','*.png'))
paths_test_e=glob.glob(os.path.join('testing-e','*.png'))

paths_test_f=glob.glob(os.path.join('testing-f','*.png'))+glob.glob(os.path.join('testing-f','*.JPG'))
paths_test_auga=glob.glob(os.path.join('testing-auga','*.png'))
paths_test_augc=glob.glob(os.path.join('testing-augc','*.png'))
paths_test_all=paths_test_a+paths_test_b+paths_test_c+paths_test_d+paths_test_e+paths_test_f+paths_test_auga+paths_test_augc

#Labeling Data
path_label_train_a=os.path.join('training-a.csv')
path_label_train_b=os.path.join('training-b.csv')
path_label_train_e=os.path.join('training-e.csv')
path_label_train_c=os.path.join('training-c.csv')
path_label_train_d=os.path.join('training-d.csv')

def get_key(path):
    key=path.split(sep=os.sep)[-1]
    return key

def get_data(paths_img,path_label=None,resize_dim=None):
    X=[]
    for i,path in enumerate(paths_img):
        img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        if resize_dim!=None:
            img=cv2.resize(img,(RESIZE_DIM,RESIZE_DIM),interpolation=cv2.INTER_AREA)
        gaussian_3=cv2.GaussianBlur(img,(9,9),10.0)
        img=cv2.addWeighted(img,1.5,gaussian_3,-0.5,0,img)
        kernel=np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img=cv2.filter2D(img,-1,kernel)
        X.append(img)
        
        #display progress
        
        if i==len(paths_img)-1:
            end='\n'
        else:
            end='\r'
        print('processed {}/{}'.format(i+1,len(paths_img)),end=end)
    X=np.array(X)
    if path_label is None:
        return X
    else:
        df=pd.read_csv(path_label)
        df=df.set_index('filename')
        y_label=[df.loc[get_key(path)]['digit'] for path in  paths_img]
        y=to_categorical(y_label,10)
        return X,y

def imshow_group(X,y,y_pred=None,n_per_row=10,phase='processed'):
    n_sample=len(X)
    img_dim=X.shape[1]
    j=np.ceil(n_sample/n_per_row)
    fig=plt.figure(figsize=(FIG_WIDTH,ROW_HEIGHT*j))
    
    for i,img in enumerate(X):
        plt.subplot(j,n_per_row,i+1)
        plt.imshow(img)
        if phase=='processed':
            plt.title(np.argmax(y[i]))
        if phase=='prediction':
            top_n=3
            ind_sorted=np.argsort(y_pred[i])[::-1]
            h=img_dim+4
            for k in range(top_n):
                string='pred: {} ({:.0f}%)\n'.format(ind_sorted[k],y_pred[i,ind_sorted[k]]*100)
                plt.text(img_dim/2, h, string, horizontalalignment='center',verticalalignment='center')
                h+=4
            if y is not None:
                plt.text(img_dim/2, -4, 'true label: {}'.format(np.argmax(y[i])), 
                         horizontalalignment='center',verticalalignment='center')
            plt.axis('off')
    plt.show()

X_train_a, y_train_a = get_data(paths_train_a, path_label_train_a, resize_dim = RESIZE_DIM)
X_train_b, y_train_b = get_data(paths_train_b, path_label_train_b, resize_dim = RESIZE_DIM)
X_train_c, y_train_c = get_data(paths_train_c, path_label_train_c, resize_dim = RESIZE_DIM)
X_train_d, y_train_d = get_data(paths_train_d, path_label_train_d, resize_dim = RESIZE_DIM)
X_train_e, y_train_e = get_data(paths_train_e, path_label_train_e, resize_dim = RESIZE_DIM)

X_train_all=np.concatenate((X_train_a,X_train_b,X_train_c,X_train_d,X_train_e),axis=0)
y_train_all=np.concatenate((y_train_a,y_train_b,y_train_c,y_train_d,y_train_e),axis=0)
X_train_all.shape, y_train_all.shape

X_show_all=X_train_all

plt.subplot(221)
plt.imshow(X_train_all[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train_all[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train_all[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train_all[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()

plt.imshow(X_train_all[1])

X_test_a=get_data(paths_test_a,resize_dim=RESIZE_DIM)
X_test_b=get_data(paths_test_b,resize_dim=RESIZE_DIM)
X_test_c=get_data(paths_test_c,resize_dim=RESIZE_DIM)
X_test_d=get_data(paths_test_d,resize_dim=RESIZE_DIM)
X_test_e=get_data(paths_test_e,resize_dim=RESIZE_DIM)
X_test_f=get_data(paths_test_f,resize_dim=RESIZE_DIM)
X_test_auga=get_data(paths_test_auga,resize_dim=RESIZE_DIM)
X_test_augc=get_data(paths_test_augc,resize_dim=RESIZE_DIM)

X_test_all=np.concatenate((X_test_a,X_test_b,X_test_c,X_test_d,X_test_e,X_test_f,X_test_auga,X_test_augc))

X_tshow_all=X_test_all
X_tshow_all.shape

X_train_all = X_train_all.reshape(X_train_all.shape[0],32, 32,1).astype('float32')
X_test_all = X_test_all.reshape(X_test_all.shape[0],32, 32,1).astype('float32')

X_train_all = X_train_all/255.0
X_test_all=X_test_all/255.0

indices=list(range(len(X_train_all)))
np.random.seed(42)
np.random.shuffle(indices)

ind=int(len(indices)*0.80)
# train data
X_train=X_train_all[indices[:ind]] 
y_train=y_train_all[indices[:ind]]
# validation data
X_val=X_train_all[indices[-(len(indices)-ind):]] 
y_val=y_train_all[indices[-(len(indices)-ind):]]
