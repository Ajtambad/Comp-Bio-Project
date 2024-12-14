import sys
import time
import os
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import torch
import pickle
from numpy import mean, std
from tensorflow import keras
from tensorflow.keras import Sequential
from sklearn.metrics import accuracy_score
from tensorflow.keras.activations import relu
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.metrics import precision_recall_fscore_support,roc_auc_score, precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, LeakyReLU, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    BatchNormalization, SeparableConv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense, LayerNormalization
)

warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def data_split(split):
    if split == 'tcr':
        embeddings_file = './sample_tcr_train.pkl'
    elif split == 'epi':
        embeddings_file = './sample_epi_train.pkl'

    with open(embeddings_file, 'rb') as f:
        data = pickle.load(f)

    df = pd.DataFrame(data)
    df = df[:999]
    df.loc[500:1000, 'label'] = 0

    X = df[['tcr_embeds', 'epi_embeds']]
    y = df['label']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, random_state=0)

    X_train_tcr  = np.stack(X_train['tcr_embeds'])
    X_train_epi = np.stack(X_train['epi_embeds'])

    X_val_tcr = np.stack(X_val['tcr_embeds'])
    X_val_epi = np.stack(X_val['epi_embeds'])

    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()

    return X_train_tcr, X_train_epi, y_train, X_val_tcr, X_val_epi, y_val

def train_(split, X_train_tcr, X_train_epi, y_train, X_val_tcr, X_val_epi, y_val):
    # define two sets of inputs
    inputA = Input(shape=(len(X_train_tcr[0]),))
    inputB = Input(shape=(len(X_train_epi[0]),))
    
    x = Dense(2048,kernel_initializer = 'he_uniform')(inputA)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = relu(x)
    x = Model(inputs=inputA, outputs=x)
    
    y = Dense(2048,kernel_initializer = 'he_uniform')(inputB)
    y = BatchNormalization()(y)
    y = Dropout(0.3)(y)
    y = relu(y)
    y = Model(inputs=inputB, outputs=y)
    combined = concatenate([x.output, y.output])
    
    z = Dense(1024)(combined)
    z = BatchNormalization()(z)
    z = Dropout(0.3)(z)
    z = relu(z)
    z = Dense(1, activation='sigmoid')(z)
    model = Model(inputs=[x.input, y.input], outputs=z)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam')
    model.summary()
    
    
    ## model fit
    if split == 'tcr':
        checkpoint_filepath = './best_tcr_model.weights.h5'
    elif split == 'epi':
        checkpoint_filepath = './best_epi_model.weights.h5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                    save_weights_only=True,
                                                                    monitor='val_loss',
                                                                    mode='min',
                                                                    save_best_only=True)
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 30)
    model.fit([X_train_tcr, X_train_epi], y_train, validation_data=([X_val_tcr, X_val_epi], y_val), verbose=0,  epochs=200, batch_size = 32, callbacks=[es, model_checkpoint_callback])
    model.save_weights(checkpoint_filepath, overwrite=True)
    y_preds = model.predict([X_val_tcr, X_val_epi])

    print('================Performance========================')
    print('AUC: ' + str(roc_auc_score(y_val, y_preds)))

    
    y_preds[y_preds>=0.5] = 1
    y_preds[y_preds<0.5] = 0
    
    accuracy = accuracy_score(y_val, y_preds)
    precision1 = precision_score(
        y_val, y_preds, pos_label=1, zero_division=0)
    precision0 = precision_score(
        y_val, y_preds, pos_label=0, zero_division=0)
    recall1 = recall_score(y_val, y_preds, pos_label=1, zero_division=0)
    recall0 = recall_score(y_val, y_preds, pos_label=0, zero_division=0)
    f1macro = f1_score(y_val, y_preds, average='macro')
    f1micro = f1_score(y_val, y_preds, average='micro')
    
    print('precision_recall_fscore_macro ' + str(precision_recall_fscore_support(y_val,y_preds, average='macro')))
    print('acc is '  + str(accuracy))
    print('precision1 is '  + str(precision1))
    print('precision0 is '  + str(precision0))
    print('recall1 is '  + str(recall1))
    print('recall0 is '  + str(recall0))
    print('f1macro is '  + str(f1macro))
    print('f1micro is '  + str(f1micro))


def main(split, gpu):
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu
    X_train_tcr, X_train_epi, y_train, X_val_tcr, X_val_epi, y_val = data_split(split)
    train_(split, X_train_tcr, X_train_epi, y_train, X_val_tcr, X_val_epi, y_val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str,help='tcr or epi')
    parser.add_argument('--gpu', type=str)
    args = parser.parse_args()
    main(args.split, args.gpu)
