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
from GNN import prepare_model
from numpy import mean, std
from tensorflow import keras
from tensorflow.keras import Sequential
from sklearn.metrics import accuracy_score
from tensorflow.keras.activations import relu
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support,roc_auc_score, precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, LeakyReLU, concatenate, Conv1D, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    BatchNormalization, SeparableConv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense, LayerNormalization, Add
)
from tensorflow.keras.regularizers import l2

warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def data_split(data):

    X = data[['tcr_embeds', 'epi_embeds']]
    y = data['label']

    shuffle_idx = np.random.permutation(len(X))
    X = X.iloc[shuffle_idx].reset_index(drop=True)
    y = y.iloc[shuffle_idx].reset_index(drop=True)
    
    results = []
    #Cross Validation Split
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        # Split the data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        results.append((X_train, X_val, y_train, y_val))

    return results


def resnet_block(input_layer, filters, kernel_size=3, downsample=False):
    # Shortcut/skip connection
    shortcut = input_layer
    
    # Adjust shortcut if changing number of filters or downsampling
    if downsample or input_layer.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same')(shortcut)
    
    # Main path
    x = Conv1D(filters, kernel_size, padding='same', 
               kernel_regularizer=l2(0.001))(input_layer)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    
    x = Conv1D(filters, kernel_size, padding='same', 
               kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    
    # Combine main path and shortcut
    x = Add()([x, shortcut])
    x = LeakyReLU(alpha=0.01)(x)
    
    return x


def train_(split, X_train_tcr, X_train_epi, y_train, X_val_tcr, X_val_epi, y_val):
    # Define two sets of inputs with more complex initial processing
    inputA = Input(shape=(len(X_train_tcr[0]),))
    inputB = Input(shape=(len(X_train_epi[0]),))
   
    # First branch with residual-like connections
    x = Dense(2048, kernel_initializer='he_uniform', kernel_regularizer=l2(0.001))(inputA)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.4)(x)
    
    x_res = Dense(2048, kernel_initializer='he_uniform')(inputA)
    x = Add()([x, x_res])
    x = BatchNormalization()(x)
    
    # Second branch with similar structure
    y = Dense(2048, kernel_initializer='he_uniform', kernel_regularizer=l2(0.001))(inputB)
    y = BatchNormalization()(y)
    y = LeakyReLU(alpha=0.01)(y)
    y = Dropout(0.4)(y)
    
    y_res = Dense(2048, kernel_initializer='he_uniform')(inputB)
    y = Add()([y, y_res])
    y = BatchNormalization()(y)
    
    combined = concatenate([x, y])
    
    # Deeper and more complex final layers
    z = Dense(1536, activation='selu')(combined)
    z = BatchNormalization()(z)
    z = Dropout(0.5)(z)
    
    z = Dense(1024, activation='selu')(z)
    z = BatchNormalization()(z)
    z = Dropout(0.5)(z)
    
    z = Dense(512, activation='selu')(z)
    z = BatchNormalization()(z)
    z = Dropout(0.4)(z)
    
    z = Dense(1, activation='sigmoid')(z)
    
    # Create and compile the model
    model = Model(inputs=[inputA, inputB], outputs=z)
    
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam')
    
    model.summary()
    
    if split == 'tcr':
        checkpoint_filepath = './best_tcr_model.weights.h5'
    elif split == 'epi':
        checkpoint_filepath = './best_epi_model.weights.h5'

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 30)


    tbCallBack = TensorBoard(log_dir='./logs',
                                histogram_freq=0,
                                write_graph=True, 
                                write_images=True,
                            )
    tbCallBack.set_model(model)


    lrate_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10,
                                        min_delta=0.0001, min_lr=1e-6, verbose=1)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                    save_weights_only=True,
                                                                    monitor='val_loss',
                                                                    mode='min',
                                                                    save_best_only=True)
    
    callbacks = [model_checkpoint_callback, es, tbCallBack, lrate_scheduler]
    
    model.fit([X_train_tcr, X_train_epi], y_train, validation_data=([X_val_tcr, X_val_epi], y_val), callbacks=callbacks, verbose = 1, epochs = 20, batch_size=32)
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 30)
    model.fit([X_train_tcr, X_train_epi], y_train, validation_data=([X_val_tcr, X_val_epi], y_val), verbose=0,  epochs=10, batch_size = 32, callbacks=[es, model_checkpoint_callback])
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

def main(split, gpu):

    if split == 'tcr':
        embeddings_file = './sample_tcr_train.pkl'
    elif split == 'epi':
        embeddings_file = './sample_epi_train.pkl'

    with open(embeddings_file, 'rb') as f:
        data = pickle.load(f)

    res = data_split(data)

    for fold, (X_train, X_val, y_train, y_val) in enumerate(res):

        X_train_tcr  = np.stack(X_train['tcr_embeds'])
        X_train_epi = np.stack(X_train['epi_embeds'])

        X_val_tcr = np.stack(X_val['tcr_embeds'])
        X_val_epi = np.stack(X_val['epi_embeds'])

        y_train = y_train.to_numpy()
        y_val = y_val.to_numpy()

        train_(split, X_train_tcr, X_train_epi, y_train, X_val_tcr, X_val_epi, y_val)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str,help='tcr or epi')
    parser.add_argument('--gpu', type=str)
    args = parser.parse_args()
    main(args.split, args.gpu)