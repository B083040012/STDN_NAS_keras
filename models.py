import pickle
from random import choice, choices, random
from unicodedata import name
import numpy as np
import json
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Activation, ReLU, PReLU, Input, Conv2D, Reshape, Flatten, Dropout, BatchNormalization, Concatenate, LSTM, MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, Callback, ModelCheckpoint
import attention

class baselines:
    def __init__(self):
        pass

class STDN_NAS(keras.Model):
    def __init__(self, att_lstm_num, att_lstm_seq_len, lstm_seq_len, feature_vec_len, cnn_flat_size = 128, lstm_out_size = 128,\
    nbhd_size = 3, nbhd_type = 2, map_x_num = 10, map_y_num = 20, flow_type = 4, output_shape = 2):
        super(STDN_NAS, self).__init__()

        self.att_lstm_num = att_lstm_num
        self.att_lstm_seq_len = att_lstm_seq_len
        self.lstm_seq_len = lstm_seq_len
        self.feature_vec_len = feature_vec_len
        self.cnn_flat_size = cnn_flat_size
        self.lstm_out_size = lstm_out_size
        self.nbhd_size = nbhd_size
        self.nbhd_type = nbhd_type
        self.map_x_num = map_x_num
        self.map_y_num = map_y_num
        self.flow_type = flow_type
        self.output_size = output_shape

        self.level=3
        self.kernel_list=[1,2,3]
        self.pool_size_list=[2,3,4]

        # nbhd / flow layer in the short-term part
        self.nbhd_convs_list_time0=[]
        self.nbhd_convs_list_time1=[]
        self.nbhd_convs_list_time2=[]
        self.flow_convs_list_time0=[]
        self.flow_convs_list_time1=[]
        self.flow_convs_list_time2=[]
        for ts in range(self.lstm_seq_len):
            for kernel_size in self.kernel_list:
                self.nbhd_convs_list_time0.append(Conv2D(filters = 64, kernel_size = (kernel_size, kernel_size), padding="same", \
                                                    name = "nbhd_convs_time0_{0}".format(ts+1)))
                self.nbhd_convs_list_time1.append(Conv2D(filters = 64, kernel_size = (kernel_size, kernel_size), padding="same", \
                                                    name = "nbhd_convs_time1_{0}".format(ts+1)))
                self.nbhd_convs_list_time2.append(Conv2D(filters = 64, kernel_size = (kernel_size, kernel_size), padding="same", \
                                                    name = "nbhd_convs_time2_{0}".format(ts+1)))
                self.flow_convs_list_time0.append(Conv2D(filters = 64, kernel_size = (kernel_size, kernel_size), padding="same", \
                                                    name = "flow_convs_time0_{0}".format(ts+1)))
                self.flow_convs_list_time1.append(Conv2D(filters = 64, kernel_size = (kernel_size, kernel_size), padding="same", \
                                                    name = "flow_convs_time1_{0}".format(ts+1)))
                self.flow_convs_list_time2.append(Conv2D(filters = 64, kernel_size = (kernel_size, kernel_size), padding="same", \
                                                    name = "flow_convs_time2_{0}".format(ts+1)))

        # nbhd / flow relu activation function in short-term part
        self.nbhd_relu_list_time0=[]
        self.nbhd_relu_list_time1=[]
        self.nbhd_relu_list_time2=[]
        self.flow_relu_list_time0=[]
        self.flow_relu_list_time1=[]
        self.flow_relu_list_time2=[]
        for ts in range(self.lstm_seq_len):
            self.nbhd_relu_list_time0.append(ReLU(name = "nbhd_relu_time0_{0}".format(ts+1)))
            self.nbhd_relu_list_time0.append(ReLU(max_value = 6.0, name = "nbhd_relu6_time0_{0}".format(ts+1)))
            self.nbhd_relu_list_time0.append(PReLU(name = "nbhd_prelu_time0_{0}".format(ts+1)))
            self.nbhd_relu_list_time1.append(ReLU(name = "nbhd_relu_time1_{0}".format(ts+1)))
            self.nbhd_relu_list_time1.append(ReLU(max_value = 6.0, name = "nbhd_relu6_time1_{0}".format(ts+1)))
            self.nbhd_relu_list_time1.append(PReLU(name = "nbhd_prelu_time1_{0}".format(ts+1)))
            self.nbhd_relu_list_time2.append(ReLU(name = "nbhd_relu_time2_{0}".format(ts+1)))
            self.nbhd_relu_list_time2.append(ReLU(max_value = 6.0, name = "nbhd_relu6_time2_{0}".format(ts+1)))
            self.nbhd_relu_list_time2.append(PReLU(name = "nbhd_prelu_time2_{0}".format(ts+1)))
            self.flow_relu_list_time0.append(ReLU(name = "flow_relu_time0_{0}".format(ts+1)))
            self.flow_relu_list_time0.append(ReLU(max_value = 6.0, name = "flow_relu6_time0_{0}".format(ts+1)))
            self.flow_relu_list_time0.append(PReLU(name = "flow_prelu_time0_{0}".format(ts+1)))
            self.flow_relu_list_time1.append(ReLU(name = "flow_relu_time1_{0}".format(ts+1)))
            self.flow_relu_list_time1.append(ReLU(max_value = 6.0, name = "flow_relu6_time1_{0}".format(ts+1)))
            self.flow_relu_list_time1.append(PReLU(name = "flow_prelu_time1_{0}".format(ts+1)))
            self.flow_relu_list_time2.append(ReLU(name = "flow_relu_time2_{0}".format(ts+1)))
            self.flow_relu_list_time2.append(ReLU(max_value = 6.0, name = "flow_relu6_time2_{0}".format(ts+1)))
            self.flow_relu_list_time2.append(PReLU(name = "flow_prelu_time2_{0}".format(ts+1)))


        # nbhd / flow pooling layer in the short-term part
        self.nbhd_pooling_list_time0=[]
        self.nbhd_pooling_list_time1=[]
        self.nbhd_pooling_list_time2=[]
        self.flow_pooling_list_time0=[]
        self.flow_pooling_list_time1=[]
        self.flow_pooling_list_time2=[]
        for ts in range(self.lstm_seq_len):
            for pooling_size in self.pool_size_list:
                self.nbhd_pooling_list_time0.append(MaxPooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                    name = "nbhd_max_pooling_time0_{0}".format(ts+1)))
                self.nbhd_pooling_list_time0.append(AveragePooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                    name = "nbhd_avg_pooling_time0_{0}".format(ts+1)))
                self.nbhd_pooling_list_time1.append(MaxPooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                    name = "nbhd_max_pooling_time1_{0}".format(ts+1)))
                self.nbhd_pooling_list_time1.append(AveragePooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                    name = "nbhd_avg_pooling_time1_{0}".format(ts+1)))
                self.nbhd_pooling_list_time2.append(MaxPooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                    name = "nbhd_max_pooling_time0_{0}".format(ts+1)))
                self.nbhd_pooling_list_time2.append(AveragePooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                    name = "nbhd_avg_pooling_time0_{0}".format(ts+1)))
                self.flow_pooling_list_time0.append(MaxPooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                    name = "flow_max_pooling_time0_{0}".format(ts+1)))
                self.flow_pooling_list_time0.append(AveragePooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                    name = "flow_avg_pooling_time0_{0}".format(ts+1)))
                self.flow_pooling_list_time1.append(MaxPooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                    name = "flow_max_pooling_time1_{0}".format(ts+1)))
                self.flow_pooling_list_time1.append(AveragePooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                    name = "flow_avg_pooling_time1_{0}".format(ts+1)))
                self.flow_pooling_list_time2.append(MaxPooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                    name = "flow_max_pooling_time2_{0}".format(ts+1)))
                self.flow_pooling_list_time2.append(AveragePooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                    name = "flow_avg_pooling_time2_{0}".format(ts+1)))


        # dense layer in the short-term part
        self.short_term_dense_list=[]
        for ts in range(self.lstm_seq_len):
            self.short_term_dense_list.append(Dense(units = self.cnn_flat_size, name = "nbhd_dense_time_{0}".format(ts+1)))

        # # weather dense layer in the short-term part
        # self.short_term_weather_dense_list=[]
        # for ts in range(self.lstm_seq_len):
        #     self.short_term_weather_dense_list.append(Dense(units = self.cnn_flat_size, name = "weather_dense_time_{0}".format(ts+1)))

        # lstm layer in short-term part
        self.short_term_lstm=LSTM(units=self.lstm_out_size, return_sequences=False, dropout=0.1, recurrent_dropout=0)

        # nbhd / flow conv layers in the att portion
        self.att_nbhd_convs_list_time0=[]
        self.att_nbhd_convs_list_time1=[]
        self.att_nbhd_convs_list_time2=[]
        self.att_flow_convs_list_time0=[]
        self.att_flow_convs_list_time1=[]
        self.att_flow_convs_list_time2=[]
        for att in range(self.att_lstm_num):
            self.att_nbhd_convs_list_time0.append([])
            self.att_nbhd_convs_list_time1.append([])
            self.att_nbhd_convs_list_time2.append([])
            self.att_flow_convs_list_time0.append([])
            self.att_flow_convs_list_time1.append([])
            self.att_flow_convs_list_time2.append([])
            for ts in range(self.att_lstm_seq_len):
                self.att_nbhd_convs_list_time0[att].append([])
                self.att_nbhd_convs_list_time1[att].append([])
                self.att_nbhd_convs_list_time2[att].append([])
                self.att_flow_convs_list_time0[att].append([])
                self.att_flow_convs_list_time1[att].append([])
                self.att_flow_convs_list_time2[att].append([])
                for kernel_size in self.kernel_list:
                    self.att_nbhd_convs_list_time0[att][ts].append(Conv2D(filters = 64, kernel_size = (kernel_size, kernel_size), padding="same", \
                                                name = "att_nbhd_convs_time0_{0}_{1}".format(att+1,ts+1)))
                    self.att_nbhd_convs_list_time1[att][ts].append(Conv2D(filters = 64, kernel_size = (kernel_size, kernel_size), padding="same", \
                                                name = "att_nbhd_convs_time1_{0}_{1}".format(att+1,ts+1)))
                    self.att_nbhd_convs_list_time2[att][ts].append(Conv2D(filters = 64, kernel_size = (kernel_size, kernel_size), padding="same", \
                                                name = "att_nbhd_convs_time2_{0}_{1}".format(att+1,ts+1)))
                    self.att_flow_convs_list_time0[att][ts].append(Conv2D(filters = 64, kernel_size = (kernel_size, kernel_size), padding="same", \
                                                name = "att_flow_convs_time0_{0}_{1}".format(att+1,ts+1)))
                    self.att_flow_convs_list_time1[att][ts].append(Conv2D(filters = 64, kernel_size = (kernel_size, kernel_size), padding="same", \
                                                name = "att_flow_convs_time1_{0}_{1}".format(att+1,ts+1)))
                    self.att_flow_convs_list_time2[att][ts].append(Conv2D(filters = 64, kernel_size = (kernel_size, kernel_size), padding="same", \
                                                name = "att_flow_convs_time2_{0}_{1}".format(att+1,ts+1)))

        # nbhd / flow pooling layers in the att portion
        self.att_nbhd_pooling_list_time0=[]
        self.att_nbhd_pooling_list_time1=[]
        self.att_nbhd_pooling_list_time2=[]
        self.att_flow_pooling_list_time0=[]
        self.att_flow_pooling_list_time1=[]
        self.att_flow_pooling_list_time2=[]
        for att in range(self.att_lstm_num):
            self.att_nbhd_pooling_list_time0.append([])
            self.att_nbhd_pooling_list_time1.append([])
            self.att_nbhd_pooling_list_time2.append([])
            self.att_flow_pooling_list_time0.append([])
            self.att_flow_pooling_list_time1.append([])
            self.att_flow_pooling_list_time2.append([])
            for ts in range(self.att_lstm_seq_len):
                self.att_nbhd_pooling_list_time0[att].append([])
                self.att_nbhd_pooling_list_time1[att].append([])
                self.att_nbhd_pooling_list_time2[att].append([])
                self.att_flow_pooling_list_time0[att].append([])
                self.att_flow_pooling_list_time1[att].append([])
                self.att_flow_pooling_list_time2[att].append([])
                for pooling_size in self.pool_size_list:
                    self.att_nbhd_pooling_list_time0[att][ts].append(MaxPooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                                        name = "att_nbhd_max_pooling_time0_{0}_{1}".format(att+1, ts+1)))
                    self.att_nbhd_pooling_list_time0[att][ts].append(AveragePooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                                        name = "att_nbhd_avg_pooling_time0_{0}_{1}".format(att+1, ts+1)))
                    self.att_nbhd_pooling_list_time1[att][ts].append(MaxPooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                                        name = "att_nbhd_max_pooling_time1_{0}_{1}".format(att+1, ts+1)))
                    self.att_nbhd_pooling_list_time1[att][ts].append(AveragePooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                                        name = "att_nbhd_avg_pooling_time1_{0}_{1}".format(att+1, ts+1)))
                    self.att_nbhd_pooling_list_time2[att][ts].append(MaxPooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                                        name = "att_nbhd_max_pooling_time2_{0}_{1}".format(att+1, ts+1)))
                    self.att_nbhd_pooling_list_time2[att][ts].append(AveragePooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                                        name = "att_nbhd_avg_pooling_time2_{0}_{1}".format(att+1, ts+1)))
                    self.att_flow_pooling_list_time0[att][ts].append(MaxPooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                                        name = "att_flow_max_pooling_time0_{0}_{1}".format(att+1, ts+1)))
                    self.att_flow_pooling_list_time0[att][ts].append(AveragePooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                                        name = "att_flow_avg_pooling_time0_{0}_{1}".format(att+1, ts+1)))
                    self.att_flow_pooling_list_time1[att][ts].append(MaxPooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                                        name = "att_flow_max_pooling_time1_{0}_{1}".format(att+1, ts+1)))
                    self.att_flow_pooling_list_time1[att][ts].append(AveragePooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                                        name = "att_flow_avg_pooling_time1_{0}_{1}".format(att+1, ts+1)))
                    self.att_flow_pooling_list_time2[att][ts].append(MaxPooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                                        name = "att_flow_max_pooling_time2_{0}_{1}".format(att+1, ts+1)))
                    self.att_flow_pooling_list_time2[att][ts].append(AveragePooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                                        name = "att_flow_avg_pooling_time2_{0}_{1}".format(att+1, ts+1)))

        # nbhd / flow relu layers in the att portion
        self.att_nbhd_relu_list_time0=[]
        self.att_nbhd_relu_list_time1=[]
        self.att_nbhd_relu_list_time2=[]
        self.att_flow_relu_list_time0=[]
        self.att_flow_relu_list_time1=[]
        self.att_flow_relu_list_time2=[]
        for att in range(self.att_lstm_num):
            self.att_nbhd_relu_list_time0.append([])
            self.att_nbhd_relu_list_time1.append([])
            self.att_nbhd_relu_list_time2.append([])
            self.att_flow_relu_list_time0.append([])
            self.att_flow_relu_list_time1.append([])
            self.att_flow_relu_list_time2.append([])
            for ts in range(self.att_lstm_seq_len):
                self.att_nbhd_relu_list_time0[att].append([])
                self.att_nbhd_relu_list_time1[att].append([])
                self.att_nbhd_relu_list_time2[att].append([])
                self.att_flow_relu_list_time0[att].append([])
                self.att_flow_relu_list_time1[att].append([])
                self.att_flow_relu_list_time2[att].append([])
                self.att_nbhd_relu_list_time0[att][ts].append(ReLU(name = "att_nbhd_relu_time0_{0}_{1}".format(att+1, ts+1)))
                self.att_nbhd_relu_list_time0[att][ts].append(ReLU(max_value = 6.0, name = "att_nbhd_relu6_time0_{0}_{1}".format(att+1, ts+1)))
                self.att_nbhd_relu_list_time0[att][ts].append(PReLU(name = "att_nbhd_prelu_time0_{0}_{1}".format(att+1, ts+1)))
                self.att_nbhd_relu_list_time1[att][ts].append(ReLU(name = "att_nbhd_relu_time1_{0}_{1}".format(att+1, ts+1)))
                self.att_nbhd_relu_list_time1[att][ts].append(ReLU(max_value = 6.0, name = "att_nbhd_relu6_time1_{0}_{1}".format(att+1, ts+1)))
                self.att_nbhd_relu_list_time1[att][ts].append(PReLU(name = "att_nbhd_prelu_time1_{0}_{1}".format(att+1, ts+1)))
                self.att_nbhd_relu_list_time2[att][ts].append(ReLU(name = "att_nbhd_relu_time2_{0}_{1}".format(att+1, ts+1)))
                self.att_nbhd_relu_list_time2[att][ts].append(ReLU(max_value = 6.0, name = "att_nbhd_relu6_time2_{0}_{1}".format(att+1, ts+1)))
                self.att_nbhd_relu_list_time2[att][ts].append(PReLU(name = "att_nbhd_prelu_time2_{0}_{1}".format(att+1, ts+1)))
                self.att_flow_relu_list_time0[att][ts].append(ReLU(name = "att_flow_relu_time0_{0}_{1}".format(att+1, ts+1)))
                self.att_flow_relu_list_time0[att][ts].append(ReLU(max_value = 6.0, name = "att_flow_relu6_time0_{0}_{1}".format(att+1, ts+1)))
                self.att_flow_relu_list_time0[att][ts].append(PReLU(name = "att_flow_prelu_time0_{0}_{1}".format(att+1, ts+1)))
                self.att_flow_relu_list_time1[att][ts].append(ReLU(name = "att_flow_relu_time1_{0}_{1}".format(att+1, ts+1)))
                self.att_flow_relu_list_time1[att][ts].append(ReLU(max_value = 6.0, name = "att_flow_relu6_time1_{0}_{1}".format(att+1, ts+1)))
                self.att_flow_relu_list_time1[att][ts].append(PReLU(name = "att_flow_prelu_time1_{0}_{1}".format(att+1, ts+1)))
                self.att_flow_relu_list_time2[att][ts].append(ReLU(name = "att_flow_relu_time2_{0}_{1}".format(att+1, ts+1)))
                self.att_flow_relu_list_time2[att][ts].append(ReLU(max_value = 6.0, name = "att_flow_relu6_time2_{0}_{1}".format(att+1, ts+1)))
                self.att_flow_relu_list_time2[att][ts].append(PReLU(name = "att_flow_prelu_time2_{0}_{1}".format(att+1, ts+1)))
                    

        # attention part dense list
        self.att_dense_list=[]
        for att in range(self.att_lstm_num):
            self.att_dense_list.append([])
            for ts in range(self.att_lstm_seq_len):
                self.att_dense_list[att].append(Dense(units = self.cnn_flat_size, name = "att_nbhd_dense_time_{0}_{1}".format(att+1,ts+1)))

        # attention part lstm list
        self.att_lstm_list=[]
        for att in range(self.att_lstm_num):
            self.att_lstm_list.append(LSTM(units=self.lstm_out_size, return_sequences=True, dropout=0.1, \
                                        recurrent_dropout=0,recurrent_activation='sigmoid', name="att_lstm_{0}".format(att + 1)))
    
        # high_level_lstm
        self.high_level_lstm=LSTM(units=self.lstm_out_size, return_sequences=False, dropout=0.1, recurrent_dropout=0)
        
        # lstm_all_dense
        self.lstm_all_dense=Dense(units = self.output_size)

    def train_step(self, data):
        x, y =data

        num_choice=3
        num_layers=6
        # conv size choice
        short_conv_choice = list(np.random.randint(num_choice, size=num_layers*self.lstm_seq_len))
        att_conv_choice = list(np.random.randint(num_choice, size=num_layers*self.att_lstm_num*self.att_lstm_seq_len))
        # pooling choice: [2,3,4] pool_size for max/avg pooling and no pooling: 3*2+1 choices
        short_pooling_choice = list(np.random.randint(num_choice*2+1, size=num_layers*self.lstm_seq_len))
        att_pooling_choice = list(np.random.randint(num_choice*2+1, size=num_layers*self.att_lstm_num*self.att_lstm_seq_len))
        # relu choice: relu, relu6, prelu
        short_relu_choice = list(np.random.randint(num_choice, size=num_layers*self.lstm_seq_len))
        att_relu_choice = list(np.random.randint(num_choice, size=num_layers*self.att_lstm_num*self.att_lstm_seq_len))
        self.choice=[short_conv_choice, att_conv_choice, short_pooling_choice, att_pooling_choice, short_relu_choice, att_relu_choice]

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        # gradients = tape.gradient(loss, trainable_vars)
        gradients = tape.gradient(loss, self.trainable_variables, 
                unconnected_gradients=tf.UnconnectedGradients.ZERO)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def set_choice(self, choice):
        self.choice=choice

    # def predict_step(self, data):
    #     # Unpack the data
    #     x = data
    #     print("len of data: ", len(data))
    #     # Compute predictions
    #     y_pred = self(x, training=False)
    #     return y_pred

    def call(self, input):
        
        flatten_att_nbhd_inputs, flatten_att_flow_inputs, att_lstm_inputs, nbhd_inputs, flow_inputs, [lstm_inputs,], weather, att_weather = input

        att_nbhd_inputs = []
        att_flow_inputs = []
        for att in range(self.att_lstm_num):
            att_nbhd_inputs.append(flatten_att_nbhd_inputs[att*self.att_lstm_seq_len:(att+1)*self.att_lstm_seq_len])
            att_flow_inputs.append(flatten_att_flow_inputs[att*self.att_lstm_seq_len:(att+1)*self.att_lstm_seq_len])

        # print("choice:", self.choice)

        #short-term part

        #1st level gate
        #nbhd cnn
        nbhd_convs = [self.nbhd_convs_list_time0[self.choice[0][(self.lstm_seq_len-1)]](nbhd_inputs[ts]) for ts in range(self.lstm_seq_len)]
        # nbhd pooling
        nbhd_convs = [self.nbhd_pooling_list_time0[self.choice[2][self.lstm_seq_len-1]](nbhd_convs[ts]) \
                        if self.choice[2][self.lstm_seq_len-1] < len(self.nbhd_pooling_list_time0) else nbhd_convs[ts] for ts in range(self.lstm_seq_len)]
        nbhd_convs = [self.nbhd_relu_list_time0[self.choice[4][self.lstm_seq_len-1]](nbhd_convs[ts]) for ts in range(self.lstm_seq_len)]
        #flow cnn
        flow_convs = [self.flow_convs_list_time0[self.choice[0][(self.lstm_seq_len-1)+1]](flow_inputs[ts]) for ts in range(self.lstm_seq_len)]
        # flow pooling
        flow_convs = [self.flow_pooling_list_time0[self.choice[2][(self.lstm_seq_len-1)+1]](flow_convs[ts]) \
                        if self.choice[2][(self.lstm_seq_len-1)+1] < len(self.flow_pooling_list_time0) else flow_convs[ts] for ts in range(self.lstm_seq_len)]
        flow_convs = [self.flow_relu_list_time0[self.choice[4][(self.lstm_seq_len-1)+1]](flow_convs[ts]) for ts in range(self.lstm_seq_len)]
        #flow gate
        flow_gates = [Activation("sigmoid", name = "flow_gate0_{0}".format(ts+1))(flow_convs[ts]) for ts in range(self.lstm_seq_len)]
        nbhd_convs = [keras.layers.Multiply()([nbhd_convs[ts], flow_gates[ts]]) for ts in range(self.lstm_seq_len)]


        #2nd level gate
        nbhd_convs = [self.nbhd_convs_list_time1[self.choice[0][(self.lstm_seq_len-1)+2]](nbhd_inputs[ts]) for ts in range(self.lstm_seq_len)]
        nbhd_convs = [self.nbhd_pooling_list_time1[self.choice[2][(self.lstm_seq_len-1)+2]](nbhd_convs[ts]) \
                        if self.choice[2][(self.lstm_seq_len-1)+2] < len(self.nbhd_pooling_list_time1) else nbhd_convs[ts] for ts in range(self.lstm_seq_len)]
        nbhd_convs = [self.nbhd_relu_list_time1[self.choice[4][(self.lstm_seq_len-1)+2]](nbhd_convs[ts]) for ts in range(self.lstm_seq_len)]
        flow_convs = [self.flow_convs_list_time1[self.choice[0][(self.lstm_seq_len-1)+3]](flow_inputs[ts]) for ts in range(self.lstm_seq_len)]
        flow_convs = [self.flow_pooling_list_time1[self.choice[2][(self.lstm_seq_len-1)+3]](flow_convs[ts]) \
                        if self.choice[2][(self.lstm_seq_len-1)+3] < len(self.flow_pooling_list_time1) else flow_convs[ts] for ts in range(self.lstm_seq_len)]
        flow_convs = [self.flow_relu_list_time1[self.choice[2][(self.lstm_seq_len-1)+3]](flow_convs[ts]) for ts in range(self.lstm_seq_len)]
        flow_gates = [Activation("sigmoid", name = "flow_gate1_{0}".format(ts+1))(flow_convs[ts]) for ts in range(self.lstm_seq_len)]
        nbhd_convs = [keras.layers.Multiply()([nbhd_convs[ts], flow_gates[ts]]) for ts in range(self.lstm_seq_len)]

        #3rd level gate
        nbhd_convs = [self.nbhd_convs_list_time2[self.choice[0][(self.lstm_seq_len-1)+4]](nbhd_inputs[ts]) for ts in range(self.lstm_seq_len)]
        nbhd_convs = [self.nbhd_pooling_list_time2[self.choice[2][(self.lstm_seq_len-1)+4]](nbhd_convs[ts]) \
                        if self.choice[2][(self.lstm_seq_len-1)+4] < len(self.nbhd_pooling_list_time2) else nbhd_convs[ts] for ts in range(self.lstm_seq_len)]
        nbhd_convs = [self.nbhd_relu_list_time2[self.choice[4][(self.lstm_seq_len-1)+4]](nbhd_convs[ts]) for ts in range(self.lstm_seq_len)]
        flow_convs = [self.flow_convs_list_time2[self.choice[0][(self.lstm_seq_len-1)+5]](flow_inputs[ts]) for ts in range(self.lstm_seq_len)]
        flow_convs = [self.flow_pooling_list_time2[self.choice[2][(self.lstm_seq_len-1)+5]](flow_convs[ts]) \
                        if self.choice[2][(self.lstm_seq_len-1)+5] < len(self.flow_pooling_list_time2) else flow_convs[ts] for ts in range(self.lstm_seq_len)]
        flow_convs = [self.flow_relu_list_time2[self.choice[4][(self.lstm_seq_len-1)+5]](flow_convs[ts]) for ts in range(self.lstm_seq_len)]
        flow_gates = [Activation("sigmoid", name = "flow_gate2_{0}".format(ts+1))(flow_convs[ts]) for ts in range(self.lstm_seq_len)]
        nbhd_convs = [keras.layers.Multiply()([nbhd_convs[ts], flow_gates[ts]]) for ts in range(self.lstm_seq_len)]


        #dense part
        nbhd_vecs = [Flatten(name = "nbhd_flatten_time_{0}".format(ts+1))(nbhd_convs[ts]) for ts in range(self.lstm_seq_len)]
        nbhd_vecs = [self.short_term_dense_list[ts](nbhd_vecs[ts]) for ts in range(self.lstm_seq_len)]
        nbhd_vecs = [Activation("relu", name = "nbhd_dense_activation_time_{0}".format(ts+1))(nbhd_vecs[ts]) for ts in range(self.lstm_seq_len)]

        # should weather run FC???

        #feature concatenate
        nbhd_vec = Concatenate(axis=-1)(nbhd_vecs)
        nbhd_vec = Reshape(target_shape = (self.lstm_seq_len, self.cnn_flat_size))(nbhd_vec)
        lstm_input = Concatenate(axis=-1)([lstm_inputs, nbhd_vec, weather])

        #lstm
        lstm = self.short_term_lstm(lstm_input)

        #attention part
        index_num = (self.att_lstm_num - 1)
        index_len = (self.att_lstm_seq_len - 1)
        att_nbhd_convs = [[self.att_nbhd_convs_list_time0[att][ts][self.choice[1][index_num*att + index_len*ts]](att_nbhd_inputs[att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[self.att_nbhd_pooling_list_time0[att][ts][self.choice[3][index_num*att + index_len*ts]](att_nbhd_convs[att][ts]) \
                            if self.choice[3][index_num*att + index_len*ts] < len(self.att_nbhd_pooling_list_time0[att]) else att_nbhd_convs[att][ts] \
                            for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[self.att_nbhd_relu_list_time0[att][ts][self.choice[5][index_num*att+index_len*ts]](att_nbhd_convs[att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.att_flow_convs_list_time0[att][ts][self.choice[1][index_num*att + index_len*ts + 1]](att_flow_inputs[att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.att_flow_pooling_list_time0[att][ts][self.choice[3][index_num*att + index_len*ts + 1]](att_flow_convs[att][ts]) \
                            if self.choice[3][index_num*att + index_len*ts + 1] < len(self.att_flow_pooling_list_time0[att]) else att_flow_convs[att][ts] \
                            for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.att_flow_relu_list_time0[att][ts][self.choice[5][index_num*att+index_len*ts+1]](att_flow_convs[att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_gates = [[Activation("sigmoid", name = "att_flow_gate0_{0}_{1}".format(att+1, ts+1))(att_flow_convs[att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[keras.layers.Multiply()([att_nbhd_convs[att][ts], att_flow_gates[att][ts]]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]

        att_nbhd_convs = [[self.att_nbhd_convs_list_time1[att][ts][self.choice[1][index_num*att + index_len*ts + 2]](att_nbhd_convs[att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[self.att_nbhd_pooling_list_time1[att][ts][self.choice[3][index_num*att + index_len*ts + 2]](att_nbhd_convs[att][ts]) \
                            if self.choice[3][index_num*att + index_len*ts + 2] < len(self.att_nbhd_pooling_list_time1[att]) else att_nbhd_convs[att][ts] \
                            for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[self.att_nbhd_relu_list_time1[att][ts][self.choice[5][index_num*att+index_len*ts+2]](att_nbhd_convs[att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.att_flow_convs_list_time1[att][ts][self.choice[1][index_num*att + index_len*ts + 3]](att_flow_convs[att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.att_flow_pooling_list_time1[att][ts][self.choice[3][index_num*att + index_len*ts + 3]](att_flow_convs[att][ts]) \
                            if self.choice[3][index_num*att + index_len*ts + 3] < len(self.att_flow_pooling_list_time1[att]) else att_flow_convs[att][ts] \
                            for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.att_flow_relu_list_time1[att][ts][self.choice[5][index_num*att+index_len*ts+3]](att_flow_convs[att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_gates = [[Activation("sigmoid", name = "att_flow_gate1_{0}_{1}".format(att+1, ts+1))(att_flow_convs[att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[keras.layers.Multiply()([att_nbhd_convs[att][ts], att_flow_gates[att][ts]]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]

        att_nbhd_convs = [[self.att_nbhd_convs_list_time2[att][ts][self.choice[1][index_num*att + index_len*ts + 4]](att_nbhd_convs[att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[self.att_nbhd_pooling_list_time2[att][ts][self.choice[3][index_num*att + index_len*ts + 4]](att_nbhd_convs[att][ts]) \
                            if self.choice[3][index_num*att + index_len*ts + 4] < len(self.att_nbhd_pooling_list_time2[att]) else att_nbhd_convs[att][ts] \
                            for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[self.att_nbhd_relu_list_time2[att][ts][self.choice[5][index_num*att+index_len*ts+4]](att_nbhd_convs[att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.att_flow_convs_list_time2[att][ts][self.choice[1][index_num*att + index_len*ts + 5]](att_flow_convs[att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.att_flow_pooling_list_time2[att][ts][self.choice[3][index_num*att + index_len*ts + 5]](att_flow_convs[att][ts]) \
                            if self.choice[3][index_num*att + index_len*ts + 5] < len(self.att_flow_pooling_list_time2[att]) else att_flow_convs[att][ts] \
                            for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.att_flow_relu_list_time2[att][ts][self.choice[5][index_num*att+index_len*ts+5]](att_flow_convs[att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_gates = [[Activation("sigmoid", name = "att_flow_gate2_{0}_{1}".format(att+1, ts+1))(att_flow_convs[att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[keras.layers.Multiply()([att_nbhd_convs[att][ts], att_flow_gates[att][ts]]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]

        att_nbhd_vecs = [[Flatten(name = "att_nbhd_flatten_time_{0}_{1}".format(att+1,ts+1))(att_nbhd_convs[att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_vecs = [[self.att_dense_list[att][ts](att_nbhd_vecs[att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_vecs = [[Activation("relu", name = "att_nbhd_dense_activation_time_{0}_{1}".format(att+1,ts+1))(att_nbhd_vecs[att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]


        att_nbhd_vec = [Concatenate(axis=-1)(att_nbhd_vecs[att]) for att in range(self.att_lstm_num)]
        att_nbhd_vec = [Reshape(target_shape = (self.att_lstm_seq_len, self.cnn_flat_size))(att_nbhd_vec[att]) for att in range(self.att_lstm_num)]
        att_lstm_input = [Concatenate(axis=-1)([att_lstm_inputs[att], att_nbhd_vec[att], att_weather[att]]) for att in range(self.att_lstm_num)]

        att_lstms = [self.att_lstm_list[att](att_lstm_input[att]) for att in range(self.att_lstm_num)]

        #compare
        att_low_level=[attention.Attention(method='cba')([att_lstms[att], lstm]) for att in range(self.att_lstm_num)]
        att_low_level=Concatenate(axis=-1)(att_low_level)
        att_low_level=Reshape(target_shape=(self.att_lstm_num, self.lstm_out_size))(att_low_level)


        att_high_level = self.high_level_lstm(att_low_level)

        lstm_all = Concatenate(axis=-1)([att_high_level, lstm])
        # lstm_all = Dropout(rate = .3)(lstm_all)
        lstm_all = self.lstm_all_dense(lstm_all)
        pred_volume = Activation('tanh')(lstm_all)

        return pred_volume


class STDN_Network(keras.Model):
    def __init__(self,choice, att_lstm_num, att_lstm_seq_len, lstm_seq_len, feature_vec_len, cnn_flat_size = 128, lstm_out_size = 128,\
    nbhd_size = 3, nbhd_type = 2, map_x_num = 10, map_y_num = 20, flow_type = 4, output_shape = 2):
        super(STDN_Network, self).__init__()

        self.choice = choice
        self.att_lstm_num = att_lstm_num
        self.att_lstm_seq_len = att_lstm_seq_len
        self.lstm_seq_len = lstm_seq_len
        self.feature_vec_len = feature_vec_len
        self.cnn_flat_size = cnn_flat_size
        self.lstm_out_size = lstm_out_size
        self.nbhd_size = nbhd_size
        self.nbhd_type = nbhd_type
        self.map_x_num = map_x_num
        self.map_y_num = map_y_num
        self.flow_type = flow_type
        self.output_size = output_shape

        self.level=3
        self.kernel_list=[1,2,3]
        self.pool_size_list=[2,3,4]
        self.nbhd_channel=[64,64,64]

        # nbhd / flow layer in the short-term part
        self.nbhd_convs_list_time0=[]
        self.nbhd_convs_list_time1=[]
        self.nbhd_convs_list_time2=[]
        self.flow_convs_list_time0=[]
        self.flow_convs_list_time1=[]
        self.flow_convs_list_time2=[]
        for ts in range(self.lstm_seq_len):
            index = (self.lstm_seq_len-1)
            self.nbhd_convs_list_time0.append(Conv2D(filters = 64, kernel_size = (self.kernel_list[choice[0][index*ts]], self.kernel_list[choice[0][index*ts]]), \
                                                padding="same", name = "nbhd_convs_time0_{0}".format(ts+1)))
            self.nbhd_convs_list_time1.append(Conv2D(filters = 64, kernel_size = (self.kernel_list[choice[0][index*ts+1]], self.kernel_list[choice[0][index*ts+1]]), \
                                                padding="same", name = "nbhd_convs_time1_{0}".format(ts+1)))
            self.nbhd_convs_list_time2.append(Conv2D(filters = 64, kernel_size = (self.kernel_list[choice[0][index*ts+2]], self.kernel_list[choice[0][index*ts+2]]), \
                                                padding="same", name = "nbhd_convs_time2_{0}".format(ts+1)))
            self.flow_convs_list_time0.append(Conv2D(filters = 64, kernel_size = (self.kernel_list[choice[0][index*ts+3]], self.kernel_list[choice[0][index*ts+3]]), \
                                                padding="same", name = "flow_convs_time0_{0}".format(ts+1)))
            self.flow_convs_list_time1.append(Conv2D(filters = 64, kernel_size = (self.kernel_list[choice[0][index*ts+4]], self.kernel_list[choice[0][index*ts+4]]), \
                                                padding="same", name = "flow_convs_time1_{0}".format(ts+1)))
            self.flow_convs_list_time2.append(Conv2D(filters = 64, kernel_size = (self.kernel_list[choice[0][index*ts+5]], self.kernel_list[choice[0][index*ts+5]]), \
                                                padding="same", name = "flow_convs_time2_{0}".format(ts+1)))

        # nbhd / flow relu activation function in short-term part
        self.nbhd_relu_list_time0=[]
        self.nbhd_relu_list_time1=[]
        self.nbhd_relu_list_time2=[]
        self.flow_relu_list_time0=[]
        self.flow_relu_list_time1=[]
        self.flow_relu_list_time2=[]
        for ts in range(self.lstm_seq_len):
            self.nbhd_relu_list_time0.append(ReLU(name = "nbhd_relu_time0_{0}".format(ts+1)))
            self.nbhd_relu_list_time0.append(ReLU(max_value = 6.0, name = "nbhd_relu6_time0_{0}".format(ts+1)))
            self.nbhd_relu_list_time0.append(PReLU(name = "nbhd_prelu_time0_{0}".format(ts+1)))
            self.nbhd_relu_list_time1.append(ReLU(name = "nbhd_relu_time1_{0}".format(ts+1)))
            self.nbhd_relu_list_time1.append(ReLU(max_value = 6.0, name = "nbhd_relu6_time1_{0}".format(ts+1)))
            self.nbhd_relu_list_time1.append(PReLU(name = "nbhd_prelu_time1_{0}".format(ts+1)))
            self.nbhd_relu_list_time2.append(ReLU(name = "nbhd_relu_time2_{0}".format(ts+1)))
            self.nbhd_relu_list_time2.append(ReLU(max_value = 6.0, name = "nbhd_relu6_time2_{0}".format(ts+1)))
            self.nbhd_relu_list_time2.append(PReLU(name = "nbhd_prelu_time2_{0}".format(ts+1)))
            self.flow_relu_list_time0.append(ReLU(name = "flow_relu_time0_{0}".format(ts+1)))
            self.flow_relu_list_time0.append(ReLU(max_value = 6.0, name = "flow_relu6_time0_{0}".format(ts+1)))
            self.flow_relu_list_time0.append(PReLU(name = "flow_prelu_time0_{0}".format(ts+1)))
            self.flow_relu_list_time1.append(ReLU(name = "flow_relu_time1_{0}".format(ts+1)))
            self.flow_relu_list_time1.append(ReLU(max_value = 6.0, name = "flow_relu6_time1_{0}".format(ts+1)))
            self.flow_relu_list_time1.append(PReLU(name = "flow_prelu_time1_{0}".format(ts+1)))
            self.flow_relu_list_time2.append(ReLU(name = "flow_relu_time2_{0}".format(ts+1)))
            self.flow_relu_list_time2.append(ReLU(max_value = 6.0, name = "flow_relu6_time2_{0}".format(ts+1)))
            self.flow_relu_list_time2.append(PReLU(name = "flow_prelu_time2_{0}".format(ts+1)))


        # nbhd / flow pooling layer in the short-term part
        self.nbhd_pooling_list_time0=[]
        self.nbhd_pooling_list_time1=[]
        self.nbhd_pooling_list_time2=[]
        self.flow_pooling_list_time0=[]
        self.flow_pooling_list_time1=[]
        self.flow_pooling_list_time2=[]
        for ts in range(self.lstm_seq_len):
            for pooling_size in self.pool_size_list:
                self.nbhd_pooling_list_time0.append(MaxPooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                    name = "nbhd_max_pooling_time0_{0}".format(ts+1)))
                self.nbhd_pooling_list_time0.append(AveragePooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                    name = "nbhd_avg_pooling_time0_{0}".format(ts+1)))
                self.nbhd_pooling_list_time1.append(MaxPooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                    name = "nbhd_max_pooling_time1_{0}".format(ts+1)))
                self.nbhd_pooling_list_time1.append(AveragePooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                    name = "nbhd_avg_pooling_time1_{0}".format(ts+1)))
                self.nbhd_pooling_list_time2.append(MaxPooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                    name = "nbhd_max_pooling_time0_{0}".format(ts+1)))
                self.nbhd_pooling_list_time2.append(AveragePooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                    name = "nbhd_avg_pooling_time0_{0}".format(ts+1)))
                self.flow_pooling_list_time0.append(MaxPooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                    name = "flow_max_pooling_time0_{0}".format(ts+1)))
                self.flow_pooling_list_time0.append(AveragePooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                    name = "flow_avg_pooling_time0_{0}".format(ts+1)))
                self.flow_pooling_list_time1.append(MaxPooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                    name = "flow_max_pooling_time1_{0}".format(ts+1)))
                self.flow_pooling_list_time1.append(AveragePooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                    name = "flow_avg_pooling_time1_{0}".format(ts+1)))
                self.flow_pooling_list_time2.append(MaxPooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                    name = "flow_max_pooling_time2_{0}".format(ts+1)))
                self.flow_pooling_list_time2.append(AveragePooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                    name = "flow_avg_pooling_time2_{0}".format(ts+1)))

        # dense layer in the short-term part
        self.short_term_dense_list=[]
        for ts in range(self.lstm_seq_len):
            self.short_term_dense_list.append(Dense(units = self.cnn_flat_size, name = "nbhd_dense_time_{0}".format(ts+1)))

        # lstm layer in short-term part
        self.short_term_lstm=LSTM(units=self.lstm_out_size, return_sequences=False, dropout=0.1, recurrent_dropout=0)

        # nbhd / flow conv layers in the att portion
        self.att_nbhd_convs_list_time0=[]
        self.att_nbhd_convs_list_time1=[]
        self.att_nbhd_convs_list_time2=[]
        self.att_flow_convs_list_time0=[]
        self.att_flow_convs_list_time1=[]
        self.att_flow_convs_list_time2=[]
        for att in range(self.att_lstm_num):
            self.att_nbhd_convs_list_time0.append([])
            self.att_nbhd_convs_list_time1.append([])
            self.att_nbhd_convs_list_time2.append([])
            self.att_flow_convs_list_time0.append([])
            self.att_flow_convs_list_time1.append([])
            self.att_flow_convs_list_time2.append([])
            for ts in range(self.att_lstm_seq_len):
                index_num = (self.att_lstm_num - 1)
                index_len = (self.att_lstm_seq_len - 1)
                self.att_nbhd_convs_list_time0[att].append(Conv2D(filters = 64, kernel_size = (self.kernel_list[choice[1][index_num*att+index_len*ts]], \
                                            self.kernel_list[choice[1][index_num*att+index_len*ts]]), padding="same", name = "att_nbhd_convs_time0_{0}_{1}".format(att+1,ts+1)))
                self.att_nbhd_convs_list_time1[att].append(Conv2D(filters = 64, kernel_size = (self.kernel_list[choice[1][index_num*att+index_len*ts+1]], \
                                            self.kernel_list[choice[1][index_num*att+index_len*ts+1]]), padding="same", name = "att_nbhd_convs_time1_{0}_{1}".format(att+1,ts+1)))
                self.att_nbhd_convs_list_time2[att].append(Conv2D(filters = 64, kernel_size = (self.kernel_list[choice[1][index_num*att+index_len*ts+2]], \
                                            self.kernel_list[choice[1][index_num*att+index_len*ts]]), padding="same", name = "att_nbhd_convs_time2_{0}_{1}".format(att+1,ts+1)))
                self.att_flow_convs_list_time0[att].append(Conv2D(filters = 64, kernel_size = (self.kernel_list[choice[1][index_num*att+index_len*ts+3]], \
                                            self.kernel_list[choice[1][index_num*att+index_len*ts+3]]), padding="same", name = "att_flow_convs_time0_{0}_{1}".format(att+1,ts+1)))
                self.att_flow_convs_list_time1[att].append(Conv2D(filters = 64, kernel_size = (self.kernel_list[choice[1][index_num*att+index_len*ts+4]], \
                                            self.kernel_list[choice[1][index_num*att+index_len*ts+4]]), padding="same", name = "att_flow_convs_time1_{0}_{1}".format(att+1,ts+1)))
                self.att_flow_convs_list_time2[att].append(Conv2D(filters = 64, kernel_size = (self.kernel_list[choice[1][index_num*att+index_len*ts+5]], \
                                            self.kernel_list[choice[1][index_num*att+index_len*ts+5]]), padding="same", name = "att_flow_convs_time2_{0}_{1}".format(att+1,ts+1)))

        # nbhd / flow pooling layers in the att portion
        self.att_nbhd_pooling_list_time0=[]
        self.att_nbhd_pooling_list_time1=[]
        self.att_nbhd_pooling_list_time2=[]
        self.att_flow_pooling_list_time0=[]
        self.att_flow_pooling_list_time1=[]
        self.att_flow_pooling_list_time2=[]
        for att in range(self.att_lstm_num):
            self.att_nbhd_pooling_list_time0.append([])
            self.att_nbhd_pooling_list_time1.append([])
            self.att_nbhd_pooling_list_time2.append([])
            self.att_flow_pooling_list_time0.append([])
            self.att_flow_pooling_list_time1.append([])
            self.att_flow_pooling_list_time2.append([])
            for ts in range(self.att_lstm_seq_len):
                self.att_nbhd_pooling_list_time0[att].append([])
                self.att_nbhd_pooling_list_time1[att].append([])
                self.att_nbhd_pooling_list_time2[att].append([])
                self.att_flow_pooling_list_time0[att].append([])
                self.att_flow_pooling_list_time1[att].append([])
                self.att_flow_pooling_list_time2[att].append([])
                for pooling_size in self.pool_size_list:
                    self.att_nbhd_pooling_list_time0[att][ts].append(MaxPooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                                        name = "att_nbhd_max_pooling_time0_{0}_{1}".format(att+1, ts+1)))
                    self.att_nbhd_pooling_list_time0[att][ts].append(AveragePooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                                        name = "att_nbhd_avg_pooling_time0_{0}_{1}".format(att+1, ts+1)))
                    self.att_nbhd_pooling_list_time1[att][ts].append(MaxPooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                                        name = "att_nbhd_max_pooling_time1_{0}_{1}".format(att+1, ts+1)))
                    self.att_nbhd_pooling_list_time1[att][ts].append(AveragePooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                                        name = "att_nbhd_avg_pooling_time1_{0}_{1}".format(att+1, ts+1)))
                    self.att_nbhd_pooling_list_time2[att][ts].append(MaxPooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                                        name = "att_nbhd_max_pooling_time2_{0}_{1}".format(att+1, ts+1)))
                    self.att_nbhd_pooling_list_time2[att][ts].append(AveragePooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                                        name = "att_nbhd_avg_pooling_time2_{0}_{1}".format(att+1, ts+1)))
                    self.att_flow_pooling_list_time0[att][ts].append(MaxPooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                                        name = "att_flow_max_pooling_time0_{0}_{1}".format(att+1, ts+1)))
                    self.att_flow_pooling_list_time0[att][ts].append(AveragePooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                                        name = "att_flow_avg_pooling_time0_{0}_{1}".format(att+1, ts+1)))
                    self.att_flow_pooling_list_time1[att][ts].append(MaxPooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                                        name = "att_flow_max_pooling_time1_{0}_{1}".format(att+1, ts+1)))
                    self.att_flow_pooling_list_time1[att][ts].append(AveragePooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                                        name = "att_flow_avg_pooling_time1_{0}_{1}".format(att+1, ts+1)))
                    self.att_flow_pooling_list_time2[att][ts].append(MaxPooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                                        name = "att_flow_max_pooling_time2_{0}_{1}".format(att+1, ts+1)))
                    self.att_flow_pooling_list_time2[att][ts].append(AveragePooling2D(pool_size = (pooling_size, pooling_size), strides = (1,1), padding = 'same', \
                                                                        name = "att_flow_avg_pooling_time2_{0}_{1}".format(att+1, ts+1)))

        # nbhd / flow relu layers in the att portion
        self.att_nbhd_relu_list_time0=[]
        self.att_nbhd_relu_list_time1=[]
        self.att_nbhd_relu_list_time2=[]
        self.att_flow_relu_list_time0=[]
        self.att_flow_relu_list_time1=[]
        self.att_flow_relu_list_time2=[]
        for att in range(self.att_lstm_num):
            self.att_nbhd_relu_list_time0.append([])
            self.att_nbhd_relu_list_time1.append([])
            self.att_nbhd_relu_list_time2.append([])
            self.att_flow_relu_list_time0.append([])
            self.att_flow_relu_list_time1.append([])
            self.att_flow_relu_list_time2.append([])
            for ts in range(self.att_lstm_seq_len):
                self.att_nbhd_relu_list_time0[att].append([])
                self.att_nbhd_relu_list_time1[att].append([])
                self.att_nbhd_relu_list_time2[att].append([])
                self.att_flow_relu_list_time0[att].append([])
                self.att_flow_relu_list_time1[att].append([])
                self.att_flow_relu_list_time2[att].append([])
                self.att_nbhd_relu_list_time0[att][ts].append(ReLU(name = "att_nbhd_relu_time0_{0}_{1}".format(att+1, ts+1)))
                self.att_nbhd_relu_list_time0[att][ts].append(ReLU(max_value = 6.0, name = "att_nbhd_relu6_time0_{0}_{1}".format(att+1, ts+1)))
                self.att_nbhd_relu_list_time0[att][ts].append(PReLU(name = "att_nbhd_prelu_time0_{0}_{1}".format(att+1, ts+1)))
                self.att_nbhd_relu_list_time1[att][ts].append(ReLU(name = "att_nbhd_relu_time1_{0}_{1}".format(att+1, ts+1)))
                self.att_nbhd_relu_list_time1[att][ts].append(ReLU(max_value = 6.0, name = "att_nbhd_relu6_time1_{0}_{1}".format(att+1, ts+1)))
                self.att_nbhd_relu_list_time1[att][ts].append(PReLU(name = "att_nbhd_prelu_time1_{0}_{1}".format(att+1, ts+1)))
                self.att_nbhd_relu_list_time2[att][ts].append(ReLU(name = "att_nbhd_relu_time2_{0}_{1}".format(att+1, ts+1)))
                self.att_nbhd_relu_list_time2[att][ts].append(ReLU(max_value = 6.0, name = "att_nbhd_relu6_time2_{0}_{1}".format(att+1, ts+1)))
                self.att_nbhd_relu_list_time2[att][ts].append(PReLU(name = "att_nbhd_prelu_time2_{0}_{1}".format(att+1, ts+1)))
                self.att_flow_relu_list_time0[att][ts].append(ReLU(name = "att_flow_relu_time0_{0}_{1}".format(att+1, ts+1)))
                self.att_flow_relu_list_time0[att][ts].append(ReLU(max_value = 6.0, name = "att_flow_relu6_time0_{0}_{1}".format(att+1, ts+1)))
                self.att_flow_relu_list_time0[att][ts].append(PReLU(name = "att_flow_prelu_time0_{0}_{1}".format(att+1, ts+1)))
                self.att_flow_relu_list_time1[att][ts].append(ReLU(name = "att_flow_relu_time1_{0}_{1}".format(att+1, ts+1)))
                self.att_flow_relu_list_time1[att][ts].append(ReLU(max_value = 6.0, name = "att_flow_relu6_time1_{0}_{1}".format(att+1, ts+1)))
                self.att_flow_relu_list_time1[att][ts].append(PReLU(name = "att_flow_prelu_time1_{0}_{1}".format(att+1, ts+1)))
                self.att_flow_relu_list_time2[att][ts].append(ReLU(name = "att_flow_relu_time2_{0}_{1}".format(att+1, ts+1)))
                self.att_flow_relu_list_time2[att][ts].append(ReLU(max_value = 6.0, name = "att_flow_relu6_time2_{0}_{1}".format(att+1, ts+1)))
                self.att_flow_relu_list_time2[att][ts].append(PReLU(name = "att_flow_prelu_time2_{0}_{1}".format(att+1, ts+1)))

        # attention part dense list
        self.att_dense_list=[]
        for att in range(self.att_lstm_num):
            self.att_dense_list.append([])
            for ts in range(self.att_lstm_seq_len):
                self.att_dense_list[att].append(Dense(units = self.cnn_flat_size, name = "att_nbhd_dense_time_{0}_{1}".format(att+1,ts+1)))

        # attention part lstm list
        self.att_lstm_list=[]
        for att in range(self.att_lstm_num):
            self.att_lstm_list.append(LSTM(units=self.lstm_out_size, return_sequences=True, dropout=0.1, \
                                        recurrent_dropout=0,recurrent_activation='sigmoid', name="att_lstm_{0}".format(att + 1)))
    
        # high_level_lstm
        self.high_level_lstm=LSTM(units=self.lstm_out_size, return_sequences=False, dropout=0.1, recurrent_dropout=0)
        
        # lstm_all_dense
        self.lstm_all_dense=Dense(units = self.output_size)

    def train_step(self, data):
        x, y =data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        # gradients = tape.gradient(loss, trainable_vars)
        gradients = tape.gradient(loss, self.trainable_variables, 
                unconnected_gradients=tf.UnconnectedGradients.ZERO)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def call(self, input):
        
        flatten_att_nbhd_inputs, flatten_att_flow_inputs, att_lstm_inputs, nbhd_inputs, flow_inputs, [lstm_inputs,], weather, att_weather = input

        att_nbhd_inputs = []
        att_flow_inputs = []
        for att in range(self.att_lstm_num):
            att_nbhd_inputs.append(flatten_att_nbhd_inputs[att*self.att_lstm_seq_len:(att+1)*self.att_lstm_seq_len])
            att_flow_inputs.append(flatten_att_flow_inputs[att*self.att_lstm_seq_len:(att+1)*self.att_lstm_seq_len])

        # print("choice:", self.choice)

        # short-term part
        # 1st level gate
        #nbhd cnn
        nbhd_convs = [self.nbhd_convs_list_time0[ts](nbhd_inputs[ts]) for ts in range(self.lstm_seq_len)]
        nbhd_convs = [self.nbhd_pooling_list_time0[self.choice[2][self.lstm_seq_len-1]](nbhd_convs[ts]) \
                        if self.choice[2][self.lstm_seq_len-1] < len(self.nbhd_pooling_list_time0) else nbhd_convs[ts] for ts in range(self.lstm_seq_len)]
        nbhd_convs = [self.nbhd_relu_list_time0[self.choice[4][self.lstm_seq_len-1]](nbhd_convs[ts]) for ts in range(self.lstm_seq_len)]
        #flow cnn
        flow_convs = [self.flow_convs_list_time0[ts](flow_inputs[ts]) for ts in range(self.lstm_seq_len)]
        flow_convs = [self.flow_pooling_list_time0[self.choice[2][(self.lstm_seq_len-1)+1]](flow_convs[ts]) \
                        if self.choice[2][(self.lstm_seq_len-1)+1] < len(self.flow_pooling_list_time0) else flow_convs[ts] for ts in range(self.lstm_seq_len)]
        flow_convs = [self.flow_relu_list_time0[self.choice[4][(self.lstm_seq_len-1)+1]](flow_convs[ts]) for ts in range(self.lstm_seq_len)]
        #flow gate
        flow_gates = [Activation("sigmoid", name = "flow_gate0_{0}".format(ts+1))(flow_convs[ts]) for ts in range(self.lstm_seq_len)]
        nbhd_convs = [keras.layers.Multiply()([nbhd_convs[ts], flow_gates[ts]]) for ts in range(self.lstm_seq_len)]

        #2nd level gate
        nbhd_convs = [self.nbhd_convs_list_time1[ts](nbhd_inputs[ts]) for ts in range(self.lstm_seq_len)]
        nbhd_convs = [self.nbhd_pooling_list_time1[self.choice[2][(self.lstm_seq_len-1)+2]](nbhd_convs[ts]) \
                        if self.choice[2][(self.lstm_seq_len-1)+2] < len(self.nbhd_pooling_list_time1) else nbhd_convs[ts] for ts in range(self.lstm_seq_len)]
        nbhd_convs = [self.nbhd_relu_list_time1[self.choice[4][(self.lstm_seq_len-1)+2]](nbhd_convs[ts]) for ts in range(self.lstm_seq_len)]
        flow_convs = [self.flow_convs_list_time1[ts](flow_inputs[ts]) for ts in range(self.lstm_seq_len)]
        flow_convs = [self.flow_pooling_list_time1[self.choice[2][(self.lstm_seq_len-1)+3]](flow_convs[ts]) \
                        if self.choice[2][(self.lstm_seq_len-1)+3] < len(self.flow_pooling_list_time1) else flow_convs[ts] for ts in range(self.lstm_seq_len)]
        flow_convs = [self.flow_relu_list_time1[self.choice[2][(self.lstm_seq_len-1)+3]](flow_convs[ts]) for ts in range(self.lstm_seq_len)]
        flow_gates = [Activation("sigmoid", name = "flow_gate1_{0}".format(ts+1))(flow_convs[ts]) for ts in range(self.lstm_seq_len)]
        nbhd_convs = [keras.layers.Multiply()([nbhd_convs[ts], flow_gates[ts]]) for ts in range(self.lstm_seq_len)]

        #3rd level gate
        nbhd_convs = [self.nbhd_convs_list_time2[ts](nbhd_inputs[ts]) for ts in range(self.lstm_seq_len)]
        nbhd_convs = [self.nbhd_pooling_list_time2[self.choice[2][(self.lstm_seq_len-1)+4]](nbhd_convs[ts]) \
                        if self.choice[2][(self.lstm_seq_len-1)+4] < len(self.nbhd_pooling_list_time2) else nbhd_convs[ts] for ts in range(self.lstm_seq_len)]
        nbhd_convs = [self.nbhd_relu_list_time2[self.choice[4][(self.lstm_seq_len-1)+4]](nbhd_convs[ts]) for ts in range(self.lstm_seq_len)]
        flow_convs = [self.flow_convs_list_time2[ts](flow_inputs[ts]) for ts in range(self.lstm_seq_len)]
        flow_convs = [self.flow_pooling_list_time2[self.choice[2][(self.lstm_seq_len-1)+5]](flow_convs[ts]) \
                        if self.choice[2][(self.lstm_seq_len-1)+5] < len(self.flow_pooling_list_time2) else flow_convs[ts] for ts in range(self.lstm_seq_len)]
        flow_convs = [self.flow_relu_list_time2[self.choice[4][(self.lstm_seq_len-1)+5]](flow_convs[ts]) for ts in range(self.lstm_seq_len)]
        flow_gates = [Activation("sigmoid", name = "flow_gate2_{0}".format(ts+1))(flow_convs[ts]) for ts in range(self.lstm_seq_len)]
        nbhd_convs = [keras.layers.Multiply()([nbhd_convs[ts], flow_gates[ts]]) for ts in range(self.lstm_seq_len)]

        #dense part
        nbhd_vecs = [Flatten(name = "nbhd_flatten_time_{0}".format(ts+1))(nbhd_convs[ts]) for ts in range(self.lstm_seq_len)]
        nbhd_vecs = [self.short_term_dense_list[ts](nbhd_vecs[ts]) for ts in range(self.lstm_seq_len)]
        nbhd_vecs = [Activation("relu", name = "nbhd_dense_activation_time_{0}".format(ts+1))(nbhd_vecs[ts]) for ts in range(self.lstm_seq_len)]

        #feature concatenate
        nbhd_vec = Concatenate(axis=-1)(nbhd_vecs)
        nbhd_vec = Reshape(target_shape = (self.lstm_seq_len, self.cnn_flat_size))(nbhd_vec)
        lstm_input = Concatenate(axis=-1)([lstm_inputs, nbhd_vec, weather])

        #lstm
        lstm = self.short_term_lstm(lstm_input)

        #attention part
        index_num = (self.att_lstm_num - 1)
        index_len = (self.att_lstm_seq_len - 1)
        att_nbhd_convs = [[self.att_nbhd_convs_list_time0[att][ts](att_nbhd_inputs[att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[self.att_nbhd_pooling_list_time0[att][ts][self.choice[3][index_num*att + index_len*ts]](att_nbhd_convs[att][ts]) \
                            if self.choice[3][index_num*att + index_len*ts] < len(self.att_nbhd_pooling_list_time0[att]) else att_nbhd_convs[att][ts] \
                            for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[self.att_nbhd_relu_list_time0[att][ts][self.choice[5][index_num*att+index_len*ts]](att_nbhd_convs[att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.att_flow_convs_list_time0[att][ts](att_flow_inputs[att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.att_flow_pooling_list_time0[att][ts][self.choice[3][index_num*att + index_len*ts + 1]](att_flow_convs[att][ts]) \
                            if self.choice[3][index_num*att + index_len*ts + 1] < len(self.att_flow_pooling_list_time0[att]) else att_flow_convs[att][ts] \
                            for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.att_flow_relu_list_time0[att][ts][self.choice[5][index_num*att+index_len*ts+1]](att_flow_convs[att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_gates = [[Activation("sigmoid", name = "att_flow_gate0_{0}_{1}".format(att+1, ts+1))(att_flow_convs[att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[keras.layers.Multiply()([att_nbhd_convs[att][ts], att_flow_gates[att][ts]]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]

        att_nbhd_convs = [[self.att_nbhd_convs_list_time1[att][ts](att_nbhd_convs[att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[self.att_nbhd_pooling_list_time1[att][ts][self.choice[3][index_num*att + index_len*ts + 2]](att_nbhd_convs[att][ts]) \
                            if self.choice[3][index_num*att + index_len*ts + 2] < len(self.att_nbhd_pooling_list_time1[att]) else att_nbhd_convs[att][ts] \
                            for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[self.att_nbhd_relu_list_time1[att][ts][self.choice[5][index_num*att+index_len*ts+2]](att_nbhd_convs[att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.att_flow_convs_list_time1[att][ts](att_flow_convs[att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.att_flow_pooling_list_time1[att][ts][self.choice[3][index_num*att + index_len*ts + 3]](att_flow_convs[att][ts]) \
                            if self.choice[3][index_num*att + index_len*ts + 3] < len(self.att_flow_pooling_list_time1[att]) else att_flow_convs[att][ts] \
                            for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.att_flow_relu_list_time1[att][ts][self.choice[5][index_num*att+index_len*ts+3]](att_flow_convs[att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_gates = [[Activation("sigmoid", name = "att_flow_gate1_{0}_{1}".format(att+1, ts+1))(att_flow_convs[att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[keras.layers.Multiply()([att_nbhd_convs[att][ts], att_flow_gates[att][ts]]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]

        att_nbhd_convs = [[self.att_nbhd_convs_list_time2[att][ts](att_nbhd_convs[att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[self.att_nbhd_pooling_list_time2[att][ts][self.choice[3][index_num*att + index_len*ts + 4]](att_nbhd_convs[att][ts]) \
                            if self.choice[3][index_num*att + index_len*ts + 4] < len(self.att_nbhd_pooling_list_time2[att]) else att_nbhd_convs[att][ts] \
                            for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[self.att_nbhd_relu_list_time2[att][ts][self.choice[5][index_num*att+index_len*ts+4]](att_nbhd_convs[att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.att_flow_convs_list_time2[att][ts](att_flow_convs[att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.att_flow_pooling_list_time2[att][ts][self.choice[3][index_num*att + index_len*ts + 5]](att_flow_convs[att][ts]) \
                            if self.choice[3][index_num*att + index_len*ts + 5] < len(self.att_flow_pooling_list_time2[att]) else att_flow_convs[att][ts] \
                            for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.att_flow_relu_list_time2[att][ts][self.choice[5][index_num*att+index_len*ts+5]](att_flow_convs[att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_gates = [[Activation("sigmoid", name = "att_flow_gate2_{0}_{1}".format(att+1, ts+1))(att_flow_convs[att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[keras.layers.Multiply()([att_nbhd_convs[att][ts], att_flow_gates[att][ts]]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]

        att_nbhd_vecs = [[Flatten(name = "att_nbhd_flatten_time_{0}_{1}".format(att+1,ts+1))(att_nbhd_convs[att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_vecs = [[self.att_dense_list[att][ts](att_nbhd_vecs[att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_vecs = [[Activation("relu", name = "att_nbhd_dense_activation_time_{0}_{1}".format(att+1,ts+1))(att_nbhd_vecs[att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]


        att_nbhd_vec = [Concatenate(axis=-1)(att_nbhd_vecs[att]) for att in range(self.att_lstm_num)]
        att_nbhd_vec = [Reshape(target_shape = (self.att_lstm_seq_len, self.cnn_flat_size))(att_nbhd_vec[att]) for att in range(self.att_lstm_num)]
        att_lstm_input = [Concatenate(axis=-1)([att_lstm_inputs[att], att_nbhd_vec[att], att_weather[att]]) for att in range(self.att_lstm_num)]

        att_lstms = [self.att_lstm_list[att](att_lstm_input[att]) for att in range(self.att_lstm_num)]

        #compare
        att_low_level=[attention.Attention(method='cba')([att_lstms[att], lstm]) for att in range(self.att_lstm_num)]
        att_low_level=Concatenate(axis=-1)(att_low_level)
        att_low_level=Reshape(target_shape=(self.att_lstm_num, self.lstm_out_size))(att_low_level)


        att_high_level = self.high_level_lstm(att_low_level)

        lstm_all = Concatenate(axis=-1)([att_high_level, lstm])
        # lstm_all = Dropout(rate = .3)(lstm_all)
        lstm_all = self.lstm_all_dense(lstm_all)
        pred_volume = Activation('tanh')(lstm_all)

        return pred_volume