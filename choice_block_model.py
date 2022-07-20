from optparse import check_choice
from turtle import shape
from warnings import filters
import keras, itertools
from random import choice, choices, random
from unicodedata import name
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Model, Sequential, load_model
from keras.layers import Layer, Dense, Activation, ReLU, PReLU, Input, Conv2D, Reshape, Flatten, Dropout, BatchNormalization, Concatenate, LSTM, MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, Callback, ModelCheckpoint
import attention

class Conv_Choice_Block(Layer):
    def __init__(self, layer_name, choice_num = 3, filters = 64, padding = "same"):
        super(Conv_Choice_Block, self).__init__()
        self.choice = -1
        self.layer_name = layer_name
        self.choice_num = choice_num
        self.filters = filters
        self.padding = padding

    def build(self, inputs):
        super(Conv_Choice_Block, self).build(inputs)
        self.conv1 = Conv2D(filters = self.filters, kernel_size = (1, 1), padding = self.padding, name = self.layer_name+"_size_"+str(1))
        self.conv2 = Conv2D(filters = self.filters, kernel_size = (2, 2), padding = self.padding, name = self.layer_name+"_size_"+str(2))
        self.conv3 = Conv2D(filters = self.filters, kernel_size = (3, 3), padding = self.padding, name = self.layer_name+"_size_"+str(3))

    def call(self, inputs, **kwargs):
        if kwargs.get("training", True):
            # randomly choose when training phase
            self.choice = np.random.randint(self.choice_num)
        else:
            # default choice is 2 if no choice given when evaluating
            self.choice = kwargs.get("choice", 2)
            # print("choice from searching phase: ", self.choice)
        self.choice = tf.convert_to_tensor(self.choice)
        output = tf.switch_case(self.choice, {  
                                                0: lambda: self.conv1(inputs),
                                                1: lambda: self.conv2(inputs),
                                                2: lambda: self.conv3(inputs)
                                            })
        return output

class Pooling_Choice_Block(Layer):
    def __init__(self, layer_name, choice_num = 7, padding = "same"):
        super(Pooling_Choice_Block, self).__init__()
        self.choice = -1
        self.layer_name = layer_name
        self.choice_num = choice_num
        self.padding = padding

    def build(self, inputs):
        super(Pooling_Choice_Block, self).build(inputs)
        self.max2 = MaxPooling2D(pool_size = (2, 2), strides = (1, 1), padding = self.padding, name = self.layer_name+"_max_size_"+str(2))
        self.max3 = MaxPooling2D(pool_size = (3, 3), strides = (1, 1), padding = self.padding, name = self.layer_name+"_max_size_"+str(3))
        self.max4 = MaxPooling2D(pool_size = (4, 4), strides = (1, 1), padding = self.padding, name = self.layer_name+"_max_size_"+str(4))
        self.avg2 = AveragePooling2D(pool_size = (2, 2), strides = (1, 1), padding = self.padding, name = self.layer_name+"_avg_size_"+str(2))
        self.avg3 = AveragePooling2D(pool_size = (3, 3), strides = (1, 1), padding = self.padding, name = self.layer_name+"_avg_size_"+str(3))
        self.avg4 = AveragePooling2D(pool_size = (4, 4), strides = (1, 1), padding = self.padding, name = self.layer_name+"_avg_size_"+str(4))

    def call(self, inputs, **kwargs):
        if kwargs.get("training", True):
            self.choice = np.random.randint(self.choice_num)
        else:
            # default not having pooling layer
            self.choice = kwargs.get("choice", 6)
        if self.choice == 6:
            return inputs
        self.choice = tf.convert_to_tensor(self.choice)
        output = tf.switch_case(self.choice, {  
            0: lambda: self.max2(inputs),
            1: lambda: self.max3(inputs),
            2: lambda: self.max4(inputs),
            3: lambda: self.avg2(inputs),
            4: lambda: self.avg3(inputs),
            5: lambda: self.avg4(inputs)
        })
        return output

class Conv_Activ_Choice_Block(Layer):
    def __init__(self, layer_name, choice_num = 3):
        super(Conv_Activ_Choice_Block, self).__init__()
        self.layer_name = layer_name
        self.choice_num = choice_num

    def build(self, inputs):
        super(Conv_Activ_Choice_Block, self).build(inputs)
        self.relu = ReLU(name = self.layer_name+"_relu")
        self.relu6 = ReLU(max_value = 6.0, name = self.layer_name+"_relu")
        self.prelu = PReLU(name = self.layer_name+"_prelu")

    def call(self, inputs, **kwargs):
        if kwargs.get("training", True):
            self.choice = np.random.randint(self.choice_num)
        else:
            # default choice relu layer
            self.choice = kwargs.get("choice", 0)
        self.choice = tf.convert_to_tensor(self.choice)
        output = tf.switch_case(self.choice, {
            0: lambda: self.relu(inputs),
            1: lambda: self.relu6(inputs),
            2: lambda: self.prelu(inputs)
        })
        return output

class Gate_Activ_Choice_Block(Layer):
    def __init__(self, layer_name, choice_num = 3):
        super(Gate_Activ_Choice_Block, self).__init__()
        self.layer_name = layer_name
        self.choice_num = choice_num

    def build(self, inputs):
        super(Gate_Activ_Choice_Block, self).build(inputs)
        # self.sigmoid = Activation("sigmoid", name = self.layer_name+"_sigmoid")
        self.relu6 = ReLU(max_value = 6.0, name = self.layer_name+"_relu6")
        # self.tanh = Activation("tanh", name = self.layer_name+"_tanh")

    def call(self, inputs, **kwargs):
        if kwargs.get("training", True):
            self.choice = np.random.randint(self.choice_num)
        else:
            # default choice sigmoid
            self.choice = kwargs.get("choice", 0)
        self.choice = tf.convert_to_tensor(self.choice)
        output = tf.switch_case(self.choice, {
            0: lambda: Activation("sigmoid", name = self.layer_name+"_sigmoid")(inputs),
            1: lambda: self.relu6(inputs),
            2: lambda: Activation("tanh", name = self.layer_name+"_tanh")(inputs)
        })
        return output


class SAGAN_Suprtnet_Subclass_model(keras.Model):
    def __init__(self, att_lstm_num, att_lstm_seq_len, lstm_seq_len, feature_vec_len, cnn_flat_size, lstm_out_size, output_shape):
        super(SAGAN_Suprtnet_Subclass_model, self).__init__()
        self.gate_level = 3
        self.lstm_seq_len = lstm_seq_len
        self.cnn_flat_size = cnn_flat_size
        self.att_lstm_seq_len = att_lstm_seq_len
        self.att_lstm_num = att_lstm_num
        self.lstm_out_size = lstm_out_size

        # short term choice
        short_conv_choice, short_pooling_choice, short_conv_activ_choice = (np.zeros(shape = (self.gate_level, self.lstm_seq_len, 2), dtype = int) for cnt in range(3))
        short_gate_activ_choice = np.zeros(shape = (self.gate_level, self.lstm_seq_len), dtype = int)

        # long term choice
        long_conv_choice, long_pooling_choice, long_conv_activ_choice = (np.zeros(shape = (self.gate_level, self.att_lstm_num, self.att_lstm_seq_len, 2), dtype = int) for cnt in range(3))
        long_gate_activ_choice = np.zeros(shape = (self.gate_level, self.att_lstm_num, self.att_lstm_seq_len), dtype = int)
        self.nas_choice = [short_conv_choice, short_pooling_choice, short_conv_activ_choice, short_gate_activ_choice, \
            long_conv_choice, long_pooling_choice, long_conv_activ_choice, long_gate_activ_choice]
        """
        Initialize short-term choice block
        layers include:
            1. short-term conv (nbhd / flow): [1, 2, 3] kernel size
            2. short-term pooling (nbhd / flow): [2, 3, 4] pooling size for avg / max pooling & ignore
            3. short-term conv activation  (nbhd / flow): ReLU, PReLU, ReLU6
            4. short-term gate activation: sigmoid, ReLU6, tanh
        """
        # short_conv: gate_level * lstm_seq_len * 2 (nbhd / flow)
        # short_pooling: gate_level * lstm_seq_len * 2 (nbhd / flow)
        # short_conv_activ: gate_level * lstm_seq_len * 2 (nbhd / flow)
        # short_gate_activ: gate_level * lstm_seq_len * 1
        self.short_conv, self.short_pooling, self.short_conv_activ = \
            ([[[] for ts in range(self.lstm_seq_len)] for level in range(self.gate_level)] for cnt in range(3))
        self.short_gate_activ = [[] for level in range(self.gate_level)]

        for args in itertools.product(range(self.gate_level), range(self.lstm_seq_len)):
            level, ts = args[0], args[1]
            self.short_gate_activ[level].append(Gate_Activ_Choice_Block(layer_name = "gate_activ_time{0}_{1}".format(level, ts+1), choice_num = 3))
            for data_type in range(0, 2):
                layer_name = "nbhd" if data_type == 0 else "flow"
                self.short_conv[level][ts].append(Conv_Choice_Block(layer_name = layer_name+"_conv_time{0}_{1}".format(level, ts+1), choice_num = 3))
                self.short_pooling[level][ts].append(Pooling_Choice_Block(layer_name = layer_name+"_pooling_time{0}_{1}".format(level, ts+1), choice_num = 7))
                self.short_conv_activ[level][ts].append(Conv_Activ_Choice_Block(layer_name = layer_name+"_activ_time{0}_{1}".format(level, ts+1), choice_num = 3))

        """
        Initialize other layers in short-term
        including:
            1. short-term poi conv
            2. short-term poi dense
            3. short-term nbhd dense
            4. short-term lstm
        """
        # short_poi_conv: lstm_seq_len * 1
        # short_poi_dense: lstm_seq_len * 1
        # short_nbhd_dense: lstm_seq_len * 1
        # short_lstm: 1
        self.short_poi_conv, self.short_poi_dense, self.short_nbhd_dense, = ([] for cnt in range(3))
        for ts in range(self.lstm_seq_len):
            self.short_poi_conv.append(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same', name = 'poi_conv_time{0}'.format(ts+1)))
            self.short_poi_dense.append(Dense(units = self.cnn_flat_size, name = "poi_dense_time{0}".format(ts+1)))
            self.short_nbhd_dense.append(Dense(units = self.cnn_flat_size, name = "nbhd_dense_time{0}".format(ts+1)))
        self.short_lstm = LSTM(units = self.lstm_out_size, return_sequences = False, dropout = 0.1, recurrent_dropout = 0.1)

        """
        Initialize long-term choice block
        layer include:
            1. long-term conv (nbhd / flow): [1, 2, 3] kernel size
            2. long-term pooling (nbhd / flow): [2, 3, 4] pooling size for avg / max pooling & ignore
            3. long-term conv activation: ReLU, PReLU, ReLU6
            4. long-term gate activation: Sigmoid, ReLU6, tanh
        """
        # long_conv: gate_level * att_lstm_num * att_lstm_seq_len * 2 (nbhd / flow)
        # long_pooling: gate_level * att_lstm_num * att_lstm_seq_len * (nbhd  / flow)
        # long_conv_activ: gate_level * att_lstm_num * att_lstm_seq_len * 2 (nbhd / flow)
        # long_gate_activ: gate_level * att_lstm_num * att_lstm_seq_len * 1
        self.long_conv, self.long_pooling, self.long_conv_activ = \
            ([[[[] for att in range(self.att_lstm_seq_len)] for ts in range(self.att_lstm_num)] for level in range(self.gate_level)] for cnt in range(3))
        self.long_gate_activ = [[[] for att in range(self.att_lstm_num)] for level in range(self.gate_level)]

        for args in itertools.product(range(self.gate_level), range(self.att_lstm_num), range(self.att_lstm_seq_len)):
            level, att, ts = args[0], args[1], args[2]
            self.long_gate_activ[level][att].append(Gate_Activ_Choice_Block(layer_name = "att_gate_activ_time{0}_{1}_{2}".format(level, att+1, ts+1), choice_num = 3))
            for data_type in range(0, 2):
                layer_name = "nbhd" if data_type == 0 else "flow"
                self.long_conv[level][att][ts].append(Conv_Choice_Block(layer_name = "att_"+layer_name+"_conv_time{0}_{1}_{2}".format(level, att+1, ts+1), choice_num = 3))
                self.long_pooling[level][att][ts].append(Pooling_Choice_Block(layer_name = "att_"+layer_name+"_pooling_time{0}_{1}_{2}".format(level, att+1, ts+1), choice_num = 7))
                self.long_conv_activ[level][att][ts].append(Conv_Activ_Choice_Block(layer_name = "att_"+layer_name+"_activ_time{0}_{1}_{2}".format(level, att+1, ts+1), choice_num = 3))

        """
        Initialize other layers in long-term
        including:
            1. long-term poi conv
            2. long-term poi dense
            3. long-term nbhd dense
            4. long-term lstm
            5. long-term attention
        """
        # long_poi_conv: att_lstm_num * att_lstm_seq_len * 1
        # long_poi_dense: att_lstm_num * att_lstm_seq_len * 1
        # long_nbhd_dense: att_lstm_num * att_lstm_seq_len * 1
        # long_lstm: att_lstm_num * 1
        # long_attention: att_lstm_num * 1
        self.long_poi_conv, self.long_poi_dense, self.long_nbhd_dense = \
            ([[] for att in range(self.att_lstm_num)] for cnt in range(3))
        for args in itertools.product(range(self.att_lstm_num), range(self.att_lstm_seq_len)):
            att, ts= args[0], args[1]
            self.long_poi_conv[att].append(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same', name = "att_poi_conv_time{0}_{1}".format(att+1, ts+1)))
            self.long_poi_dense[att].append(Dense(units = self.cnn_flat_size, name = "att_poi_dense_time{0}_{1}".format(att+1, ts+1)))
            self.long_nbhd_dense[att].append(Dense(units = self.cnn_flat_size, name = "att_nbhd_dense_time{0}_{1}".format(att+1, ts+1)))
        self.long_lstm, self.long_attention = ([] for cnt in range(2))
        for att in range(self.att_lstm_num):
            self.long_lstm.append(LSTM(units=self.lstm_out_size, return_sequences=True, dropout=0.1, recurrent_dropout=0.1, name="att_lstm_time_{0}".format(att+1)))
            self.long_attention.append(attention.Attention(method='cba'))

        """
        Initialize layers after long-short-term
        including:
            1. high level lstm
            2. lstm all dense
        """
        # high level lstm: 1
        # lstm all dense: 1
        self.high_level_lstm = LSTM(units=lstm_out_size, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)
        self.lstm_all_dense = Dense(units = output_shape)
    
    def set_choice(self, choice):
        self.nas_choice = choice

    def train_step(self, data):
        x, y = data

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

    @tf.function
    def call(self, inputs):
        flatten_att_nbhd_inputs, flatten_att_flow_inputs, att_lstm_inputs, att_weather, nbhd_inputs, flow_inputs, [lstm_inputs, ], weather, poi_data = inputs

        att_nbhd_inputs = []
        att_flow_inputs = []
        # att_poi_inputs = []
        for att in range(self.att_lstm_num):
            att_nbhd_inputs.append(flatten_att_nbhd_inputs[att*self.att_lstm_seq_len:(att+1)*self.att_lstm_seq_len])
            att_flow_inputs.append(flatten_att_flow_inputs[att*self.att_lstm_seq_len:(att+1)*self.att_lstm_seq_len])
            # att_poi_inputs.append(att_poi_data[att*self.att_lstm_seq_len:(att+1)*self.att_lstm_seq_len])

        # print("choice:", self.choice)
        # 1st level gate
        level = 0
        nbhd_convs = [self.short_conv[level][ts][0](nbhd_inputs[ts], choice = self.nas_choice[0][level][ts][0]) for ts in range(self.lstm_seq_len)]
        nbhd_convs = [self.short_pooling[level][ts][0](nbhd_convs[ts], choice = self.nas_choice[1][level][ts][0]) for ts in range(self.lstm_seq_len)]
        nbhd_convs = [self.short_conv_activ[level][ts][0](nbhd_convs[ts], choice = self.nas_choice[2][level][ts][0]) for ts in range(self.lstm_seq_len)]
        flow_convs = [self.short_conv[level][ts][1](flow_inputs[ts], choice = self.nas_choice[0][level][ts][1]) for ts in range(self.lstm_seq_len)]
        flow_convs = [self.short_pooling[level][ts][1](flow_convs[ts], choice = self.nas_choice[1][level][ts][1]) for ts in range(self.lstm_seq_len)]
        flow_convs = [self.short_conv_activ[level][ts][1](flow_convs[ts], choice = self.nas_choice[2][level][ts][1]) for ts in range(self.lstm_seq_len)]
        flow_gates = [self.short_gate_activ[level][ts](flow_convs[ts], choice = self.nas_choice[3][level][ts]) for ts in range(self.lstm_seq_len)]
        nbhd_convs = [keras.layers.Multiply()([nbhd_convs[ts], flow_gates[ts]]) for ts in range(self.lstm_seq_len)]

        # 2nd level gate
        level = 1
        nbhd_convs = [self.short_conv[level][ts][0](nbhd_convs[ts], choice = self.nas_choice[0][level][ts][0]) for ts in range(self.lstm_seq_len)]
        nbhd_convs = [self.short_pooling[level][ts][0](nbhd_convs[ts], choice = self.nas_choice[1][level][ts][0]) for ts in range(self.lstm_seq_len)]
        nbhd_convs = [self.short_conv_activ[level][ts][0](nbhd_convs[ts], choice = self.nas_choice[2][level][ts][0]) for ts in range(self.lstm_seq_len)]
        flow_convs = [self.short_conv[level][ts][1](flow_inputs[ts], choice = self.nas_choice[0][level][ts][1]) for ts in range(self.lstm_seq_len)]
        flow_convs = [self.short_pooling[level][ts][1](flow_convs[ts], choice = self.nas_choice[1][level][ts][1]) for ts in range(self.lstm_seq_len)]
        flow_convs = [self.short_conv_activ[level][ts][1](flow_convs[ts], choice = self.nas_choice[2][level][ts][1]) for ts in range(self.lstm_seq_len)]
        flow_gates = [self.short_gate_activ[level][ts](flow_convs[ts], choice = self.nas_choice[3][level][ts]) for ts in range(self.lstm_seq_len)]
        nbhd_convs = [keras.layers.Multiply()([nbhd_convs[ts], flow_gates[ts]]) for ts in range(self.lstm_seq_len)]

        # 3rd level gate
        level = 2
        nbhd_convs = [self.short_conv[level][ts][0](nbhd_convs[ts], choice = self.nas_choice[0][level][ts][0]) for ts in range(self.lstm_seq_len)]
        nbhd_convs = [self.short_pooling[level][ts][0](nbhd_convs[ts], choice = self.nas_choice[1][level][ts][0]) for ts in range(self.lstm_seq_len)]
        nbhd_convs = [self.short_conv_activ[level][ts][0](nbhd_convs[ts], choice = self.nas_choice[2][level][ts][0]) for ts in range(self.lstm_seq_len)]
        flow_convs = [self.short_conv[level][ts][1](flow_inputs[ts], choice = self.nas_choice[0][level][ts][1]) for ts in range(self.lstm_seq_len)]
        flow_convs = [self.short_pooling[level][ts][1](flow_convs[ts], choice = self.nas_choice[1][level][ts][1]) for ts in range(self.lstm_seq_len)]
        flow_convs = [self.short_conv_activ[level][ts][1](flow_convs[ts], choice = self.nas_choice[2][level][ts][1]) for ts in range(self.lstm_seq_len)]
        flow_gates = [self.short_gate_activ[level][ts](flow_convs[ts], choice = self.nas_choice[3][level][ts]) for ts in range(self.lstm_seq_len)]
        nbhd_convs = [keras.layers.Multiply()([nbhd_convs[ts], flow_gates[ts]]) for ts in range(self.lstm_seq_len)]

        # dense part
        nbhd_vecs = [Flatten(name = "nbhd_flatten_time{0}".format(ts+1))(nbhd_convs[ts]) for ts in range(self.lstm_seq_len)]
        nbhd_vecs = [self.short_nbhd_dense[ts](nbhd_vecs[ts]) for ts in range(self.lstm_seq_len)]
        nbhd_vecs = [Activation("relu", name = "nbhd_dense_activation_time{0}".format(ts+1))(nbhd_vecs[ts]) for ts in range(self.lstm_seq_len)]

        # poi part
        poi_convs = [self.short_poi_conv[ts](poi_data) for ts in range(self.lstm_seq_len)]
        poi_convs = [Activation("relu", name = "poi_conv_activation_time{0}".format(ts+1))(poi_convs[ts]) for ts in range(self.lstm_seq_len)]
        poi_vecs = [Flatten(name = "poi_flatten_time{0}".format(ts+1))(poi_convs[ts]) for ts in range(self.lstm_seq_len)]
        poi_vecs = [self.short_poi_dense[ts](poi_vecs[ts]) for ts in range(self.lstm_seq_len)]
        poi_vecs = [Activation("relu", name = "poi_dense_activation_{0}".format(ts+1))(poi_vecs[ts]) for ts in range(self.lstm_seq_len)]
        poi_vec = Concatenate(axis = -1)(poi_vecs)
        poi_vec = Reshape(target_shape = (self.lstm_seq_len, self.cnn_flat_size))(poi_vec)

        # feature concatenate
        # lstm_feature, nbhd_convs, weather, poi
        nbhd_vec = Concatenate(axis=-1)(nbhd_vecs)
        nbhd_vec = Reshape(target_shape = (self.lstm_seq_len, self.cnn_flat_size))(nbhd_vec)
        lstm_input = Concatenate(axis=-1)([lstm_inputs, nbhd_vec, weather, poi_vec])

        #lstm
        lstm = self.short_lstm(lstm_input)

        # attention part
        level = 0
        att_nbhd_convs = [[self.long_conv[level][att][ts][0](att_nbhd_inputs[att][ts], choice = self.nas_choice[4][level][att][ts][0]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[self.long_pooling[level][att][ts][0](att_nbhd_convs[att][ts], choice = self.nas_choice[5][level][att][ts][0]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[self.long_conv_activ[level][att][ts][0](att_nbhd_convs[att][ts], choice = self.nas_choice[6][level][att][ts][0]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.long_conv[level][att][ts][1](att_flow_inputs[att][ts], choice = self.nas_choice[4][level][att][ts][1]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.long_pooling[level][att][ts][1](att_flow_convs[att][ts], choice = self.nas_choice[5][level][att][ts][1]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.long_conv_activ[level][att][ts][1](att_flow_convs[att][ts], choice = self.nas_choice[6][level][att][ts][1]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_gates = [[self.long_gate_activ[level][att][ts](att_flow_convs[att][ts], choice = self.nas_choice[7][level][att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[keras.layers.Multiply()([att_nbhd_convs[att][ts], att_flow_gates[att][ts]]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]

        level = 1
        att_nbhd_convs = [[self.long_conv[level][att][ts][0](att_nbhd_convs[att][ts], choice = self.nas_choice[4][level][att][ts][0]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[self.long_pooling[level][att][ts][0](att_nbhd_convs[att][ts], choice = self.nas_choice[5][level][att][ts][0]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[self.long_conv_activ[level][att][ts][0](att_nbhd_convs[att][ts], choice = self.nas_choice[6][level][att][ts][0]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.long_conv[level][att][ts][1](att_flow_inputs[att][ts], choice = self.nas_choice[4][level][att][ts][1]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.long_pooling[level][att][ts][1](att_flow_convs[att][ts], choice = self.nas_choice[5][level][att][ts][1]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.long_conv_activ[level][att][ts][1](att_flow_convs[att][ts], choice = self.nas_choice[6][level][att][ts][1]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_gates = [[self.long_gate_activ[level][att][ts](att_flow_convs[att][ts], choice = self.nas_choice[7][level][att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[keras.layers.Multiply()([att_nbhd_convs[att][ts], att_flow_gates[att][ts]]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]

        level = 2
        att_nbhd_convs = [[self.long_conv[level][att][ts][0](att_nbhd_convs[att][ts], choice = self.nas_choice[4][level][att][ts][0]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[self.long_pooling[level][att][ts][0](att_nbhd_convs[att][ts], choice = self.nas_choice[5][level][att][ts][0]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[self.long_conv_activ[level][att][ts][0](att_nbhd_convs[att][ts], choice = self.nas_choice[6][level][att][ts][0]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.long_conv[level][att][ts][1](att_flow_inputs[att][ts], choice = self.nas_choice[4][level][att][ts][1]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.long_pooling[level][att][ts][1](att_flow_convs[att][ts], choice = self.nas_choice[5][level][att][ts][1]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.long_conv_activ[level][att][ts][1](att_flow_convs[att][ts], choice = self.nas_choice[6][level][att][ts][1]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_gates = [[self.long_gate_activ[level][att][ts](att_flow_convs[att][ts], choice = self.nas_choice[7][level][att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[keras.layers.Multiply()([att_nbhd_convs[att][ts], att_flow_gates[att][ts]]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]

        att_nbhd_vecs = [[Flatten(name = "att_nbhd_flatten_time_{0}_{1}".format(att+1,ts+1))(att_nbhd_convs[att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_vecs = [[self.long_nbhd_dense[att][ts](att_nbhd_vecs[att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_vecs = [[Activation("relu", name = "att_nbhd_dense_activation_time_{0}_{1}".format(att+1,ts+1))(att_nbhd_vecs[att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]

        att_poi_convs = [[self.long_poi_conv[att][ts](poi_data) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_poi_vecs = [[Flatten(name = "att_poi_flatten_{0}_{1}".format(att+1, ts+1))(att_poi_convs[att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_poi_vecs = [[self.long_poi_dense[att][ts](att_poi_vecs[att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_poi_vecs = [[Activation("relu", name = "att_poi_dense_activation_time_{0}_{1}".format(att+1, ts+1))(att_poi_vecs[att][ts]) for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_poi_vec = [Concatenate(axis = -1)(att_poi_vecs[att]) for att in range(self.att_lstm_num)]
        att_poi_vec = [Reshape(target_shape = (self.att_lstm_seq_len, self.cnn_flat_size))(att_poi_vec[att]) for att in range(self.att_lstm_num)]

        att_nbhd_vec = [Concatenate(axis=-1)(att_nbhd_vecs[att]) for att in range(self.att_lstm_num)]
        att_nbhd_vec = [Reshape(target_shape = (self.att_lstm_seq_len, self.cnn_flat_size))(att_nbhd_vec[att]) for att in range(self.att_lstm_num)]
        att_lstm_input = [Concatenate(axis=-1)([att_lstm_inputs[att], att_nbhd_vec[att], att_weather[att], att_poi_vec[att]]) for att in range(self.att_lstm_num)]

        att_lstms = [self.long_lstm[att](att_lstm_input[att]) for att in range(self.att_lstm_num)]

        #compare
        att_low_level=[self.long_attention[att]([att_lstms[att], lstm]) for att in range(self.att_lstm_num)]
        att_low_level=Concatenate(axis=-1)(att_low_level)
        att_low_level=Reshape(target_shape=(self.att_lstm_num, self.lstm_out_size))(att_low_level)


        att_high_level = self.high_level_lstm(att_low_level)

        lstm_all = Concatenate(axis=-1)([att_high_level, lstm])
        # lstm_all = Dropout(rate = .3)(lstm_all)
        lstm_all = self.lstm_all_dense(lstm_all)
        pred_volume = Activation('tanh')(lstm_all)

        return pred_volume
        

class Custom_Model(keras.Model):
    def set_choice(self, choice):
        self.choice = choice
    def train_step(self, data):
        x, y =data

        num_choice=3
        num_layers=6
        # conv size choice: [1,2,3]: 3 choices
        short_conv_choice = list(np.random.randint(num_choice, size=num_layers*self.lstm_seq_len))
        att_conv_choice = list(np.random.randint(num_choice, size=num_layers*self.att_lstm_num*self.att_lstm_seq_len))
        # pooling choice: [2,3,4] pool_size for max/avg pooling and no pooling: 3*2+1 choices
        short_pooling_choice = list(np.random.randint(num_choice*2+1, size=num_layers*self.lstm_seq_len))
        att_pooling_choice = list(np.random.randint(num_choice*2+1, size=num_layers*self.att_lstm_num*self.att_lstm_seq_len))
        # relu choice: relu, relu6, prelu: 3 choices
        short_relu_choice = list(np.random.randint(num_choice, size=num_layers*self.lstm_seq_len))
        att_relu_choice = list(np.random.randint(num_choice, size=num_layers*self.att_lstm_num*self.att_lstm_seq_len))
        # flow gate choice: sigmoid, relu6, tanh: 3 choices
        flow_gate_choice = list(np.random.randint(num_choice, size=3*self.lstm_seq_len))
        att_flow_gate_choice = list(np.random.randint(num_choice, size=3*self.att_lstm_num*self.att_lstm_seq_len))
        self.choice=[short_conv_choice, att_conv_choice, short_pooling_choice, att_pooling_choice, short_relu_choice, att_relu_choice, flow_gate_choice, att_flow_gate_choice]

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

class SAGAN_Functional_Model():
    def func_model(self, att_lstm_num, att_lstm_seq_len, lstm_seq_len, feature_vec_len, cnn_flat_size, lstm_out_size, \
              nbhd_size, poi_size, nbhd_type, flow_type, weather_type, poi_type, output_shape, optimizer = 'adagrad', loss = 'mse', metrics=[]):
        """
        short-term input
        including:
            1. nbhd: lstm_seq_len, (nbhd_size, nbhd_size, nbhd_type,)
            2. flow: lstm_seq_len, (nbhd_size, nbhd_size, flow_type,)
            3. lstm: (lstm_seq_len, feature_vec_len,)
            4. weather: (lstm_seq_len, weather_type,)
            5. poi: (poi_size, poi_size, poi_type,)
        """
        nbhd_inputs = [Input(shape = (nbhd_size, nbhd_size, nbhd_type,), name = "nbhd_volume_input_time_{0}".format(ts+1)) for ts in range(lstm_seq_len)]
        flow_inputs = [Input(shape = (nbhd_size, nbhd_size, flow_type,), name = "flow_volume_input_time_{0}".format(ts+1)) for ts in range(lstm_seq_len)]
        lstm_inputs = Input(shape = (lstm_seq_len, feature_vec_len,), name = "lstm_input")
        weather_inputs = Input(shape = (lstm_seq_len, weather_type,), name = "weather_input")
        poi_inputs = Input(shape = (poi_size, poi_size, poi_type,), name = "poi_input")

        """
        long-term input
        including:
            1. nbhd: att_lstm_num, att_lstm_seq_len, (nbhd_size, nbhd_size, nbhd_type,)
            2. flow: att_lstm_num, att_lstm_seq_len, (nbhd_size, nbhd_size, flow_type,)
            3. lstm: att_lstm_num, (att_lstm_seq_len, feature_vec_len,)
            4. weather: att_lstm_num, (att_lstm_seq_len, feature_vec_len,)
            5. poi: take the same short-term poi data
        """
        flatten_att_nbhd_inputs = [Input(shape = (nbhd_size, nbhd_size, nbhd_type,), name = "att_nbhd_volume_input_time_{0}_{1}".format(att+1, ts+1)) for ts in range(att_lstm_seq_len) for att in range(att_lstm_num)]
        flatten_att_flow_inputs = [Input(shape = (nbhd_size, nbhd_size, flow_type,), name = "att_flow_volume_input_time_{0}_{1}".format(att+1, ts+1)) for ts in range(att_lstm_seq_len) for att in range(att_lstm_num)]

        att_nbhd_inputs = []
        att_flow_inputs = []
        for att in range(att_lstm_num):
            att_nbhd_inputs.append(flatten_att_nbhd_inputs[att*att_lstm_seq_len:(att+1)*att_lstm_seq_len])
            att_flow_inputs.append(flatten_att_flow_inputs[att*att_lstm_seq_len:(att+1)*att_lstm_seq_len])
        att_lstm_inputs = [Input(shape = (att_lstm_seq_len, feature_vec_len,), name = "att_lstm_input_{0}".format(att+1)) for att in range(att_lstm_num)]
        att_weather_inputs = [Input(shape = (att_lstm_seq_len, weather_type,), name = "att_weather_input_{0}".format(att+1)) for att in range(att_lstm_num)]

        """
        Record the Convs Kernel Size
        """
        gate_level = 3
        # kernel size list: 2 (nbhd / flow) * gate_level * lstm_seq_len
        short_conv_choice = []
        for conv_type in range(2):
            short_conv_choice.append([])
            for gate_level in range(3):
                short_conv_choice[conv_type].append([])
                interval = (conv_type*gate_level*lstm_seq_len)
                short_conv_choice[conv_type][gate_level].append(self.choice[0][interval:(interval + lstm_seq_len)])

        """
        Multiple Input Model Building
        """
        # short-term
        # first level gate
        nbhd_convs = [Conv2D(filters = 64, kernel_size = (short_conv_choice[0][0][ts], short_conv_choice[0][0][ts]), \
            padding = 'same', name = "nbhd_convs_time0_{0}_size{1}".format(ts+1, short_conv_choice[0][0][ts]))(nbhd_inputs[ts]) for ts in range(lstm_seq_len)]
        nbhd_convs = [Activation("relu", name = "nbhd_convs_activation_time0_{0}".format(ts+1))(nbhd_convs[ts]) for ts in range(lstm_seq_len)]
        #flow cnn
        flow_convs = [Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "flow_convs_time0_{0}".format(ts+1))(flow_inputs[ts]) for ts in range(lstm_seq_len)]
        flow_convs = [Activation("relu", name = "flow_convs_activation_time0_{0}".format(ts+1))(flow_convs[ts]) for ts in range(lstm_seq_len)]
        #flow gate
        flow_gates = [Activation("sigmoid", name = "flow_gate0_{0}".format(ts+1))(flow_convs[ts]) for ts in range(lstm_seq_len)]
        nbhd_convs = [keras.layers.Multiply()([nbhd_convs[ts], flow_gates[ts]]) for ts in range(lstm_seq_len)]


        #2nd level gate
        nbhd_convs = [Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "nbhd_convs_time1_{0}".format(ts+1))(nbhd_convs[ts]) for ts in range(lstm_seq_len)]
        nbhd_convs = [Activation("relu", name = "nbhd_convs_activation_time1_{0}".format(ts+1))(nbhd_convs[ts]) for ts in range(lstm_seq_len)]
        flow_convs = [Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "flow_convs_time1_{0}".format(ts+1))(flow_inputs[ts]) for ts in range(lstm_seq_len)]
        flow_convs = [Activation("relu", name = "flow_convs_activation_time1_{0}".format(ts+1))(flow_convs[ts]) for ts in range(lstm_seq_len)]
        flow_gates = [Activation("sigmoid", name = "flow_gate1_{0}".format(ts+1))(flow_convs[ts]) for ts in range(lstm_seq_len)]
        nbhd_convs = [keras.layers.Multiply()([nbhd_convs[ts], flow_gates[ts]]) for ts in range(lstm_seq_len)]

        #3rd level gate
        nbhd_convs = [Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "nbhd_convs_time2_{0}".format(ts+1))(nbhd_convs[ts]) for ts in range(lstm_seq_len)]
        nbhd_convs = [Activation("relu", name = "nbhd_convs_activation_time2_{0}".format(ts+1))(nbhd_convs[ts]) for ts in range(lstm_seq_len)]
        flow_convs = [Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "flow_convs_time2_{0}".format(ts+1))(flow_inputs[ts]) for ts in range(lstm_seq_len)]
        flow_convs = [Activation("relu", name = "flow_convs_activation_time2_{0}".format(ts+1))(flow_convs[ts]) for ts in range(lstm_seq_len)]
        flow_gates = [Activation("sigmoid", name = "flow_gate2_{0}".format(ts+1))(flow_convs[ts]) for ts in range(lstm_seq_len)]
        nbhd_convs = [keras.layers.Multiply()([nbhd_convs[ts], flow_gates[ts]]) for ts in range(lstm_seq_len)]


        #dense part
        nbhd_vecs = [Flatten(name = "nbhd_flatten_time_{0}".format(ts+1))(nbhd_convs[ts]) for ts in range(lstm_seq_len)]
        nbhd_vecs = [Dense(units = cnn_flat_size, name = "nbhd_dense_time_{0}".format(ts+1))(nbhd_vecs[ts]) for ts in range(lstm_seq_len)]
        nbhd_vecs = [Activation("relu", name = "nbhd_dense_activation_time_{0}".format(ts+1))(nbhd_vecs[ts]) for ts in range(lstm_seq_len)]

        #feature concatenate
        nbhd_vec = Concatenate(axis=-1)(nbhd_vecs)
        nbhd_vec = Reshape(target_shape = (lstm_seq_len, cnn_flat_size))(nbhd_vec)
        lstm_input = Concatenate(axis=-1)([lstm_inputs, nbhd_vec])

        #lstm
        lstm = LSTM(units=lstm_out_size, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)(lstm_input)

        #attention part
        att_nbhd_convs = [[Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "att_nbhd_convs_time0_{0}_{1}".format(att+1,ts+1))(att_nbhd_inputs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_convs = [[Activation("relu", name = "att_nbhd_convs_activation_time0_{0}_{1}".format(att+1,ts+1))(att_nbhd_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_convs = [[Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "att_flow_convs_time0_{0}_{1}".format(att+1,ts+1))(att_flow_inputs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_convs = [[Activation("relu", name = "att_flow_convs_activation_time0_{0}_{1}".format(att+1,ts+1))(att_flow_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_gates = [[Activation("sigmoid", name = "att_flow_gate0_{0}_{1}".format(att+1, ts+1))(att_flow_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_convs = [[keras.layers.Multiply()([att_nbhd_convs[att][ts], att_flow_gates[att][ts]]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]

        att_nbhd_convs = [[Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "att_nbhd_convs_time1_{0}_{1}".format(att+1,ts+1))(att_nbhd_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_convs = [[Activation("relu", name = "att_nbhd_convs_activation_time1_{0}_{1}".format(att+1,ts+1))(att_nbhd_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_convs = [[Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "att_flow_convs_time1_{0}_{1}".format(att+1,ts+1))(att_flow_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_convs = [[Activation("relu", name = "att_flow_convs_activation_time1_{0}_{1}".format(att+1,ts+1))(att_flow_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_gates = [[Activation("sigmoid", name = "att_flow_gate1_{0}_{1}".format(att+1, ts+1))(att_flow_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_convs = [[keras.layers.Multiply()([att_nbhd_convs[att][ts], att_flow_gates[att][ts]]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]

        att_nbhd_convs = [[Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "att_nbhd_convs_time2_{0}_{1}".format(att+1,ts+1))(att_nbhd_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_convs = [[Activation("relu", name = "att_nbhd_convs_activation_time2_{0}_{1}".format(att+1,ts+1))(att_nbhd_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_convs = [[Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "att_flow_convs_time2_{0}_{1}".format(att+1,ts+1))(att_flow_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_convs = [[Activation("relu", name = "att_flow_convs_activation_time2_{0}_{1}".format(att+1,ts+1))(att_flow_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_gates = [[Activation("sigmoid", name = "att_flow_gate2_{0}_{1}".format(att+1, ts+1))(att_flow_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_convs = [[keras.layers.Multiply()([att_nbhd_convs[att][ts], att_flow_gates[att][ts]]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]

        att_nbhd_vecs = [[Flatten(name = "att_nbhd_flatten_time_{0}_{1}".format(att+1,ts+1))(att_nbhd_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_vecs = [[Dense(units = cnn_flat_size, name = "att_nbhd_dense_time_{0}_{1}".format(att+1,ts+1))(att_nbhd_vecs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_vecs = [[Activation("relu", name = "att_nbhd_dense_activation_time_{0}_{1}".format(att+1,ts+1))(att_nbhd_vecs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]


        att_nbhd_vec = [Concatenate(axis=-1)(att_nbhd_vecs[att]) for att in range(att_lstm_num)]
        att_nbhd_vec = [Reshape(target_shape = (att_lstm_seq_len, cnn_flat_size))(att_nbhd_vec[att]) for att in range(att_lstm_num)]
        att_lstm_input = [Concatenate(axis=-1)([att_lstm_inputs[att], att_nbhd_vec[att]]) for att in range(att_lstm_num)]

        att_lstms = [LSTM(units=lstm_out_size, return_sequences=True, dropout=0.1, recurrent_dropout=0.1, name="att_lstm_{0}".format(att + 1))(att_lstm_input[att]) for att in range(att_lstm_num)]

        #compare
        att_low_level=[attention.Attention(method='cba')([att_lstms[att], lstm]) for att in range(att_lstm_num)]
        att_low_level=Concatenate(axis=-1)(att_low_level)
        att_low_level=Reshape(target_shape=(att_lstm_num, lstm_out_size))(att_low_level)


        att_high_level = LSTM(units=lstm_out_size, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)(att_low_level)

        lstm_all = Concatenate(axis=-1)([att_high_level, lstm])
        # lstm_all = Dropout(rate = .3)(lstm_all)
        lstm_all = Dense(units = output_shape)(lstm_all)
        pred_volume = Activation('tanh')(lstm_all)

        inputs = flatten_att_nbhd_inputs + flatten_att_flow_inputs + att_lstm_inputs + nbhd_inputs + flow_inputs + [lstm_inputs,]
        # print("Model input length: {0}".format(len(inputs)))
        # ipdb.set_trace()
        model = Model(inputs = inputs, outputs = pred_volume)
        model.compile(optimizer = optimizer, loss = loss, metrics=metrics)
        return model