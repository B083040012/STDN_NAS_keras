import keras, logging, yaml, time
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from choice_block_model import SAGAN_Functional_Model
from sagan_file_loader import SAGAN_fileloader
from criterion import eval_together, eval_lstm
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

K.set_session(tf.compat.v1.Session(config=config))
from keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

# load yaml
with open("parameters.yml", "r") as stream:
    config=yaml.load(stream, Loader=yaml.FullLoader)

batch_size=config["training"]["batch_size"]
max_epochs = config["training"]["max_epochs"]
att_lstm_num = config["dataset"]["att_lstm_num"]
long_term_lstm_seq_len = config["dataset"]["long_term_lstm_seq_len"]
short_term_lstm_seq_len = config["dataset"]["short_term_lstm_seq_len"]
cnn_nbhd_size = config["dataset"]["cnn_nbhd_size"]
nbhd_size = config["dataset"]["nbhd_size"]
cnn_flat_size = config["dataset"]["cnn_flat_size"]
hist_feature_daynum = config["dataset"]["hist_feature_daynum"]
last_feature_num = config["dataset"]["last_feature_num"]

# custom for early stop
class CustomStopper(keras.callbacks.EarlyStopping):
    # add argument for starting epoch
    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', start_epoch=40):
        super().__init__(monitor=monitor, min_delta=min_delta, patience=patience, verbose=verbose, mode=mode)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)

def retrain_architecture(batch_size=64, max_epochs=100, validation_split=0.2, early_stop=EarlyStopping()):

    # load log file
    file_handlers=[
            logging.FileHandler(config["file"]["path"]+"STDN_Network_retraining_testing.log"),
            logging.StreamHandler()
    ]
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt='%m/%d %I:%M:%S %p',
        handlers=file_handlers
    )

    # architecture training
    logging.info("[Retraining Architecture Phase]")

    # loading data
    logging.info("loading training data...")
    dataloader = SAGAN_fileloader()
    att_cnn, att_flow, att_lstm, att_weather, short_cnn, short_flow, short_lstm, short_weather, short_poi, y = dataloader.sample_sagan("train",\
                                                                                              att_lstm_num, long_term_lstm_seq_len,\
                                                                                              short_term_lstm_seq_len, hist_feature_daynum,\
                                                                                              last_feature_num)

    print("size of training data: ")
    print("att_cnn: ", len(att_cnn), att_cnn[0].shape)
    print("att_flow: ", len(att_flow), att_flow[0].shape)
    print("att_lstm: ", len(att_lstm), att_lstm[0].shape)
    print("att_weather: ", len(att_weather), att_weather[0].shape)
    # print("att poi: ", len(att_poi), att_poi[0].shape)
    print("short_cnn: ", len(short_cnn), short_cnn[0].shape)
    print("short_flow: ", len(short_flow), short_flow[0].shape)
    print("short_lstm: ", short_lstm.shape)
    print("short_weather: ", short_weather.shape)
    print("short poi: ", short_poi.shape)
    print("y: ", y.shape)

    # train_data = [att_cnn, att_flow, att_lstm, att_weather, short_cnn, short_flow, [short_lstm, ], short_weather, short_poi]
    train_label = y
    print("Start training supernet with input shape {1} / {0}".format(short_lstm.shape, short_cnn[0].shape))
    logging.info("train data loading complete")

    logging.info("loading architecture...")
    filepath=config["file"]["path"] + 'retrained_best_weights'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, \
                            save_best_only=True, save_weights_only=True, \
                            mode='auto')
    # tf.config.run_functions_eagerly(True)
    searched_choice=np.load(open(config["file"]["path"]+"searched_choice_list.npy", "rb"), allow_pickle = True)

    modeler = SAGAN_Functional_Model()
    model = modeler.func_model(nas_choice = searched_choice, att_lstm_num = att_lstm_num, att_lstm_seq_len = long_term_lstm_seq_len, lstm_seq_len = short_term_lstm_seq_len, \
        feature_vec_len = short_lstm.shape[-1], cnn_flat_size = cnn_flat_size, lstm_out_size = 128, nbhd_size = short_cnn[0].shape[1], poi_size = short_poi.shape[1], \
        nbhd_type = 2, flow_type = short_flow[0].shape[-1], weather_type = short_weather.shape[-1], poi_type = short_poi.shape[-1], output_shape = 2, optimizer = 'adagrad', loss = 'mse', metrics=[])

    logging.info("architecture loading complete")

    logging.info("retraining start")
    start=time.time()
    model.fit( \
        x = att_cnn + att_flow + att_lstm + att_weather + short_cnn + short_flow + [short_lstm,] + [short_weather,] + [short_poi,], \
        y = train_label, \
        batch_size=batch_size, validation_split=validation_split, epochs=max_epochs, callbacks=[early_stop, checkpoint])
    end=time.time()
    logging.info("retraining complete")

    model.save_weights(config["file"]["path"]+"retrained_final_weights")
    logging.info("[Retrained Architecture Weight Saved]")
    model.save(config["file"]["path"]+"retrained_final_model.hdf5")
    logging.info("[Retrained Architecture Model Saved]")
    logging.info("[Retraining Architecture Phase End] : total time: {0} sec".format(end-start))

if __name__=='__main__':
    stop = CustomStopper(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min', start_epoch=40)
    retrain_architecture(batch_size=batch_size, max_epochs=max_epochs, early_stop=stop)