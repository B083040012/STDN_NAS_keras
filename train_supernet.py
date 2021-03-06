import keras, logging, yaml, time
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from models import STDN_NAS
from file_loader import STDN_fileloader
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
long_term_lstm_seq_num = config["dataset"]["long_term_lstm_seq_num"]
short_term_lstm_seq_num = config["dataset"]["short_term_lstm_seq_num"]
cnn_nbhd_size = config["dataset"]["cnn_nbhd_size"]
nbhd_size = config["dataset"]["nbhd_size"]
cnn_flat_size = config["dataset"]["cnn_flat_size"]

# custom for early stop
class CustomStopper(keras.callbacks.EarlyStopping):
    # add argument for starting epoch
    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', start_epoch=40):
        super().__init__(monitor=monitor, min_delta=min_delta, patience=patience, verbose=verbose, mode=mode)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)


def supernet_training(batch_size=64, max_epochs=100, validation_split=0.2, early_stop=EarlyStopping()):

    # load log file
    file_handlers=[
            logging.FileHandler(config["file"]["path"]+"STDN_NAS_Supernet_training.log"),
            logging.StreamHandler()
    ]
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt='%m/%d %I:%M:%S %p',
        handlers=file_handlers
    )

    # supernet training
    logging.info("[Supernet Training Phase]")

    # loading data
    logging.info("loading training data...")
    dataloader = STDN_fileloader(config_path = "data_bike.json")
    att_cnnx, att_flow, att_x, cnnx, flow, x, y, weather, att_weather = dataloader.sample_stdn(datatype="train",
                                                                        att_lstm_num=att_lstm_num, \
                                                                        long_term_lstm_seq_len=long_term_lstm_seq_num,
                                                                        short_term_lstm_seq_len=short_term_lstm_seq_num, \
                                                                        nbhd_size=nbhd_size,
                                                                        cnn_nbhd_size=cnn_nbhd_size)

    train_data = [att_cnnx, att_flow, att_x, cnnx, flow, [x, ], weather, att_weather]
    train_label = y
    print("Start training supernet with input shape {1} / {0}".format(x.shape, cnnx[0].shape))
    logging.info("train data loading complete")

    filepath=config["file"]["path"] + 'best_supernet_cpt'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, \
                            save_best_only=True, save_weights_only=True, \
                            mode='auto')

    logging.info("loading supernet...")
    tf.config.run_functions_eagerly(True)
    model=STDN_NAS(att_lstm_num=att_lstm_num, att_lstm_seq_len=long_term_lstm_seq_num, \
                            lstm_seq_len=len(cnnx), feature_vec_len=x.shape[-1], \
                            cnn_flat_size=cnn_flat_size, nbhd_size=cnnx[0].shape[1], nbhd_type=cnnx[0].shape[-1])
    logging.info("supernet loading complete")

    model.compile(optimizer = 'adagrad', loss = 'mse', metrics=[])
    logging.info("start training supernet...")
    start = time.time()
    model.fit( \
        x = train_data, \
        y = train_label, \
        batch_size=batch_size, validation_split=validation_split, epochs=max_epochs, callbacks=[early_stop, checkpoint])
    end = time.time()
    model.save_weights(config["file"]["path"]+"final_architecture")
    logging.info("supernet training complete")
    logging.info("[Supernet Architecture Weight Saved]")
    logging.info("[Supernet Training Phase End] : total time: {0} sec".format(end-start))


if __name__ == "__main__":
    stop = CustomStopper(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min', start_epoch=40)
    supernet_training(batch_size=batch_size, max_epochs=max_epochs, early_stop=stop)
