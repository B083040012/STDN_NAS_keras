import keras, logging, yaml, time
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from models import STDN_Network
from file_loader import STDN_fileloader
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
long_term_lstm_seq_num = config["dataset"]["long_term_lstm_seq_num"]
short_term_lstm_seq_num = config["dataset"]["short_term_lstm_seq_num"]
cnn_nbhd_size = config["dataset"]["cnn_nbhd_size"]
nbhd_size = config["dataset"]["nbhd_size"]
cnn_flat_size = config["dataset"]["cnn_flat_size"]

def eval_architecture():
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
    
    logging.info("[Architecture Testing Phase]")
    logging.info("loading testing data...")
    dataloader = STDN_fileloader(config_path = "data_bike.json")
    att_cnnx, att_flow, att_x, cnnx, flow, x, y, weather, att_weather = dataloader.sample_stdn(datatype="test", nbhd_size=nbhd_size,
                                                                          cnn_nbhd_size=cnn_nbhd_size)

    test_data = [att_cnnx, att_flow, att_x, cnnx, flow, [x, ], weather, att_weather]
    test_label = y * config["dataset"]["volume_test_max"]
    logging.info("test data loading complete")

    logging.info("loading architecture...")
    tf.config.run_functions_eagerly(True)
    searched_choice=np.load(open(config["file"]["path"]+"searched_choice_list.npy", "rb"), allow_pickle = True)
    model=STDN_Network(searched_choice, att_lstm_num=att_lstm_num, att_lstm_seq_len=long_term_lstm_seq_num, \
                            lstm_seq_len=len(cnnx), feature_vec_len=x.shape[-1], \
                            cnn_flat_size=cnn_flat_size, nbhd_size=cnnx[0].shape[1], nbhd_type=cnnx[0].shape[-1])

    checkpoint_file= config["file"]["path"] + 'retrained_best_weights'
    model.load_weights(checkpoint_file).expect_partial()
    logging.info("Finishing import the pretrained supernet")
    logging.info("architecture loading complete")

    logging.info("evaluating start (without denormalize)...")
    test_pred = model.predict( x=test_data )
    test_label = y

    threshold = float(config["dataset"]["threshold"]) / config["dataset"]["volume_test_max"]
    print("Evaluating threshold: {0}.".format(threshold))

    total_loss_rmse, total_loss_mape = eval_together(test_label, test_pred, threshold)
    (prmse, pmape), (drmse, dmape) = eval_lstm(test_label, test_pred, threshold)
    logging.info("final_architecture_testing complete (without denormalize)")
    logging.info("[Final Testing Result Without Denormalize] pickup rmse = {0}, pickup mape = {1}%\ndropoff rmse = {2}, dropoff mape = {3}%".format(prmse, pmape * 100, drmse, dmape * 100))
    logging.info("[Final Testing Result Without Denormalize] total_rmse = {0}, total_mape = {1}".format(total_loss_rmse, total_loss_mape * 100))

    logging.info("evaluating start (with denormalize)...")
    test_pred = test_pred * config["dataset"]["volume_test_max"]
    test_label = y * config["dataset"]["volume_test_max"]

    threshold = float(config["dataset"]["threshold"])
    print("Evaluating threshold: {0}.".format(threshold))

    total_loss_rmse, total_loss_mape = eval_together(test_label, test_pred, threshold)
    (prmse, pmape), (drmse, dmape) = eval_lstm(test_label, test_pred, threshold)
    logging.info("final_architecture_testing complete (with denormalize)")
    logging.info("[Final Testing Result With Denormalize] pickup rmse = {0}, pickup mape = {1}%\ndropoff rmse = {2}, dropoff mape = {3}%".format(prmse, pmape * 100, drmse, dmape * 100))
    logging.info("[Final Testing Result With Denormalize] total_rmse = {0}, total_mape = {1}".format(total_loss_rmse, total_loss_mape * 100))
    logging.info("[Final Architecture Testing Phase End]")

if __name__=='__main__':
    eval_architecture()