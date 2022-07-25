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
    dataloader = SAGAN_fileloader()
    att_cnn, att_flow, att_lstm, att_weather, short_cnn, short_flow, short_lstm, short_weather, short_poi, y = dataloader.sample_sagan("test",\
                                                                                              att_lstm_num, long_term_lstm_seq_len,\
                                                                                              short_term_lstm_seq_len, hist_feature_daynum,\
                                                                                              last_feature_num)

    # test_data = [att_cnn, att_flow, att_lstm, att_weather, short_cnn, short_flow, [short_lstm, ], short_weather, short_poi]
    logging.info("test data loading complete")

    logging.info("loading architecture...")
    # tf.config.run_functions_eagerly(True)
    searched_choice=np.load(open(config["file"]["path"]+"searched_choice_list.npy", "rb"), allow_pickle = True)
    # model = keras.models.load_model(config["file"]["path"]+"retrained_final_model.h5")
    # model=STDN_Network(searched_choice, att_lstm_num=att_lstm_num, att_lstm_seq_len=long_term_lstm_seq_len, \
    #                         lstm_seq_len=len(short_cnn), feature_vec_len=short_lstm.shape[-1], \
    #                         cnn_flat_size=cnn_flat_size, nbhd_size=short_cnn[0].shape[1], nbhd_type=short_cnn[0].shape[-1])

    modeler = SAGAN_Functional_Model()
    model = modeler.func_model(nas_choice = searched_choice, att_lstm_num = att_lstm_num, att_lstm_seq_len = long_term_lstm_seq_len, lstm_seq_len = short_term_lstm_seq_len, \
        feature_vec_len = short_lstm.shape[-1], cnn_flat_size = cnn_flat_size, lstm_out_size = 128, nbhd_size = short_cnn[0].shape[1], poi_size = short_poi.shape[1], \
        nbhd_type = 2, flow_type = short_flow[0].shape[-1], weather_type = short_weather.shape[-1], poi_type = short_poi.shape[-1], output_shape = 2, optimizer = 'adagrad', loss = 'mse', metrics=[])

    checkpoint_file= config["file"]["path"] + 'retrained_final_weights'
    model.load_weights(checkpoint_file)
    logging.info("Finishing import the pretrained supernet")
    logging.info("architecture loading complete")

    logging.info("evaluating start (without denormalize)...")
    test_pred = model.predict( x = att_cnn + att_flow + att_lstm + att_weather + short_cnn + short_flow + [short_lstm,] + [short_weather,] + [short_poi,] )
    test_label = y

    threshold = float(config["dataset"]["threshold"]) / config["dataset"]["volume_test_max"]
    print("Evaluating threshold: {0}.".format(threshold))

    total_loss_rmse, total_loss_mape = eval_together(test_label, test_pred, threshold)
    (prmse, pmape), (drmse, dmape) = eval_lstm(test_label, test_pred, threshold)
    logging.info("final_architecture_testing complete")
    logging.info("[Final Testing Result Without Denormalize] pickup rmse = {0}, pickup mape = {1}%\ndropoff rmse = {2}, dropoff mape = {3}%".format(prmse, pmape * 100, drmse, dmape * 100))
    logging.info("[Final Testing Result Without Denormalize] total_rmse = {0}, total_mape = {1}".format(total_loss_rmse, total_loss_mape * 100))

    logging.info("evaluating start (with denormalize)...")
    test_pred = test_pred * config["dataset"]["volume_test_max"]
    test_label = y * config["dataset"]["volume_test_max"]

    threshold = float(config["dataset"]["threshold"])
    print("Evaluating threshold: {0}.".format(threshold))

    total_loss_rmse, total_loss_mape = eval_together(test_label, test_pred, threshold)
    (prmse, pmape), (drmse, dmape) = eval_lstm(test_label, test_pred, threshold)
    logging.info("final_architecture_testing complete")
    logging.info("[Final Testing Result With Denormalize] pickup rmse = {0}, pickup mape = {1}%\ndropoff rmse = {2}, dropoff mape = {3}%".format(prmse, pmape * 100, drmse, dmape * 100))
    logging.info("[Final Testing Result With Denormalize] total_rmse = {0}, total_mape = {1}".format(total_loss_rmse, total_loss_mape * 100))
    logging.info("[Final Architecture Testing Phase End]")

if __name__=='__main__':
    eval_architecture()