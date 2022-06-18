from models import STDN_NAS
from ASAGA import ASAGA_Searcher
from file_loader import STDN_fileloader
import yaml, logging, time, os, keras
import numpy as np
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

def search():
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

    # load log file
    file_handlers=[
            logging.FileHandler(config["file"]["path"]+"STDN_NAS_searching.log"),
            logging.StreamHandler()
    ]

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt='%m/%d %I:%M:%S %p',
        handlers=file_handlers
    )

    """
    Load the pretrained supernet (in checkpoint.pth)
    """
    logging.info("[Architecture Searching Phase...]")
    checkpoint_file= config["file"]["path"] + 'best_supernet_cpt'

    # loading val dataset
    dataloader= STDN_fileloader(config_path = "data_bike.json")
    att_cnnx, att_flow, att_x, cnnx, flow, x, y =  dataloader.sample_stdn(datatype="validation",
                                                                        att_lstm_num=att_lstm_num, \
                                                                        long_term_lstm_seq_len=long_term_lstm_seq_num,
                                                                        short_term_lstm_seq_len=short_term_lstm_seq_num, \
                                                                        nbhd_size=nbhd_size,
                                                                        cnn_nbhd_size=cnn_nbhd_size)
    val_loader=[att_cnnx, att_flow, att_x, cnnx, flow, [x,]]
    val_label = y*config["dataset"]["volume_train_max"]

    # loading model
    # model=keras.models.load_model(checkpoint_file)
    tf.config.run_functions_eagerly(True)
    model=STDN_NAS(att_lstm_num=att_lstm_num, att_lstm_seq_len=long_term_lstm_seq_num, \
                            lstm_seq_len=len(cnnx), feature_vec_len=x.shape[-1], \
                            cnn_flat_size=cnn_flat_size, nbhd_size=cnnx[0].shape[1], nbhd_type=cnnx[0].shape[-1])
    # model.compile(optimizer = 'adagrad', loss = 'mse', metrics=[])
    # num_choice=3
    # num_layers=6
    # nas_choice=list(np.random.randint(num_choice, size=num_layers*short_term_lstm_seq_num))
    
    # model.set_choice(nas_choice)
    # model.call(val_loader)

    model.load_weights(checkpoint_file).expect_partial()
    logging.info("Finishing import the pretrained supernet")

    """
    Searching for the Best Architecture
    by ASAGA
    """

    logger=logging.getLogger('Searcher')

    start=time.time()
    searcher=ASAGA_Searcher(config, logger, model, val_loader, val_label)
    searched_architecture, loss=searcher.search()
    end=time.time()
    logging.info("Search Complete")
    # logging.info("Searched Architecture: ", searched_architecture)
    print("Searched Architecture: ", searched_architecture)
    logging.info("[Searched Architecture Saved] Total Searched Time: %.5f sec, Architecture loss:%.5f" %((end-start), loss))
    searched_file=os.path.join(config["file"]["path"]+"searched_choice_list")
    np.save(searched_file, searched_architecture)

if __name__=='__main__':
    search()