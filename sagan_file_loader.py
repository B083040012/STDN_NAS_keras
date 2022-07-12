import numpy as np
import yaml, random

class SAGAN_fileloader:
    def __init__(self, config_path = "parameter.yml"):
        with open(config_path, "r") as stream:
            self.config = yaml.load(stream, Loader = yaml.FullLoader)
        self.timeslot_daynum = int(86400 / self.config["dataset"]["timeslot_sec"])
        self.threshold = int(self.config["dataset"]["threshold"])

    def load_train(self):
        """
        Load training data
        """
        self.volume_train = np.load(self.config["file"]["volume_train"])
        self.flow_train = np.load(self.config["file"]["flow_train"])
        self.weather_train = np.load(self.config["file"]["weather_train"])
        self.poi_data = np.load(self.config["file"]["poi_data"])
        self.start_date = self.config["dataset"]["start_date_train"]
        self.end_date = self.config["dataset"]["end_date_train"]
        self.start_hour = self.config["dataset"]["start_hour_train"]
        self.end_hour = self.config["dataset"]["end_hour_train"]

    def load_test(self):
        """
        Load validation & testing data
        """
        self.volume_test = np.load(self.config["file"]["volume_test"])
        self.flow_test = np.load(self.config["file"]["flow_test"])
        self.weather_test = np.load(self.config["file"]["weather_test"])
        self.poi_data = np.load(self.config["file"]["poi_data"])
        self.start_date = self.config["dataset"]["start_date_test"]
        self.end_date = self.config["dataset"]["end_date_test"]
        self.start_hour = self.config["dataset"]["start_hour_test"]
        self.end_hour = self.config["dataset"]["end_hour_test"]

    def sample_sagan(self, datatype, att_lstm_num, long_term_lstm_seq_len, short_term_lstm_seq_len,\
        hist_feature_daynum, last_feature_num):

        if long_term_lstm_seq_len % 2 != 1:
            print("att_lstm_seq_len must be odd !")
            raise Exception

        # load data depend on datatype
        if datatype == "train":
            self.load_train()
            volume_data = self.volume_train
            flow_data = self.flow_train
            weather_data = self.weather_train
            poi_data = self.poi_data
        elif datatype == "validation" or "test":
            self.load_test()
            volume_data = self.volume_test
            flow_data = self.flow_test
            weather_data = self.weather_test
            poi_data = self.poi_data
        else:
            print("Please select 'train', 'validation', or 'test'")
            raise Exception

        # initilalize short term features & label
        cnn_features = []
        flow_features = []
        weather_features = []
        poi_features = []
        for i in range(short_term_lstm_seq_len):
            cnn_features.append([])
            flow_features.append([])
        short_term_lstm_features = []
        labels = []

        # initialize long term features
        cnn_att_features = []
        lstm_att_features = []
        flow_att_features = []
        weather_att_features = []
        poi_att_features = []
        for i in range(att_lstm_num):
            cnn_att_features.append([])
            lstm_att_features.append([])
            flow_att_features.append([])
            weather_att_features.append([])
            poi_att_features.append([])
            for j in range(long_term_lstm_seq_len):
                cnn_att_features[i].append([])
                flow_att_features[i].append([])

        # initialize dataset time interval
        time_start = (hist_feature_daynum + att_lstm_num) * self.timeslot_daynum + long_term_lstm_seq_len
        time_end = volume_data.shape[0]

        time_range_list = sorted([hour + date*48 for hour in range(self.start_hour, self.end_hour) \
            for date in range(self.start_date, self.end_date) if hour + date*48 >= time_start])
        if datatype == 'validation':
            time_range_list = sorted(random.sample(time_range_list, int(len(time_range_list) * 0.2)))

        # list all features
        print("time interval length: {0}".format(len(time_range_list)))
        for index, t in enumerate(time_range_list):
            if index % 100 == 0:
                print("Now sampling at {0}th timeslots.".format(index))
            for station_idx in range(0, volume_data.shape[1]):
                """
                short-term features
                including:
                    1. cnn_features
                    2. flow_features
                    3. short_term_lstm_features
                    4. weather_features
                    5. poi_features
                """
                short_term_lstm_samples = []
                short_term_weather_samples = []
                for seqn in range(short_term_lstm_seq_len):
                    # real_t from (t - short_term_lstm_seq_len) to (t-1)
                    real_t = t - (short_term_lstm_seq_len - seqn)

                    """
                    short-term cnn_features
                        size: 10*10*2
                    """
                    cnn_feature = np.zeros((volume_data.shape[3], volume_data.shape[4], volume_data.shape[2]))
                    cnn_feature[:, :, 0] = volume_data[real_t, station_idx, 0, :, :]
                    cnn_feature[:, :, 1] = volume_data[real_t, station_idx, 1, :, :]
                    cnn_features[seqn].append(cnn_feature)

                    """
                    short-term flow features
                        size: 10*10*4
                        including:
                            1. curr outflow
                            2. curr inflow
                            3. last outflow
                            4. last inflow
                        # some doubt about the feature of 'last_out_to_curr' and 'curr_in_from_last'
                    """
                    flow_feature_curr_out = flow_data[real_t, station_idx, 0, :, :]
                    flow_feature_curr_in = flow_data[real_t, station_idx, 1, :, :]
                    flow_feature_last_out_to_curr = flow_data[real_t - 1, station_idx, 0, :, :]
                    flow_feature_curr_in_from_last = flow_data[real_t - 1, station_idx, 1, :, :]

                    flow_feature = np.zero(flow_feature_curr_in.shape+(4,))

                    flow_feature[:, :, 0] = flow_feature_curr_out
                    flow_feature[:, :, 1] = flow_feature_curr_in
                    flow_feature[:, :, 2] = flow_feature_last_out_to_curr
                    flow_feature[:, :, 3] = flow_feature_curr_in_from_last

                    flow_features[seqn].append(flow_feature)

                    """
                    short-term lstm features
                        size: ???
                        including:
                            1. volume feature
                            2. last feature
                            3. hist feature
                    
                    short-term weather features
                        size: 17
                        including:
                            1. temparature
                            2. dew point
                            3. humidity
                            4. wind speed
                            5. wind gust
                            6. pressure
                            7. precip
                            8~17. one hot encoding for ten weather type

                    short-term poi features
                        # not done yet
                    """
                    # volume feature
                    nbhd_feature = np.zero((volume_data.shape[3], volume_data.shape[4], volume_data.shape[2]))
                    nbhd_feature[:, :, 0] = volume_data[real_t, station_idx, 0, :, :]
                    nbhd_feature[:, :, 1] = volume_data[real_t, station_idx, 1, :, :]
                    nbhd_feature = nbhd_feature.flatten()
                    
                    # last feature
                    last_feature = np.zero((volume_data.shape[3], volume_data.shape[4], volume_data.shape[2]))
                    last_feature[:, :, 0] = volume_data[real_t - last_feature_num, station_idx, 0, :, :]
                    last_feature[:, :, 1] = volume_data[real_t - last_feature_num, station_idx, 1, :, :]
                    last_feature = last_feature.flatten()

                    # hist feature
                    hist_feature = np.zero((volume_data.shape[3], volume_data.shape[4], volume_data.shape[2]))
                    hist_feature[:, :, 0] = volume_data[real_t - hist_feature_daynum*self.timeslot_daynum: real_t: self.timeslot_daynum,\
                        station_idx, 0, :, :]
                    hist_feature[:, :, 1] = volume_data[real_t - hist_feature_daynum*self.timeslot_daynum: real_t: self.timeslot_daynum,\
                        station_idx, 1, :, :]
                    hist_feature = hist_feature.flatten()

                    feature_vec = np.concatenate((hist_feature, last_feature))
                    feature_vec = np.concatenate((feature_vec, nbhd_feature))

                    short_term_lstm_samples.append(feature_vec)
                    short_term_weather_samples.append(weather_data[real_t])

                short_term_lstm_features.append(np.array(short_term_lstm_features))
                weather_features.append(np.array(short_term_weather_samples))

                """
                long-term features
                including:
                    1. cnn_att_features
                    2. flow_att_features
                    3. lstm_att_features
                    4. weather_att_features
                    5. poi_att_features
                """
                for att_lstm_cnt in range(att_lstm_num):

                    long_term_lstm_samples = []
                    long_term_weather_samples = []

                    """
                    range of att_t:
                    for target timeslot t,
                        1. att_t first reach (att_lstm_num - att_lstm_cnt) days before (same time)
                            --> t - (att_lstm_num - att_lstm_cnt) * self.timeslot_daynum
                        2. for each day, sample the time from 
                            (long_term_lstm_seq_len / 2) before target time ~ (long_term_lstm_seq_len / 2) after target time
                    for example, if (att_lstm_num, long_term_lstm_seq_len) = (3, 3), target time = (day 4, time 9), then att_t sample from
                            day 1: time 8 ~ 10
                            day 2: time 8 ~ 10
                            day 3: time 8 ~ 10
                    """
                    att_t = int(t - (att_lstm_num - att_lstm_cnt) * self.timeslot_daynum + (long_term_lstm_seq_len - 1) / 2 + 1)

                    for seqn in range(long_term_lstm_seq_len):
                        real_t = att_t - (long_term_lstm_seq_len - seqn)

                        """
                        long term cnn features
                            size: 10*10*2
                        """
                        cnn_feature = np.zeros((volume_data.shape[3], volume_data.shape[4], volume_data.shape[2]))
                        cnn_feature[:, :, 0] = volume_data[real_t, station_idx, 0, :, :]
                        cnn_feature[:, :, 1] = volume_data[real_t, station_idx, 1, :, :]
                        cnn_att_features[att_lstm_cnt][seqn].append(cnn_feature)

                        """
                        long-term flow features
                            size: 10*10*4
                            including:
                                1. curr outflow
                                2. curr inflow
                                3. last outflow
                                4. last inflow
                            # some doubt about the feature of 'last_out_to_curr' and 'curr_in_from_last'
                        """
                        flow_feature_curr_out = flow_data[real_t, station_idx, 0, :, :]
                        flow_feature_curr_in = flow_data[real_t, station_idx, 1, :, :]
                        flow_feature_last_out_to_curr = flow_data[real_t, station_idx, 0, :, :]
                        flow_feature_curr_in_from_last = flow_data[real_t, station_idx, 1, :, :]

                        flow_feature = np.zero(flow_feature_curr_in.shape+(4,))

                        flow_feature[:, :, 0] = flow_feature_curr_out
                        flow_feature[:, :, 1] = flow_feature_curr_in
                        flow_feature[:, :, 2] = flow_feature_last_out_to_curr
                        flow_feature[:, :, 3] = flow_feature_curr_in_from_last

                        flow_att_features[att_lstm_cnt][seqn] = flow_feature

                        """
                        long-term lstm features
                            size: ???
                            including:
                                1. volume feature
                                2. last feature
                                3. hist feature
                        
                        long-term weather features
                            size: 17
                            including:
                                1. temparature
                                2. dew point
                                3. humidity
                                4. wind speed
                                5. wind gust
                                6. pressure
                                7. precip
                                8~17. one hot encoding for ten weather type

                        long-term poi features
                            # not done yet
                        """

                        # volume feature
                        nbhd_feature = np.zero((volume_data.shape[3], volume_data.shape[4], volume_data.shape[2]))
                        nbhd_feature[:, :, 0] = volume_data[real_t, station_idx, 0, :, :]
                        nbhd_feature[:, :, 1] = volume_data[real_t, station_idx, 1, :, :]
                        nbhd_feature = nbhd_feature.flatten()

                        # last feature
                        last_feature = np.zero((volume_data.shape[3], volume_data.shape[4], volume_data.shape[2]))
                        last_feature[:, :, 0] = volume_data[real_t - last_feature_num, station_idx, 0, :, :]
                        last_feature[:, :, 1] = volume_data[real_t - last_feature_num, station_idx, 1, :, :]
                        last_feature = last_feature.flatten()

                        # hist feature
                        hist_feature = np.zero((volume_data.shape[3], volume_data.shape[4], volume_data.shape[2]))
                        hist_feature[:, :, 0] = volume_data[real_t - hist_feature_daynum*self.timeslot_daynum: real_t: self.timeslot_daynum,\
                            station_idx, 0, :, :]
                        hist_feature[:, :, 1] = volume_data[real_t - hist_feature_daynum*self.timeslot_daynum: real_t: self.timeslot_daynum,\
                            station_idx, 1, :, :]
                        hist_feature = hist_feature.flatten()

                        feature_vec = np.concatenate((hist_feature, last_feature))
                        feature_vec = np.concatenate((feature_vec, nbhd_feature))

                        long_term_lstm_samples.append(feature_vec)
                        long_term_weather_samples.append(weather_data[real_t])
                    lstm_att_features[att_lstm_cnt].append(np.array(long_term_lstm_samples))
                    weather_att_features[att_lstm_cnt].append(np.array(long_term_weather_samples))
                
                """
                label
                    size: 2
                    including:
                        1. outflow
                        2. inflow
                # the target station's inflow and outflow is in (5, 4) grid
                # not done yet
                """
                labels.append(flow_data[t, station_idx, :, 5, 4])

        for i in range(short_term_lstm_seq_len):
            cnn_features[i] = np.array(cnn_features[i])
            flow_features[i] = np.array(flow_features[i])
        short_term_lstm_features = np.array(short_term_lstm_features)
        weather_features = np.array(weather_features)
        
        output_cnn_att_features = []
        output_flow_att_features = []
        for i in range(att_lstm_num):
            lstm_att_features = np.array(lstm_att_features[i])
            flow_att_features = np.array(flow_att_features[i])
            for j in range(long_term_lstm_seq_len):
                cnn_att_features[i][j] = np.array(cnn_att_features[i][j])
                flow_att_features[i][j] = np.array(flow_att_features[i][j])
                output_cnn_att_features.append(cnn_att_features[i][j])
                output_flow_att_features.append(flow_att_features[i][j])
        labels = np.array(labels)

        return output_cnn_att_features, output_flow_att_features, lstm_att_features, weather_att_features,\
            cnn_features, flow_features, short_term_lstm_features, weather_features,\
            labels