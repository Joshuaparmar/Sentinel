import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from Sentinel.dataset.hpc_dataset.utils.timefeatures import time_features
import warnings
from Sentinel.dataset.hpc_dataset.utils.augmentation import run_augmentation_single
from Sentinel.dataset.hpc_dataset.data_provider.read_data import Data_wrapper

# warnings.filterwarnings('ignore')

def slide_rang(data, slide_win, slide_stride, pred_win, mode=''):
        hostname_groups = data.groupby('hostname',sort=False)
        range_list = []
        for name, group in hostname_groups:
            if type(data) == pd.DataFrame:
                total_time_len, node_num = group.shape
            else:
                node_num, total_time_len = group.shape
            if total_time_len == 0:
                continue
            idx_rang = range(slide_win, total_time_len - pred_win +1, slide_stride)
            
            rang = []
            for idx in idx_rang:
                rang.append([group.index[i] for i in range(idx-slide_win, idx+pred_win)])
            range_list.append(rang)
        result = []
        if mode == 'test_ignoresync':
            result = []
            for r in range_list:
                result.extend(r)
            return np.array(result)
        else:
            result = []
            for elements in zip(*range_list):
                for list_ in elements:
                    result.append(list_)
        return np.array(result)

def get_node_rang(num_total, num_nodes, padding = False):
    full_chunks = num_total // num_nodes
    remainder = num_total % num_nodes
    result = []
    start = 0
    if padding:
        if remainder == 0:
            for i in range(full_chunks):
                end = start + num_nodes
                result.append(np.arange(start, end))
                start = end
            return np.stack(result)
        else:
            for i in range(full_chunks):
                end = start + num_nodes
                result.append(np.arange(start, end))
                start = end  
            last_array = np.arange(start, num_total)
            padding_length = num_nodes - len(last_array)
            padded_last = np.pad(last_array, (0, padding_length), 'constant', constant_values=(-1))
            result.append(padded_last)
            return np.stack(result)
    else:
        for i in range(full_chunks):
            end = start + num_nodes
            result.append(np.arange(start, end))
            start = end
        return np.stack(result)

class Dataset_OLCF(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='M', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        self.data = Data_wrapper(args, flag)
        self.scaler = self.data.scaler
        df = self.data.df
        features_col = self.data.features_col
        self.features_col_len = len(features_col)
        self.scale = scale
        if features == 'S':
            features_col = [target]

        if size == None:
            self.seq_len = 15 * 4 * 6
            self.label_len = 15 * 6
            self.pred_len = 15 * 6
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        # Define cache file path
        cache_dir = './cache'
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file_path = os.path.join(cache_dir, f'rang_cache_{self.data.file_suffix}_{self.seq_len}_{self.pred_len}.npy')

        self.process(df)
        # df['timestamp'] = pd.to_datetime(df['timestamp'])
        df_stamp = df[['timestamp']]
        if timeenc == 0:
            df_stamp['year'] = df_stamp.timestamp.apply(lambda row: row.year, 1)
            df_stamp['month'] = df_stamp.timestamp.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.timestamp.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.timestamp.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.timestamp.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.timestamp.apply(lambda row: row.minute, 1)
            df_stamp['second'] = df_stamp.minute.map(lambda row: row.second, 1)
            data_stamp = df_stamp.drop(['timestamp'], 1).values
        elif timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['timestamp'].values), freq=freq)
            data_stamp = data_stamp.transpose(1, 0)
        self.data_stamp = data_stamp

        df_np = df[features_col].values
        self.df_np = torch.from_numpy(df_np.copy())

    def __len__(self):
        return len(self.rang)

    def process(self, data):
        if os.path.exists(self.cache_file_path):
            print(f"Loading cached rang from {self.cache_file_path}")
            self.rang = np.load(self.cache_file_path, allow_pickle=True)
        else:
            self.slide_stride = 1
            self.rang = slide_rang(data, self.seq_len, self.slide_stride, self.pred_len)
            np.save(self.cache_file_path, self.rang)
            print(f"Saved rang to {self.cache_file_path}")
        
    def __getitem__(self, index):
        i_win = self.rang[index]
        seq_x = self.df_np[i_win[:self.seq_len]]
        seq_y = self.df_np[i_win[-self.pred_len-self.label_len:]]
        seq_x_mark = self.data_stamp[i_win[:self.seq_len]]
        seq_y_mark = self.data_stamp[i_win[-self.pred_len-self.label_len:]]

        # label = self.df_np[i_win[-self.pred_len-self.label_len:],-2]
        # hostname_id = self.df_np[i_win[0],-3]
        # allocation_id = self.df_np[i_win[-self.pred_len-self.label_len:],-4]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_OLCF_anomaly(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='M', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        self.data = Data_wrapper(args, flag+'_anomaly')
        self.scaler = self.data.scaler
        self.flag = flag
        df = self.data.df
        features_col = self.data.features_col
        label_col = 'label'
        self.features_col_len = len(features_col)
        self.scale = scale
        if features == 'S':
            features_col = [target]

        if size == None:
            self.seq_len = 15 * 4 * 6
            self.label_len = 15 * 6
            self.pred_len = 15 * 6
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        # Define cache file path
        cache_dir = './cache'
        os.makedirs(cache_dir, exist_ok=True)
        # if flag == 'test':
        #     self.cache_file_path = os.path.join(cache_dir, f'rang_cache_{flag}_anomaly_{args.anomaly_tolerance}_{self.seq_len}_{self.pred_len}.npy')
        # else:
        self.cache_file_path = os.path.join(cache_dir, f'rang_cache_{self.data.file_suffix}_{self.seq_len}_{self.pred_len}.npy')

        self.process(df)
        # df_stamp = df[['timestamp']]
        # if timeenc == 0:
        #     df_stamp['year'] = df_stamp.timestamp.apply(lambda row: row.year, 1)
        #     df_stamp['month'] = df_stamp.timestamp.apply(lambda row: row.month, 1)
        #     df_stamp['day'] = df_stamp.timestamp.apply(lambda row: row.day, 1)
        #     df_stamp['weekday'] = df_stamp.timestamp.apply(lambda row: row.weekday(), 1)
        #     df_stamp['hour'] = df_stamp.timestamp.apply(lambda row: row.hour, 1)
        #     df_stamp['minute'] = df_stamp.timestamp.apply(lambda row: row.minute, 1)
        #     df_stamp['second'] = df_stamp.minute.map(lambda row: row.second, 1)
        #     data_stamp = df_stamp.drop(['timestamp'], 1).values
        # elif timeenc == 1:
        #     data_stamp = time_features(pd.to_datetime(df_stamp['timestamp'].values), freq=freq)
        #     data_stamp = data_stamp.transpose(1, 0)
        # self.data_stamp = data_stamp

        df_np = df[features_col].values
        self.df_np = torch.from_numpy(df_np.copy())
        df_label_np = df[label_col].values
        self.df_label_np = torch.from_numpy(df_label_np.copy())

    def __len__(self):
        return len(self.rang)

    def process(self, data):
        if os.path.exists(self.cache_file_path):
            print(f"Loading cached rang from {self.cache_file_path}")
            self.rang = np.load(self.cache_file_path, allow_pickle=True)
        else:
            self.slide_stride = 1
            self.rang = slide_rang(data, self.seq_len, self.slide_stride, self.pred_len, mode = 'test_ignoresync')
            np.save(self.cache_file_path, self.rang)
            print(f"Saved rang to {self.cache_file_path}")
        
    def __getitem__(self, index):
        i_win = self.rang[index]
        seq_x = self.df_np[i_win[:self.seq_len]]
        if self.flag == 'test':
            labels = self.df_label_np[i_win[:self.seq_len]]
        else:
            i_win_0 = self.rang[0]
            labels = self.df_label_np[i_win_0[:self.seq_len]]
        # seq_y = self.df_np[i_win[-self.pred_len-self.label_len:]]
        # seq_x_mark = self.data_stamp[i_win[:self.seq_len]]
        # seq_y_mark = self.data_stamp[i_win[-self.pred_len-self.label_len:]]

        # label = self.df_np[i_win[-self.pred_len-self.label_len:],-2]
        # hostname_id = self.df_np[i_win[0],-3]
        # allocation_id = self.df_np[i_win[-self.pred_len-self.label_len:],-4]
        return seq_x, labels

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_OLCF_Cluster(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='power', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        flag = flag + '_cluster'
        self.data = Data_wrapper(args, flag)
        self.scaler = self.data.scaler
        df = self.data.df
        features_col = self.data.features_col
        self.features_col_len = len(features_col)
        self.scale = scale
        if features == 'S':
            features_col = [target]

        if size == None:
            self.seq_len = 15 * 4 * 6
            self.label_len = 15 * 6
            self.pred_len = 15 * 6
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        # Define cache file path
        cache_dir = './cache'
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file_path = os.path.join(cache_dir, f'rang_cache_{self.data.file_suffix}_{self.seq_len}_{self.pred_len}.npy')

        # self.process(df)

        # df['timestamp'] = pd.to_datetime(df['timestamp'])
        df_stamp = df[['timestamp']]
        if timeenc == 0:
            df_stamp['year'] = df_stamp.timestamp.apply(lambda row: row.year, 1)
            df_stamp['month'] = df_stamp.timestamp.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.timestamp.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.timestamp.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.timestamp.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.timestamp.apply(lambda row: row.minute, 1)
            df_stamp['second'] = df_stamp.minute.map(lambda row: row.second, 1)
            data_stamp = df_stamp.drop(['timestamp'], 1).values
        elif timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['timestamp'].values), freq=freq)
            data_stamp = data_stamp.transpose(1, 0)
        self.data_stamp = data_stamp

        df_np = df[features_col].values
        self.df_np = torch.from_numpy(df_np.copy())

    def __len__(self):
        return len(self.df_np) - self.seq_len - self.pred_len + 1

    # def process(self, data):
    #     if os.path.exists(self.cache_file_path):
    #         print(f"Loading cached rang from {self.cache_file_path}")
    #         self.rang = np.load(self.cache_file_path, allow_pickle=True)
    #     else:
    #         self.slide_stride = 1
    #         self.rang = slide_rang(data, self.seq_len, self.slide_stride, self.pred_len)
    #         np.save(self.cache_file_path, self.rang)
    #         print(f"Saved rang to {self.cache_file_path}")
        
    def __getitem__(self, index):
        seq_x = self.df_np[index:index+self.seq_len]
        seq_y = self.df_np[index+self.seq_len-self.label_len:index+self.seq_len+self.pred_len]
        seq_x_mark = self.data_stamp[index:index+self.seq_len]
        seq_y_mark = self.data_stamp[index+self.seq_len-self.label_len:index+self.seq_len+self.pred_len]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_OLCF_MMD(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='M', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        self.args = args
        flag = flag + '_mmd'
        self.data = Data_wrapper(args, flag)
        self.data.category_hostname()
        self.scaler = self.data.scaler
        df = self.data.df
        features_col = self.data.features_col
        self.features_col_len = len(features_col)
        self.scale = scale
        if features == 'S':
            features_col = [target]

        if size == None:
            self.seq_len = 15 * 4 * 6
            self.label_len = 15 * 6
            self.pred_len = 15 * 6
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        # Define cache file path
        cache_dir = './cache'
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file_path = os.path.join(cache_dir, f'rang_cache_{self.data.file_suffix}_{self.seq_len}_{self.pred_len}.npy')

        self.process(df)
        # df['timestamp'] = pd.to_datetime(df['timestamp'])
        df_stamp = df[['timestamp']]
        if timeenc == 0:
            df_stamp['year'] = df_stamp.timestamp.apply(lambda row: row.year, 1)
            df_stamp['month'] = df_stamp.timestamp.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.timestamp.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.timestamp.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.timestamp.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.timestamp.apply(lambda row: row.minute, 1)
            df_stamp['second'] = df_stamp.minute.map(lambda row: row.second, 1)
            data_stamp = df_stamp.drop(['timestamp'], 1).values
        elif timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['timestamp'].values), freq=freq)
            data_stamp = data_stamp.transpose(1, 0)
        self.data_stamp = data_stamp

        df_np = df[features_col].values
        self.df_np = torch.from_numpy(df_np.copy())

        hostname_ids = df['hostname_id'].values
        self.hostname_ids = torch.from_numpy(hostname_ids.copy())

        job_ids = df['allocation_id'].values
        self.job_ids = torch.from_numpy(job_ids.copy())

        rack_ids = df['rack_id'].values
        self.rack_ids = torch.from_numpy(rack_ids.copy())

        cabinet_ids = df['cabinet_id'].values
        self.cabinet_ids = torch.from_numpy(cabinet_ids.copy())

        node_in_rack_ids = df['node_in_rack'].values
        self.node_in_rack_ids = torch.from_numpy(node_in_rack_ids.copy())

    def __len__(self):
        return len(self.node_rang)

    def process(self, data):
        if os.path.exists(self.cache_file_path):
            print(f"Loading cached rang from {self.cache_file_path}")
            self.rang = np.load(self.cache_file_path, allow_pickle=True)
        else:
            self.slide_stride = 1
            self.rang = slide_rang(data, self.seq_len, self.slide_stride, self.pred_len)
            np.save(self.cache_file_path, self.rang)
            print(f"Saved rang to {self.cache_file_path}")
        self.node_rang = get_node_rang(self.rang.shape[0],self.args.num_hostname)
        
    def __getitem__(self, index):
        h_win = self.node_rang[index]
        i_win = self.rang[h_win]
        seq_x = self.df_np[i_win[:,:self.seq_len]]
        seq_y = self.df_np[i_win[:,-self.pred_len-self.label_len:]]
        seq_x_mark = self.data_stamp[i_win[:,:self.seq_len]]
        seq_y_mark = self.data_stamp[i_win[:,-self.pred_len-self.label_len:]]

        seq_x_hostname_id = self.hostname_ids[i_win[:,:self.seq_len]]
        seq_x_job_id = self.job_ids[i_win[:,:self.seq_len]]
        seq_x_rack_id = self.rack_ids[i_win[:,:self.seq_len]]
        seq_x_cabinet_id = self.cabinet_ids[i_win[:,:self.seq_len]]
        seq_x_node_in_rack_id = self.node_in_rack_ids[i_win[:,:self.seq_len]]
        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_x_hostname_id, seq_x_job_id, seq_x_rack_id, seq_x_cabinet_id, seq_x_node_in_rack_id

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
