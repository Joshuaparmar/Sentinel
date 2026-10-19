import pandas as pd
import numpy as np
from Sentinel.dataset.hpc_dataset.data_provider.normalization import scaler_wrapper
import os
import glob
from sklearn.model_selection import train_test_split
from Sentinel.dataset.hpc_dataset.data_provider.feat_prep import get_feature_map
from Sentinel.dataset.hpc_dataset.data_provider.relabel_anomaly import relabel_test_anomaly

class Data:
    def __init__(self, path,filter=None,args=None):
        self.path = path
        self.filter = filter
        self.args = args
        if hasattr(args, 'data_fillna'):
            self.df = self.read_data(fillna=args.data_fillna)
        else:
            self.df = self.read_data() 
        
    def read_data(self, fillna=True):
        if self.path.split(".")[-1] == "csv":
            df_joined = (
                pd.read_csv(
                    self.path,
                )
            )
        elif self.path.split(".")[-1] == "parquet" or bool(glob.glob(os.path.join(self.path, '*.parquet'))):
            df_joined = (
                pd.read_parquet(
                    self.path,
                    engine="pyarrow"
                )
            )
        else:
            print("[!] File type not supported")
            return pd.DataFrame()

        if hasattr(self.args, 'timestamp_col'):
            df_joined.rename(columns={self.args.timestamp_col:'timestamp'}, inplace=True)
        if hasattr(self.args, 'hostname_col'):
            df_joined.rename(columns={self.args.hostname_col:'hostname'}, inplace=True)
        if hasattr(self.args, 'job_id_col'):
            df_joined.rename(columns={self.args.job_id_col:'allocation_id'}, inplace=True)
        if hasattr(self.args, 'label_col'):
            df_joined.rename(columns={self.args.label_col:'label'}, inplace=True)
        if hasattr(self.args, 'anomaly_source_col'):
            df_joined.rename(columns={self.args.anomaly_source_col:'anomaly_source'}, inplace=True)

        if 'label' not in df_joined.columns:
            df_joined['label'] = 0
        # if 'status' in df_joined.columns:
        #     df_joined['status'] = df_joined['status'].fillna('Unknown')
        if 'hostname' in df_joined.columns:
            df_joined['hostname_id'] = df_joined['hostname'].astype('category').cat.codes
        # if 'GPU' in df_joined.columns:
        #     if df_joined[df_joined['label']==0]['GPU'].nunique()==1 or df_joined[df_joined['label']==0]['GPU'].nunique()==0:
        #         df_joined.loc[df_joined['label']==0, 'GPU'] = -1
        #     else:
        #         print('GPU error ID appears when the record is normal!')
        if pd.api.types.is_datetime64_any_dtype(df_joined['timestamp']):
            pass
        elif pd.api.types.is_string_dtype(df_joined['timestamp']):
            unique_len_timestamp_str = df_joined['timestamp'].str.len().unique()
            if len(unique_len_timestamp_str) > 1:
                print('[WARN]: Timestamp column contains different length of string, please check the format')
                df_joined['timestamp'] = df_joined['timestamp'].str[:19]
                print('[WARN]: Timestamp column has been truncated to 19 characters, in order to fit the format: YYYY-MM-DDTHH:MM:SS')
            elif len(unique_len_timestamp_str) == 1 and (unique_len_timestamp_str[0] == 19):
                pass
            elif len(unique_len_timestamp_str) == 1 and (unique_len_timestamp_str[0] == 32):
                df_joined['timestamp'] = df_joined['timestamp'].str[:26]
            elif len(unique_len_timestamp_str) == 1 and (unique_len_timestamp_str[0] == 26):
                pass
            else:
                raise ValueError('Timestamp column contains invalid string length')
            
            df_joined['timestamp'] = pd.to_datetime(df_joined['timestamp'])
        elif pd.api.types.is_integer_dtype(df_joined['timestamp']):
            df_joined['timestamp'] = pd.to_datetime(df_joined['timestamp'])
        else:
            raise ValueError('Timestamp column must be datetime64 or string or int')
        
        df_joined= df_joined.sort_values(by=["timestamp"])
        if self.filter is not None and self.filter != {}:
            df_joined = filter_df(df_joined, self.filter)
        df_joined= self.remove_original_index(df_joined)
        df_joined = df_joined.reset_index(drop=True)
        nan_columns = df_joined.columns[df_joined.isna().any()].tolist()
        if len(nan_columns)>0:
            if fillna:
                df_joined = df_joined.fillna(0)
        
        return df_joined
    
    def set_abnormal(self, time_ranges):
        return set_val_by_time_range(self.df, time_ranges)

    def by_time_range(self, time_ranges):
        return slice_df_by_time_range(self.df, time_ranges)
        
    def remove_original_index(self, df):
        if 'index' in df.columns:
            df= df.drop('index', axis=1)
        if 'Unnamed: 0' in df.columns:
            df= df.drop('Unnamed: 0', axis=1)
        return df


def set_val_by_time_range(df, time_ranges, column='label',val=1, farmnodes=None):
    if column not in df.columns:
        df[column] = 0
    for start_time, end_time in time_ranges:
        if start_time > end_time:
            raise ValueError('Start time must be less than end time')
        if start_time == '':
            mask = (df['timestamp'] <= end_time)
        elif end_time == '':
            mask = (df['timestamp'] >= start_time)
        else:
            mask = (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)
        if farmnodes!=None:
            mask = mask & (df['hostname'].isin(farmnodes))
        if isinstance(val, list):
            df.loc[mask,column] = df.loc[mask,column].apply(lambda x: val)
        elif isinstance(val, int) or isinstance(val, float) or isinstance(val, str):
            df.loc[mask,column] = val
    return df

def slice_df_by_time_range(df, time_ranges,timestamp_col='timestamp'): 
    sliced_df = pd.DataFrame()
    for start_time, end_time in time_ranges:
        if start_time > end_time:
            raise ValueError('Start time must be less than end time')
        if start_time == '':
            mask = (df[timestamp_col] < end_time)
        elif end_time == '':
            mask = (df[timestamp_col] >= start_time)
        else:
            mask = (df[timestamp_col] >= start_time) & (df[timestamp_col] < end_time)
        df_range = df.loc[mask]
        sliced_df = pd.concat([sliced_df, df_range])
    sliced_df = sliced_df.reset_index(drop=True)
    return sliced_df

def get_mask_by_time_range(df, time_ranges,farmnodes=None):
    for start_time, end_time in time_ranges:
        if start_time > end_time:
            raise ValueError('Start time must be less than end time')
        if start_time == '':
            mask = (df['timestamp'] <= end_time)
        elif end_time == '':
            mask = (df['timestamp'] >= start_time)
        else:
            mask = (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)
        if farmnodes!=None:
            if isinstance(farmnodes, list):
                mask = mask & (df['hostname'].isin(farmnodes))
            elif isinstance(farmnodes, str):
                mask = mask & (df['hostname']==farmnodes)
            else:
                raise ValueError('farmnodes must be a list or a string')
    return mask

def filter_df(df, filter):
    for key in filter.keys():
        if key not in df.columns:
            print("[!] No column: " + key)
            return pd.DataFrame()
        if isinstance(filter[key], list):
            df = df[df[key].isin(filter[key])]
        if isinstance(filter[key], str):
            if filter[key].startswith('/') and filter[key].endswith('.*/'):
                df = df[df[key].str.startswith(filter[key][1:-3])]
            else:
                df = df[df[key] == filter[key]]
    return df.reset_index(drop=True)


def sort_wrapper(df,by_column=["hostname","timestamp"]):
    df.sort_values(by=by_column,inplace=True)
    df.reset_index(drop=True,inplace=True)
    return df

def uni_timestamp(df, gap, tolerance):
    init_time = df['timestamp'][0]
    df['uni_timestamp'] = df['timestamp'].apply(lambda x: (x-init_time).total_seconds()//gap)
    return df

def fill_missing_hostname(df):
    df = uni_timestamp(df, gap=60, tolerance=0.1)
    all_timestamps = df['uni_timestamp'].unique()
    all_hostnames = df['hostname'].unique()
    mux = pd.MultiIndex.from_product([all_timestamps, all_hostnames], names=['uni_timestamp', 'hostname'])
    df_full = pd.DataFrame(index=mux).reset_index()
    df_full = pd.merge(df_full, df, how='left', on=['uni_timestamp', 'hostname'])
    return df_full

class Data_wrapper:
    def __init__(self, args, flag):
        self.df = self.load_data(args, flag)
        self.features_col = get_feature_map(args.root_path,flag)
        self.df, self.scaler = scaler_wrapper(self.df, self.features_col, flag, scaler_type='minmax',label=args.data) # Store the scaler instance
        self.args = args
        if 'cluster' in flag:
            self.df = sort_wrapper(self.df,by_column=["timestamp"])
        else:
            self.df = sort_wrapper(self.df,by_column=["timestamp","hostname"])

    def load_data(self, args, flag):
        dataset_path = args.root_path
        if flag == 'train' or flag == 'train_anomaly':
            file_path = dataset_path +'train.parquet'
        elif flag == 'val' or flag == 'val_anomaly':
            file_path = dataset_path +'val.parquet'
        elif flag == 'test':
            file_path = dataset_path +'test.parquet'
        elif flag == 'test_anomaly':
            if args.anomaly_tolerance == '':
                file_path = dataset_path +'test_anomaly.parquet'
            else:
                file_path = dataset_path + 'test_anomaly_'+ args.xid + '_' + args.anomaly_tolerance + '.parquet'
                if not os.path.exists(file_path):
                    # raise ValueError(f"File {file_path} does not exist. Please check the file path and try again.")
                    relabel_test_anomaly(tol=args.anomaly_tolerance,xid=args.xid)

        if flag == 'train_cluster':
            file_path = dataset_path +'train_cluster.parquet'
        elif flag == 'val_cluster':
            file_path = dataset_path +'val_cluster.parquet'
        elif flag == 'test_cluster':
            file_path = dataset_path +'test_cluster.parquet'

        if flag == 'train_mmd':
            file_path = dataset_path +'train_mmd.parquet'
        elif flag == 'val_mmd':
            file_path = dataset_path +'val_mmd.parquet'
        elif flag == 'test_mmd':
            file_path = dataset_path +'test_mmd.parquet'
        self.file_suffix = file_path[len(dataset_path):-8]
        df = Data(file_path, args=args).df
        return df

    def category_hostname(self):
        if 'hostname_id' not in self.df.columns:
            try:
                with open(f"{self.args.root_path}/hostnames.txt", "r") as f:
                    hostname_id_map = {
                        line.strip(): idx for idx, line in enumerate(f)
                    }
            except FileNotFoundError:
                print(f"Error: The file {self.args.root_path}/active_instances.txt was not found.")
                hostname_id_map = {}
            except Exception as e:
                print(f"An error occurred while reading active_instances.txt: {e}")
                hostname_id_map = {}
            if len(hostname_id_map) != self.args.num_hostname:
                raise ValueError(f"Mismatch in number of hostnames. Expected {self.args.num_hostname}, found {len(hostname_id_map)}.")
            self.df['hostname_id'] = self.df['hostname'].map(hostname_id_map)
        if ('rack_id' not in self.df.columns) or ('cabinet_id' not in self.df.columns) or ('node_in_rack_id' not in self.df.columns):
            self.df['rack'] = self.df['hostname'].str.extract(r'([a-h]\d+)n\d+')
            self.df['cabinet'] = self.df['rack'].str.extract(r'([a-h])\d+')
            self.df['node_in_rack'] = self.df['hostname'].str.extract(r'[a-h]\d+n(\d+)')
            self.df['rack_id'] = self.df['rack'].astype('category').cat.codes + 1
            self.df['cabinet_id'] = self.df['cabinet'].astype('category').cat.codes + 1 
            self.df['node_in_rack'] = self.df['node_in_rack'].astype('category').cat.codes + 1 
            self.df.drop(columns=['rack','cabinet'], inplace=True)