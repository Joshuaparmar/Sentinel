# This script assume you have run generate_olcf.py and generate_olcf_mini.py first which generate data under `data/olcf_mini`
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
parent_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import pandas as pd
from pathlib import Path
import numpy as np
import random
import glob
import gc
from copy import copy
from Sentinel.dataset.hpc_dataset.src.generate_olcf import get_data, read_failures, slice_df_by_time_range
from Sentinel.dataset.hpc_dataset.src.generate_olcf_mini import read_features
import shutil

def read_data_wrapper(dataset_path='data/olcf_mini', filename_list=['202001/20200101.parquet']):
    df_list = []
    for filename in filename_list:
        powertemp = get_data(path=f'{dataset_path}/{filename}')
        df_list.append(powertemp)
    df = pd.concat(df_list)
    return df

def generate_train_data(dataset_path='data/olcf_mini', filename_list=['202001/20200101.parquet'], output_file_path='data/olcf_task'):
    df = read_data_wrapper(dataset_path, filename_list)
    df_hostname = df[['hostname']+['ps0_input_power','ps1_input_power']].groupby('hostname').mean()
    df_hostname['input_power'] = df_hostname[['ps0_input_power','ps1_input_power']].mean(axis=1)
    df_hostname['input_power'] = df_hostname['input_power'].fillna(0)
    df_hostname.sort_values(by=['input_power'], ascending=False, inplace=True)
    active_instances = df_hostname.head(100).index.unique().tolist()
    df= df[df['hostname'].isin(active_instances)]
    output_path = Path(output_file_path+'/train.parquet')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(by=["timestamp","hostname"], inplace=True)
    df.to_parquet(output_path)     

    with open(f"{output_file_path}/active_instances.txt","w") as f:
        for item in df['hostname'].unique():
            f.write("%s\n" % item)

def generate_test_data(dataset_path='data/olcf_mini', filename_list=['202001/20200102.parquet'], output_file_path='data/olcf_task'):
    df = read_data_wrapper(dataset_path, filename_list)
    train_hostnames_file = open(f"{output_file_path}/active_instances.txt", "r")
    train_hostnames = []
    for hostname in train_hostnames_file:
        train_hostnames.append(hostname.strip())

    test_hostnames = df['hostname'].unique().tolist()
    test_hostnames = set(test_hostnames) & set(train_hostnames)
    
    df = df[df['hostname'].isin(test_hostnames)]

    output_path = Path(output_file_path+'/test.parquet')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(by=["timestamp","hostname"], inplace=True)
    df = slice_df_by_time_range(df, [["2020-01-02 04:00:00", "2020-01-03 00:00:00"]])
    df.to_parquet(output_path)     

def generate_val_data(dataset_path='data/olcf_mini', filename_list=['202001/20200102.parquet'], output_file_path='data/olcf_task'):
    df = read_data_wrapper(dataset_path, filename_list)
    train_hostnames_file = open(f"{output_file_path}/active_instances.txt", "r")
    train_hostnames = []
    for hostname in train_hostnames_file:
        train_hostnames.append(hostname.strip())

    test_hostnames = df['hostname'].unique().tolist()
    test_hostnames = set(test_hostnames) & set(train_hostnames)
    
    df = df[df['hostname'].isin(test_hostnames)]

    output_path = Path(output_file_path+'/val.parquet')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(by=["timestamp","hostname"], inplace=True)
    df = slice_df_by_time_range(df, [["2020-01-02 00:00:00", "2020-01-02 03:59:59"]])
    df.to_parquet(output_path)     

def generate_test_anomaly_data(dataset_path='data/olcf_mini', filename_list=['202001/20200119.parquet'], output_file_path='data/olcf_task'):
    df_list = []
    for filename in filename_list:
        powertemp = get_data(path=f'{dataset_path}/{filename}')
        df_list.append(powertemp)
    df = pd.concat(df_list)

    train_hostnames_file = open(f"{output_file_path}/active_instances.txt", "r")
    train_hostnames = []
    for hostname in train_hostnames_file:
        train_hostnames.append(hostname.strip())

    test_hostnames = df['hostname'].unique().tolist()
    test_hostnames = set(test_hostnames) & set(train_hostnames)
    
    df = df[df['hostname'].isin(test_hostnames)]

    output_path = Path(output_file_path+'/test_anomaly.parquet')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(by=["timestamp","hostname"], inplace=True)
    df.to_parquet(output_path)     

def generate_cluster_data(dataset_path='data/olcf_mini', filename_list=['202001/20200101.parquet','202001/20200102.parquet'], output_file_path='data/olcf_task'):
    df = read_data_wrapper(dataset_path, filename_list)
    df_train = slice_df_by_time_range(df, [["2020-01-01 00:00:00", "2020-01-01 23:59:59"]])
    df_val = slice_df_by_time_range(df, [["2020-01-02 00:00:00", "2020-01-02 03:59:59"]])
    df_test = slice_df_by_time_range(df, [["2020-01-02 04:00:00", "2020-01-03 00:00:00"]])

    Path(output_file_path).mkdir(parents=True, exist_ok=True)
    df_train.to_parquet(output_file_path+'/train_cluster_full.parquet')    
    df_val.to_parquet(f'{output_file_path}/val_cluster_full.parquet')
    df_test.to_parquet(f'{output_file_path}/test_cluster_full.parquet') 

    feature_cols = read_features(path=f'{output_file_path}')
    non_feature_cols = None
    saved_filename_list=['train_cluster_full.parquet','test_cluster_full.parquet','val_cluster_full.parquet']
    for filename in saved_filename_list:
        df = get_data(path=f'{output_file_path}/{filename}')
        if non_feature_cols is None:
            non_feature_cols = [col for col in df.columns if col not in feature_cols]
        gpu_power_feats = ['p0_gpu0_power','p0_gpu1_power','p0_gpu2_power','p1_gpu0_power','p1_gpu1_power','p1_gpu2_power']
        cpu_power_feats = ['p0_power','p1_power']
        gpu_core_temp_feats = ['gpu0_core_temp','gpu1_core_temp','gpu2_core_temp','gpu3_core_temp','gpu4_core_temp','gpu5_core_temp']
        gpu_mem_temp_feats = ['gpu0_mem_temp','gpu1_mem_temp','gpu2_mem_temp','gpu3_mem_temp','gpu4_mem_temp','gpu5_mem_temp']
        power_feats = ['ps0_input_power','ps1_input_power']
        cpu_temp_feats = ['p0_temp_mean','p1_temp_mean']
        df['gpu_power'] = df[gpu_power_feats].sum(axis=1)
        df['gpu_core_temp'] = df[gpu_core_temp_feats].mean(axis=1)
        df['gpu_mem_temp'] = df[gpu_mem_temp_feats].mean(axis=1)
        df['cpu_power'] = df[cpu_power_feats].sum(axis=1)
        df['cpu_temp'] = df[cpu_temp_feats].mean(axis=1)
        df['power'] = df[power_feats].sum(axis=1)
        df = df[non_feature_cols+['gpu_power','gpu_core_temp','gpu_mem_temp','cpu_power','cpu_temp','power']]
        feats = ['gpu_power','gpu_core_temp','gpu_mem_temp','cpu_power','cpu_temp','power']
        df_cluster = df[['timestamp']+feats].groupby(['timestamp']).aggregate(
            {
                'gpu_power': 'sum',
                'gpu_core_temp': 'mean',
                'gpu_mem_temp': 'mean',
                'cpu_power': 'sum',
                'cpu_temp': 'mean',
                'power': 'sum'
            }
        )
        df_cluster = df_cluster.reset_index()
        
        Path(output_file_path).mkdir(parents=True, exist_ok=True)
        df_cluster.sort_values(by=["timestamp"], inplace=True)
        new_filename = filename.replace('_full','')
        df_cluster.to_parquet(f'{output_file_path}/{new_filename}')
        os.remove(f'{output_file_path}/{filename}')
        feature_list_path = f"{output_file_path}/features_cluster.txt"
        if not os.path.exists(feature_list_path):
            print(f"\nSaving combined feature list to: {feature_list_path}")
            with open(feature_list_path, "w") as f:
                for item in sorted(list(feats)):
                    f.write("%s\n" % item)

if __name__ == "__main__":
    dataset_path = 'data/olcf_mini'

    filename_list = ['202001/20200101.parquet']
    output_file_path = 'data/olcf_task'
    generate_train_data(dataset_path, filename_list, output_file_path)
    
    filename_list = ['202001/20200102.parquet']
    output_file_path = 'data/olcf_task'
    generate_test_data(dataset_path, filename_list, output_file_path)
    
    filename_list = ['202001/20200102.parquet']
    output_file_path = 'data/olcf_task'
    generate_val_data(dataset_path, filename_list, output_file_path)

    filename_list = ['202001/20200119.parquet']
    output_file_path = 'data/olcf_task'
    generate_test_anomaly_data(dataset_path, filename_list, output_file_path)

    source_features_path = Path(dataset_path) / 'features.txt'
    destination_features_path = Path(output_file_path) / 'features.txt'
    destination_features_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source_features_path, destination_features_path)

    generate_cluster_data(dataset_path, ['202001/20200101.parquet','202001/20200102.parquet'], output_file_path)