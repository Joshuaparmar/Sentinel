# This script assume you have run generate_olcf.py first which generate data under `data/olcf`
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
from Sentinel.dataset.hpc_dataset.src.generate_olcf import get_data

def read_features(path):
    if not os.path.isfile(f"{path}/features.txt"):
        print(f"{path}/features.txt not found")
        return None
    else:
        feature_file = open(f"{path}/features.txt", "r")
        feature_list = []
        for ft in feature_file:
            if "#" not in ft:
                feature_list.append(ft.strip())
        return feature_list

def summary_cpu_info(df):
    # Input> dataframe df owns column: p0_core[0,1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23]_temp, p1_core[0,1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23]_temp
    # Output> dataframe df: p0_temp_[mean,min,max], p1_temp_[mean,min,max]
    p0_cols = [f'p0_core{i}_temp' for i in range(24) if i != 13]
    df['p0_temp_mean'] = df[p0_cols].mean(axis=1)
    df['p0_temp_min'] = df[p0_cols].min(axis=1)
    df['p0_temp_max'] = df[p0_cols].max(axis=1)
    df.drop(columns=p0_cols, inplace=True)
    p1_cols = [f'p1_core{i}_temp' for i in range(24) if i != 13]
    df['p1_temp_mean'] = df[p1_cols].mean(axis=1)
    df['p1_temp_min'] = df[p1_cols].min(axis=1)
    df['p1_temp_max'] = df[p1_cols].max(axis=1)
    df.drop(columns=p1_cols, inplace=True)
    return df


def rename_features(df, features):
    new_features = []
    for feature in features:
        if feature.startswith('gpu'):
            feature = feature.replace('gpu0', 'p0_gpu0').replace('gpu1', 'p0_gpu1').replace('gpu2', 'p0_gpu2').replace('gpu3', 'p1_gpu0').replace('gpu4', 'p1_gpu1').replace('gpu5', 'p1_gpu2')
        elif feature == 'p0_power':
            feature = 'p0_core_power'
        elif feature == 'p1_power':
            feature = 'p1_core_power'
        new_features.append(feature)
    feat_dicts = {}
    for i, feature in enumerate(features):
        feat_dicts[feature] = new_features[i]
    df.rename(columns=feat_dicts, inplace=True)
    features = new_features
    return features, df

def get_features_group(features):
    p0_gpu_power_fmap = []
    p1_gpu_power_fmap = []
    p0_cpu_power_fmap = []
    p1_cpu_power_fmap = []
    p0_gpu_core_temp_fmap = []
    p1_gpu_core_temp_fmap = []
    p0_gpu_mem_temp_fmap = []
    p1_gpu_mem_temp_fmap = []
    p0_cpu_temp_fmap = []
    p1_cpu_temp_fmap = []
    cpu_temp_mean_fmap = []
    cpu_temp_max_fmap = []
    input_power_fmap = []
    others = []
    for feature in features:
        if feature.startswith('p0_gpu') and feature.endswith('power'):
            p0_gpu_power_fmap.append(feature)
        elif feature.startswith('p1_gpu') and feature.endswith('power'):
            p1_gpu_power_fmap.append(feature)
        elif feature.startswith('p0_core') and feature.endswith('power'):
            p0_cpu_power_fmap.append(feature)
        elif feature.startswith('p1_core') and feature.endswith('power'):
            p1_cpu_power_fmap.append(feature)
        elif feature.startswith('p0_gpu') and feature.endswith('temp') and 'core' in feature:
            p0_gpu_core_temp_fmap.append(feature)
        elif feature.startswith('p1_gpu') and feature.endswith('temp') and 'core' in feature:
            p1_gpu_core_temp_fmap.append(feature)
        elif feature.startswith('p0_gpu') and feature.endswith('temp') and 'mem' in feature:
            p0_gpu_mem_temp_fmap.append(feature)
        elif feature.startswith('p1_gpu') and feature.endswith('temp') and 'mem' in feature:
            p1_gpu_mem_temp_fmap.append(feature)
        elif feature.startswith('p0_core') and feature.endswith('temp'):
            p0_cpu_temp_fmap.append(feature)
        elif feature.startswith('p1_core') and feature.endswith('temp'):
            p1_cpu_temp_fmap.append(feature)
        elif feature in ['p0_temp_mean', 'p1_temp_mean']:
            cpu_temp_mean_fmap.append(feature)
        elif feature in ['p0_temp_max', 'p1_temp_max']:
            cpu_temp_max_fmap.append(feature)
        elif feature in ['ps0_input_power','ps1_input_power']:
            input_power_fmap.append(feature)
        else:
            others.append(feature)
    return p0_gpu_power_fmap, p1_gpu_power_fmap, p0_cpu_power_fmap, p1_cpu_power_fmap, p0_gpu_core_temp_fmap, p1_gpu_core_temp_fmap, p0_gpu_mem_temp_fmap, p1_gpu_mem_temp_fmap, p0_cpu_temp_fmap, p1_cpu_temp_fmap, input_power_fmap, cpu_temp_mean_fmap, cpu_temp_max_fmap, others

if __name__ == "__main__":
    olcf_data_base_dir = Path('data/olcf')
    olcf_mini_data_base_dir = Path('data/olcf_mini')
    
    olcf_mini_data_base_dir.mkdir(parents=True, exist_ok=True)

    print(f"Finding data files in: {olcf_data_base_dir}")
    data_files = list(olcf_data_base_dir.rglob('*.parquet'))
    print(f"Found {len(data_files)} raw parquet files.")

    features = read_features(str(olcf_data_base_dir))
    non_features_columns = None
    new_features = None
    for i, raw_file in enumerate(data_files):
        print(f"\nProcessing file {i+1}/{len(data_files)}: {raw_file}")
        try:
            relative_path = raw_file.relative_to(olcf_data_base_dir)
            output_file = olcf_mini_data_base_dir / relative_path
            file_date_str = str(raw_file).split('/')[-1].split('.')[0]
            if not output_file.exists():
                print("Reading raw data...")
                df = get_data(path=str(raw_file))
                if non_features_columns is None:
                    non_features_columns = [col for col in df.columns if col not in features]
                df = summary_cpu_info(df)
                if new_features is None:
                    new_features = [col  for col in df.columns if col not in non_features_columns]
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(output_path)
                del df
                gc.collect()
        except Exception as e:
            print(f"Error processing file {raw_file}: {e}")
            gc.collect()

    feature_list_path = olcf_mini_data_base_dir / "features.txt"
    if not feature_list_path.exists():
        print(f"\nSaving combined feature list to: {feature_list_path}")
        with open(feature_list_path, "w") as f:
            for item in sorted(list(new_features)):
                f.write("%s\n" % item)
    else:
        print(f"\nFeature list file already exists at {feature_list_path}, skipping save.")
    print("\nProcessing complete.")