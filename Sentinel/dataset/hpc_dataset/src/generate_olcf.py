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
import shutil

def get_data(path, fillna=True,sorted=False):
    if path.split(".")[-1] == "parquet" or bool(glob.glob(os.path.join(path, '*.parquet'))):
        df_joined = (
            pd.read_parquet(
                path,
                engine="pyarrow"
            )
        )
    else:
        print("[!] File type not supported")
        return pd.DataFrame()
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
    if sorted:
        df_joined= df_joined.sort_values(by=["timestamp"])
    return df_joined

def extract_featurelist(df):
    feature_list = []
    for i in df.columns:
        excluded_values = [
            "attack",
            "timestamp",
            "label",
            "timestamp_memory",
            "timestamp_disk",
            "timestamp_slurm",
            "instance",
            "timestamp_network",
            "timestamp_cpu",
            "timestamp_gpu",
            "timestamp_power",
            "timestamp_job",
            "timestamp_job_event",
            "timestamp_job_resrc_usage",
            "timestamp_job_resrc_used",
            "timestamp_job_resrc_alloc",
            "node",
            "node_state",
            "hostname",
            "yymmdd",
        ]
        if i not in excluded_values:
            feature_list.append(i)
    return feature_list

def slice_df_by_time_range(df, time_ranges,timestamp_col='timestamp'): 
    sliced_df = pd.DataFrame()
    for start_time, end_time in time_ranges:
        if start_time > end_time:
            raise ValueError('Start time must be less than end time')
        if start_time == '':
            mask = (df[timestamp_col] <= end_time)
        elif end_time == '':
            mask = (df[timestamp_col] >= start_time)
        else:
            mask = (df[timestamp_col] >= start_time) & (df[timestamp_col] <= end_time)
        df_range = df.loc[mask]
        sliced_df = pd.concat([sliced_df, df_range])
        nan_columns = sliced_df.columns[sliced_df.isna().any()].tolist()
        if len(nan_columns)>0:
            print('Nan columns:',nan_columns)
    sliced_df = sliced_df.reset_index(drop=True)
    return sliced_df

def read_joblog(path='rawdata/10.13139_OLCF_1970187/sanitized_pernode_jobs_full.csv'):
    df = pd.read_csv(path,usecols=['node_name','begin_time','end_time','allocation_id'], engine='pyarrow')
    df.rename(columns={'node_name': 'hostname'}, inplace=True)
    df['begin_time'] = pd.to_datetime(df['begin_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    if df['begin_time'].dt.tz is not None:
        df['begin_time'] = df['begin_time'].dt.tz_localize(None)
    if df['end_time'].dt.tz is not None:
        df['end_time'] = df['end_time'].dt.tz_localize(None)
    df['begin_time'] = df['begin_time'].astype('datetime64[ms]')
    df['begin_yymmdd'] = df['begin_time'].dt.strftime('%Y%m%d')
    df['end_time'] = df['end_time'].astype('datetime64[ms]')
    df['end_yymmdd'] = df['end_time'].dt.strftime('%Y%m%d')
    return df

def read_rawdata(path, file_date_str):
    df = get_data(path=path)
    df['yymmdd'] = df['timestamp'].dt.strftime('%Y%m%d')
    df = df[df['yymmdd'] == file_date_str]
    if df.duplicated(subset=['timestamp', 'hostname']).any():
        print("Powertemp: The combination of 'timestamp' and 'hostname' is not unique.")
        df = df.groupby(['timestamp', 'hostname']).first().reset_index()
    else:
        print("Powertemp: (> . <)  The combination of 'timestamp' and 'hostname' is unique.")
    df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    return df

def read_failures(path):
    df = pd.read_csv(path)
    # parse timestamp
    if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        pass
    elif pd.api.types.is_string_dtype(df['timestamp']):
        unique_len_timestamp_str = df['timestamp'].str.len().unique()
        if len(unique_len_timestamp_str) > 1:
            print('[WARN]: Timestamp column contains different length of string, please check the format')
            df['timestamp'] = df['timestamp'].str[:19]
            print('[WARN]: Timestamp column has been truncated to 19 characters, in order to fit the format: YYYY-MM-DDTHH:MM:SS')
        elif len(unique_len_timestamp_str) == 1 and (unique_len_timestamp_str[0] == 19):
            pass
        elif len(unique_len_timestamp_str) == 1 and (unique_len_timestamp_str[0] == 32):
            df['timestamp'] = df['timestamp'].str[:26]
        elif len(unique_len_timestamp_str) == 1 and (unique_len_timestamp_str[0] == 26):
            pass
        else:
            raise ValueError('Timestamp column contains invalid string length')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    elif pd.api.types.is_integer_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        raise ValueError('Timestamp column must be datetime64 or string or int')

    if df['timestamp'].dt.tz is not None:
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    df['timestamp'] = df['timestamp'].astype('datetime64[ms]')
    df = df[~df['xid'].isin([-1])]
    df = df.groupby(['timestamp', 'hostname']).first().reset_index()
    df['yymmdd'] = df['timestamp'].dt.strftime('%Y%m%d')
    df['label'] = 1
    if df.duplicated(subset=['timestamp', 'hostname']).any():
        print("Failures: The combination of 'timestamp' and 'hostname' is not unique.")
        df = df.groupby(['timestamp', 'hostname']).first().reset_index()
    else:
        print("Failures: (> . <) The combination of 'timestamp' and 'hostname' is unique.")
    df.sort_values(by=['timestamp', 'hostname'], inplace=True)
    return df

def merge_data_with_schedule(df_raw, df_log, output_file_path):
    if df_raw is None or df_raw.empty:
        print(f"[WARN] Empty df_raw received for output: {output_file_path}. Skipping merge.")
        return None
    if df_log is None or df_log.empty:
        print("[WARN] Empty df_log received. Cannot merge. Skipping.")
        return None

    start_time = df_raw['timestamp'].min()
    end_time = df_raw['timestamp'].max()
    yymmdd = df_raw['yymmdd'].iloc[0]

    mask = ((df_log['begin_yymmdd'] == yymmdd) | (df_log['end_yymmdd'] == yymmdd))
    df_log_cut = df_log[mask].copy()

    df_log_cut['end_time'] = df_log_cut['end_time'].apply(lambda x: min(x, pd.to_datetime(end_time)))
    df_log_cut = slice_df_by_time_range(df_log_cut,[[start_time,end_time]],timestamp_col='begin_time')


    if df_log_cut.empty:
         print(f"[INFO] No relevant job logs found for the period of {output_file_path}. Saving raw data with zero allocation_id.")
         df_raw['job_begin_time'] = pd.NaT
         df_raw['job_end_time'] = pd.NaT
         df_raw['allocation_id'] = 0
    else:
        df_raw = df_raw.sort_values('timestamp')
        df_log_cut = df_log_cut.sort_values('begin_time')

        df_raw = pd.merge_asof(
            df_raw,
            df_log_cut[['hostname', 'begin_time', 'end_time', 'allocation_id']],
            left_on='timestamp',
            right_on='begin_time',
            by='hostname',
            direction='backward'
        )

        df_raw['allocation_id'] = np.where(
            (df_raw['timestamp'] >= df_raw['begin_time'])
            & (df_raw['timestamp'] <= df_raw['end_time']),
            df_raw['allocation_id'],
            0
        )
        df_raw['allocation_id'] = df_raw['allocation_id'].fillna(0).astype(int)


    df_raw.rename(columns={'end_time': 'job_end_time'}, inplace=True)
    df_raw.rename(columns={'begin_time': 'job_begin_time'}, inplace=True)

    output_path = Path(output_file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_raw.sort_values(by=["timestamp","hostname"], inplace=True)
    df_raw.to_parquet(output_path)
    print(f"Saved processed data to: {output_path}")
    del df_log_cut
    gc.collect()
    return df_raw

def merge_data_with_failures(df_raw, df_fail, file_date_str, output_file_path):
    old_cols = df_raw.columns.tolist()
    old_cols = [col for col in old_cols if col not in ['label', 'xid']]
    df_fail_cut = df_fail[df_fail['yymmdd'] == file_date_str].copy()
    df = pd.merge_asof(df_raw[old_cols], 
                        df_fail_cut[['timestamp', 'hostname', 'label','xid']], 
                        on='timestamp', 
                        by='hostname',
                        tolerance=pd.Timedelta("60s"),
                        direction='nearest')
    df['label'] = df['label'].fillna(0)     
    df['xid'] = df['xid'].fillna(-1)   
    df.sort_values(by=["timestamp","hostname"], inplace=True)
    df = df[old_cols+['label','xid']]
    output_path = Path(output_file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    print(f"Saved processed data to: {output_path}")

if __name__=='__main__':
    Re_label_failure = True

    raw_data_base_dir = Path('rawdata/10.13139_OLCF_1861393/powtemp_10sec_mean')
    output_base_dir = Path('data/olcf')
    joblog_path = 'rawdata/10.13139_OLCF_1970187/sanitized_pernode_jobs_full.csv'
    failures_path = 'rawdata/10.13139_OLCF_1970187/failures.csv'

    output_base_dir.mkdir(parents=True, exist_ok=True)

    print("Reading job log...")
    df_log = read_joblog(path=joblog_path)

    print("Reading failures...")
    df_fail = read_failures(path=failures_path)

    print(f"Finding raw data files in: {raw_data_base_dir}")
    raw_files = list(raw_data_base_dir.rglob('*.parquet'))
    print(f"Found {len(raw_files)} raw parquet files.")

    all_features = set()

    for i, raw_file in enumerate(raw_files):
        print(f"\nProcessing file {i+1}/{len(raw_files)}: {raw_file}")
        try:
            relative_path = raw_file.relative_to(raw_data_base_dir)
            output_file = output_base_dir / relative_path
            file_date_str = str(raw_file).split('/')[-1].split('.')[0]

            if not output_file.exists():
                print("Reading raw data...")
                df_raw = read_rawdata(path=str(raw_file), file_date_str=file_date_str)

                if df_raw is not None and not df_raw.empty:
                    if len(all_features) == 0:
                        all_features = extract_featurelist(df_raw)
                    elif len(all_features) != 0:
                        cur_features = extract_featurelist(df_raw)
                        assert all_features == cur_features, f"Feature mismatch between {raw_file} and previous files. Expected: {all_features}, Found: {cur_features}"

                    print("Merging data with schedule...")
                    merge_data_with_schedule(df_raw, df_log, output_file_path=output_file)
                else:
                    print(f"Skipping merge for {raw_file} due to read error or empty dataframe.")
            elif output_file.exists() and Re_label_failure:
                print("Reading merged data...")
                df_raw = get_data(path=str(output_file))
                df_raw = df_raw.sort_values(by=["timestamp","hostname"])
            elif output_file.exists() and not Re_label_failure:
                print(f"Output file already exists: {output_file}. Skipping.")
                continue
            else:
                raise ValueError(f"Unexpected case for {raw_file}")
            
            if Re_label_failure:
                merge_data_with_failures(df_raw, df_fail, file_date_str,output_file_path=output_file)
            del df_raw
            gc.collect()

        except Exception as e:
            print(f"[ERROR] Failed to process {raw_file}: {e}")
            gc.collect()


    feature_list_path = output_base_dir / "features.txt"
    if not feature_list_path.exists():
        print(f"\nSaving combined feature list to: {feature_list_path}")
        with open(feature_list_path, "w") as f:
            for item in sorted(list(all_features)):
                f.write("%s\n" % item)
    else:
        print(f"\nFeature list file already exists at {feature_list_path}, skipping save.")

    destination_path = Path('data/failures.csv')
    if not destination_path.exists():
        shutil.copy(failures_path, destination_path)
    else:
        print(f"Destination {destination_path} already exists, skipping copy.")

    print("\nProcessing complete.")
