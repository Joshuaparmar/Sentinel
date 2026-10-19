import sys
import os
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

    df = df[~df['xid'].isin([-1])]
    df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    df['timestamp'] = df['timestamp'].dt.round('S')
    df['timestamp'] = df['timestamp'].astype('datetime64[ms]')
    df['yymmdd'] = df['timestamp'].dt.strftime('%Y%m%d')
    df['label'] = 1
    if df.duplicated(subset=['timestamp', 'hostname']).any():
        print("Failures: The combination of 'timestamp' and 'hostname' is not unique. We would choose one of xid as the final anomaly event. Please notice that we would lose some information.")
        df = df.groupby(['timestamp', 'hostname']).first().reset_index()
    else:
        print("Great: (> . <) The combination of 'timestamp' and 'hostname' is unique.")
    df.sort_values(by=['timestamp', 'hostname'], inplace=True)
    return df


def merge_data_with_failures(df_raw, df_fail, tol, output_file_path):
    old_cols = df_raw.columns.tolist()
    df_fail_cut = df_fail.copy()
    old_cols = [col for col in old_cols if col not in ['label', 'xid']]
    if 'yymmdd' in df_raw.columns and df_raw['yymmdd'].nunique() == 1:
        file_date_str = df_raw['yymmdd'].unique()[0]
        df_fail_cut = df_fail_cut[df_fail_cut['yymmdd'] == file_date_str]

    # ---------------------------------------------------------------------
    # When no temporal tolerance is requested (tol == '0') we just want to
    # attach the failure label/xid at the host level. A standard left merge
    # is sufficient here and avoids the dtype limitations of `merge_asof`,
    # which only supports numeric or datetime keys.
    # ---------------------------------------------------------------------
    if tol == '0':
        df_fail_cut = (
            df_fail_cut.groupby('hostname').agg({
                'label': 'max',
                'xid': 'first',
            }).reset_index()
        )

        df = pd.merge(
            df_raw[old_cols],
            df_fail_cut[['hostname', 'label', 'xid']],
            on='hostname',
            how='left',
        )

    # ---------------------------------------------------------------------
    # Otherwise we perform an as-of merge on the timestamp with the given
    # tolerance. We *must* ensure that the timestamp columns on both data
    # frames are truly datetime64 dtype; otherwise `merge_asof` will error
    # out with an incompatible dtype complaint.
    # ---------------------------------------------------------------------
    else:
        df = pd.merge_asof(
            df_raw[old_cols],
            df_fail_cut[['timestamp', 'hostname', 'label', 'xid']],
            on='timestamp',
            by='hostname',
            tolerance=pd.Timedelta(tol),
            direction='forward',
        )
    df['label'] = df['label'].fillna(0)     
    df['xid'] = df['xid'].fillna(-1)   
    df.sort_values(by=["timestamp","hostname"], inplace=True)
    df = df[old_cols+['label','xid']]
    output_path = Path(output_file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    print(f"Saved processed data to: {output_path}")

def relabel_test_anomaly(tol,xid):
    failures_path = 'data/failures.csv'
    df_fail = read_failures(path=failures_path)
    file_path = 'data/olcf_task/test_anomaly_'+xid+'.parquet'
    df = get_data(path=str(file_path))
    df = df.sort_values(by=["timestamp","hostname"])
    output_path = f'data/olcf_task/test_anomaly_{xid}_{tol}.parquet'
    merge_data_with_failures(df, df_fail, tol=tol,output_file_path=output_path)

if __name__=='__main__':
    relabel_test_anomaly(tol='60min')