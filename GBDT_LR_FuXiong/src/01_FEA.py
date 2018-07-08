import pandas as pd
import numpy as np

date_parser = lambda x: pd.datetime.strptime(x, '%y%m%d%H')
data_types = {
    'id': np.uint64,
    'click': np.int64,
    'hour': np.str,
    'C1': np.uint16,
    'banner_pos': np.uint16,
    'site_id': np.object,
    'site_domain': np.object,
    'site_category': np.object,
    'app_id': np.object,
    'app_domain': np.object,
    'app_category': np.object,
    'device_id': np.object,
    'device_ip': np.object,
    'device_model': np.object,
    'device_type': np.uint16,
    'device_conn_type': np.uint16,
    'C14': np.uint16,
    'C15': np.uint16,
    'C16': np.uint16,
    'C17': np.uint16,
    'C18': np.uint16,
    'C19': np.uint16,
    'C20': np.uint16,
    'C21': np.uint16    
}

def load_train_data(f):
    train_df = pd.read_csv(f,dtype=data_types,parse_dates=['hour'],date_parser=date_parser)
    return train_df

def derive_time_features(df, start_hour=None, remove_original_feature=False):
    if start_hour is None:
        start_hour = df['hour'][0]
        
    df['hour_int'] = train_df['hour'].apply(lambda x: np.floor((x - start_hour) / np.timedelta64(1, 'h')).astype(np.uint16))
    df['day_week'] = train_df['hour'].apply(lambda x: x.dayofweek)
    df['hour_day'] = train_df['hour'].apply(lambda x: x.hour)
    
    if remove_original_feature:
        df.drop('hour', axis=1, inplace=True)
    
    
    return df, start_hour
    


f = "~/train_sample50w_count.csv"
train_df = load_train_data(f)

train_df, _ = derive_time_features(train_df)

train_df.iloc[:, :12].head()

train_df.iloc[:, 12:].head()
train_df.to_csv('~/train_sample50w_count_day.csv', index=False)
print("complete。。。")
