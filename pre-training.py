import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import time
import glob
import argparse
import tensorflow as tf
from model import transformer
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument(
    "--filename", type=str, help="data directory"
)
parser.add_argument(
    "--n_days", type=int, default=10, help="data directory"
)


args = parser.parse_args()
BASE = args.filename
N_DAYS = args.n_days
activity_type = 'interactive' #pre-training 과정


def time_convert(number):
    result = time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(number))
    return result


def convert_to_series(ip, data, window):
    data = data.drop(['ip_dst'], axis=1)
    df, names = list(), list()
    for i in range(window, 0, -1):
        df.append(data.shift(i))
        names += [('%s(t-%d)' % (col, i)) for col in data.columns]
    df.append(data)
    names += [('%s(t)' % (col)) for col in data.columns]
    
    agg = pd.concat(df, axis=1)
    agg.columns = names
    ip = pd.DataFrame([ip for _ in range(len(agg))], columns=['ip_dst'])
    agg = pd.concat([ip, agg], axis=1)
    agg = agg.dropna().reset_index(drop=True)
    return agg


def create_time_series_dataset(activity, window_size=10, num_data=40):
    out_df = pd.DataFrame()
    files = glob.glob(BASE)
    files.sort()
    activities = list(set([file.split('/')[4] for file in files]))
    filtered_files = [file for file in files if os.path.basename(file).split('_')[0] == activity]

    for file in filtered_files:
        df = pd.read_csv(file)
        df.time = df.time.apply(time_convert)

        sum_df = df.groupby(['time', 'ip_dst'], as_index=False).sum()
        df = df.groupby(['time', 'ip_dst'], as_index=False).count().reset_index(drop=True)
        df = pd.concat([df[list(df.columns)[:3]], sum_df['data_len']], axis=1)
        df.columns = ['time', 'ip_dst', 'traffic', 'data_len']

        cnt_check = df.groupby(['ip_dst']).count()
        dst_ips = list(cnt_check[cnt_check.traffic > num_data].index)

        for ip in dst_ips:
            tmp_df = df[df.ip_dst == ip].reset_index(drop=True)
            tmp_df = convert_to_series(ip, tmp_df, window_size)
            out_df = pd.concat([out_df, tmp_df]).reset_index(drop=True)
            
    out_df.to_csv(f'{activity}_{window_size}wnd_{num_data}data.csv', index=False)


def load_and_preprocess_data(activity, window_size=10, num_data=40):
    data = pd.read_csv(f'{activity}_{window_size}wnd_{num_data}data.csv')
    columns_to_drop = ['ip_dst', 'time(t)'] + [f'time(t-{i})' for i in range(window_size, 0, -1)]
    columns_to_drop += [f'data_len(t-{i})' for i in range(window_size, 0, -1)] + ['data_len(t)']
    data.drop(columns_to_drop, axis=1, inplace=True)

    X = data.drop(['traffic(t)'], axis=1)
    Y = data['traffic(t)'].values
    
    return X,Y


create_time_series_dataset(activity_type, window_size=N_DAYS) 
X, Y = load_and_preprocess_data(activity_type, window_size=N_DAYS)
X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.2)

model = transformer(
    input_shape=[N_DAYS, 1],
    head_size=32,
    num_heads=3,
    ff_dim=3,
    num_transformer_blocks=3,
    mlp_units=[100],
    mlp_dropout=0.5,
    dropout=0.25,
)

model.compile(
    loss="mean_squared_error",
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
checkpoint = tf.keras.callbacks.ModelCheckpoint(f"model/interactive_pre_trained.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

history = model.fit(X_train, y_train, 
            epochs=300, 
            batch_size=128,
            validation_data=(X_valid, y_valid), 
            callbacks=[early_stop, checkpoint]
            )
