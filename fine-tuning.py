import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import time
import glob
import argparse
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib import pyplot as plt


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
activity_type = ['bulk', 'video', 'web'] #fine-tuning 과정


def scoring(y_true, y_pred):
    r2 = round(metrics.r2_score(y_true, y_pred), 2)
    mae = round(metrics.mean_absolute_error(y_true, y_pred), 2)
    corr = round(np.corrcoef(y_true, y_pred)[0, 1], 2)
    mape = round(np.mean(np.abs((y_true - y_pred)/y_true))*100, 2)
    rmse = round(metrics.mean_squared_error(y_true, y_pred, squared=False), 2)

    df = pd.DataFrame({
        'R2': r2,
        "Corr": corr,
        "RMSE": rmse,
        "MAPE": mape,
        "MAE": mae
    }, index=[0])

    print(df)

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


for activity in activity_type:
    print(f"Fine-tuning for activity: {activity}")
    
    create_time_series_dataset(activity, window_size=N_DAYS)
    X, Y = load_and_preprocess_data(activity, window_size=N_DAYS)

    # train: val: test = 7:1:2
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=3407)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=3407)

    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    model = tf.keras.models.load_model('model/interactive_pre_trained.h5')

    for i in range(len(model.layers) - 4):
        model.layers[i].trainable = True

    model.compile(
        loss="mean_squared_error",
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
    )

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(f"model/{activity}_fine_tuned.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    finetune_history = model.fit(X_train.values.reshape(-1, X_train.shape[1], 1), y_train,
                                   epochs=300,
                                   batch_size=128,
                                   validation_data=(X_val.values.reshape(-1, X_train.shape[1], 1), y_val),
                                   callbacks=[early_stop, checkpoint])

    pred = model.predict(X_test.values.reshape(-1, X_test.shape[1], 1))

    pred_ = pd.DataFrame({'y_true': y_test, 'y_pred': pred.reshape(-1,)}).set_index(X_test.index).sort_index()

    scoring(pred_.y_true, pred_.y_pred)

    pred_plot = pred_.loc[:300]
    plt.plot(pred_plot.y_true, color='orange')
    plt.plot(pred_plot.y_pred, color='skyblue')    
    plt.savefig(activity+'.png')
    print(f"Fine-tuning for activity {activity} completed.\n")
