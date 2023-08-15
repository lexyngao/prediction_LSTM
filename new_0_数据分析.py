# -*- coding: utf-8 -*-


import pandas as pd
import chinese_calendar as cc
import os
import glob
import datetime

df = pd.read_excel('meterID257_power_history.xlsx')

# 转换成8小时
df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%dT%H:%M:%SZ')
df['time'] += pd.Timedelta(hours=8)

# 删除2022-01-01T00:00:00SZ之前的数据
# s_date = pd.to_datetime('2022-01-01T00:00:00SZ', format='%Y-%m-%dT%H:%M:%SZ') + pd.Timedelta(hours=8)
# df = df[df['time'] >= s_date]

# 先处理异常值
# Change negative values in 'value' column to 0
df.loc[df['value'] < 0, 'value'] = 0
values = df['value']
mean = values.mean()
std = values.std()
# print(mean,std)
df = df[(values > mean - 3 * std) & (values < mean + 3 * std)]

# 检查是否为15min的间隔 if not 删除
diff = df['time'].diff()
mask = diff.dt.total_seconds() / 60 != 15
mask.iloc[0] = False

if mask.any():
    idx = mask.idxmax()
    # print(df.iloc[:idx])
    df = df.iloc[idx:]

# 如果周末或中国节假日is_weekday列为false
df['is_weekday'] = df['time'].dt.dayofweek < 5
df['is_holiday'] = df['time'].apply(lambda x: cc.is_holiday(x.date()))
df.loc[df['is_holiday'] | ~df['is_weekday'], 'is_weekday'] = False

df.drop(columns=['is_holiday'], inplace=True)

# 提取出天气+时间信息
path = r'./SSLS-Weather' # 文件夹路径
all_files = glob.glob(os.path.join(path, "*.csv")) # 获取文件夹中所有csv文件

df_from_each_file = (pd.read_csv(f) for f in all_files) # 读取每个csv文件
df_weather = pd.concat(df_from_each_file, ignore_index=True) # 合并所有csv文件
# 指定字段抽出
df_weather_time = df_weather.loc[df_weather['data_address'] == 'DE_ET', ['device_time','data']]
# 排序（时间）可以省略
df_weather_time.sort_values(by='device_time', inplace=True)

# 统一命名
df_weather_time.rename(columns={'device_time': 'time'}, inplace=True)
df_weather_time.rename(columns={'data': 'weather'}, inplace=True)
df_weather_time['time'] = pd.to_datetime(df_weather_time['time'], format='%Y-%m-%d %H:%M:%S')
# merge天气数据
df_merge = pd.merge(df, df_weather_time,on='time',how='left')
# 线性插值法
df_merge['weather'] = df_merge['weather'].interpolate(method='linear')
# Convert temperature to category
# 0 1 2表示温度版
# df_merge['weather'] = df_merge['weather'].apply(lambda x: 0 if x < 10 else (1 if x < 28 else 2))
# one hot版
bins = [-100, 10, 28, 100] # define the range of each category（默认应该不会超过-100-100摄氏度）
labels = ['cold', 'warm', 'hot'] # define the name of each category
df_merge['weather'] = pd.cut(df_merge['weather'], bins=bins, labels=labels)
df_merge = pd.get_dummies(df_merge, columns=['weather'])

# 增加month这一列
df_merge['month'] = pd.to_datetime(df_merge['time'], format='%Y-%m-%dT%H:%M:%SZ').dt.month
#  把时间更改为只保留小时
# df_merge['time'] = pd.to_datetime(df_merge['time'], format='%Y-%m-%dT%H:%M:%SZ').dt.strftime('%H:%M')
df_merge['time'] = pd.to_datetime(df_merge['time'], format='%Y-%m-%dT%H:%M:%SZ').dt.hour + pd.to_datetime(df_merge['time'], format='%Y-%m-%dT%H:%M:%SZ').dt.minute / 60
# 尝试平方
# df_merge['time'] = df_merge['time'].apply(lambda x: x**2)

# df_merge.to_csv('prepared_data_raw.csv', index=False)


# 导出csv
df_merge.to_csv('prepared_data_0.csv', index=False)

