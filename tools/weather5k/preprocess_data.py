import json

with open('/home/bingxing2/ailab/scxlab0056/CODE/Reconst_DS/tools/weather5k/meta_info_weather5k_hrrr_r1.json', 'r', encoding='utf-8') as f:
    filtered_data = json.load(f)
    
num_elements = len(filtered_data)
print("Total files number:", num_elements)

import random
# 获取所有文件名
filenames = list(filtered_data.keys())

# 打乱文件名顺序
random.shuffle(filenames)

# 计算每个集合的大小
total = len(filenames)
train_size = int(total * 0.8)
test_size = int(total * 0.1)

# 分割数据集
train_files = {name: filtered_data[name] for name in filenames[:train_size]}
test_files = {name: filtered_data[name] for name in filenames[train_size:train_size + test_size]}
val_files = {name: filtered_data[name] for name in filenames[train_size + test_size:]}

# 保存为 JSON 文件
with open('train_files.json', 'w', encoding='utf-8') as f:
    json.dump(train_files, f, ensure_ascii=False, indent=4)

with open('test_files.json', 'w', encoding='utf-8') as f:
    json.dump(test_files, f, ensure_ascii=False, indent=4)

with open('val_files.json', 'w', encoding='utf-8') as f:
    json.dump(val_files, f, ensure_ascii=False, indent=4)

print("Data split into train, test, and validation sets.")

import json
import numpy as np
import pandas as pd
from datetime import datetime

# Load train_files.json
with open('test_files.json', 'r', encoding='utf-8') as f:
    train_files = json.load(f)

# Define the date range
start_date = datetime(2021, 1, 1, 1)
end_date = datetime(2021, 12, 31, 23)

# Initialize list to store data
data_list = []

# Iterate over each CSV file in train_files
for filename, info in train_files.items():
    # Read the CSV file
    df = pd.read_csv('/home/bingxing2/ailab/group/ai4earth/data/weather-5k/WEATHER-5K/global_weather_stations/'+filename)
    
    # Convert DATE column to datetime
    df['DATE'] = pd.to_datetime(df['DATE'])
    
    # Filter by date range
    df_filtered = df[(df['DATE'] >= start_date) & (df['DATE'] <= end_date)]
    
    # Extract TMP and WND_RATE columns
    tmp_data = df_filtered['TMP'].values
    wnd_rate_data = df_filtered['WND_RATE'].values
    
    # Get longitude, latitude, and elevation from JSON
    longitude = info['longitude']
    latitude = info['latitude']
    elevation = info['ELEVATION']
    
    # Create arrays for longitude, latitude, and elevation
    lon_data = np.full(tmp_data.shape, longitude)
    lat_data = np.full(tmp_data.shape, latitude)
    elev_data = np.full(tmp_data.shape, elevation)
    
    # Stack all variables into a 2D array (T*V)
    combined_data = np.column_stack((tmp_data, wnd_rate_data, lon_data, lat_data, elev_data))
    
    # Append to the data list
    data_list.append(combined_data)

# Convert list to 3D numpy array (N*T*V)
result_array = np.array(data_list, dtype=np.float32)

# Save as npy file
np.save('/home/bingxing2/ailab/group/ai4earth/data/weather-5k/WEATHER-5K/hrrr/test_data_hrrr_r1.npy', result_array)
print(result_array.shape)