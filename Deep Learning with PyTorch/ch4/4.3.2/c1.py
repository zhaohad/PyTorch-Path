import csv

import numpy as np
import torch

wine_path = "./tabular-wine/winequality-white.csv"

wineq_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=";", skiprows=1)

print(f"wineq_numpy.shape = {wineq_numpy.shape}")

col_list = next(csv.reader(open(wine_path), delimiter=";"))

print(f"col_list = {col_list}")

wineq = torch.from_numpy(wineq_numpy)

print(f"wineq.shape = {wineq.shape}, wineq.dtype = {wineq.dtype}")

data = wineq[:, :-1]  # 所有行的除最后一列的所有列

target = wineq[:, -1].long()  # 所有行的最后一列

print(f"data = {data} data.shape = {data.shape}")

data_mean = torch.mean(data, dim=0)

print(f"data_mean = {data_mean}, data_mean.shape = {data_mean.shape}")

data_var = torch.var(data, dim=0)

print(f"data_var = {data_var} data_var.shape = {data_var.shape}")

data_normalized = (data - data_mean) / torch.sqrt(data_var)

print(f"data_normalized = {data_normalized} data_normalized.shape = {data_normalized.shape}")

bad_indexes = target <= 3  # bad_indexes = torch.le(target, 3)  按行求bool

print(f"bad_indexes = {bad_indexes} bad_indexes.shape = {bad_indexes.shape}")

bad_data = data[bad_indexes]

print(f"bad_data = {bad_data} bad_data.shape = {bad_data.shape}")

mid_data = data[(target > 3) & (target < 7)]

print(f"mid_data = {mid_data} mid_data.shape = {mid_data.shape}")

good_data = data[target >= 7]

print(f"good_data = {good_data} good_data.shape = {good_data.shape}")
