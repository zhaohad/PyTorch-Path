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
