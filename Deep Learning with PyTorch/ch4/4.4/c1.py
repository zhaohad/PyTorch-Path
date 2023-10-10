import torch
import numpy as np

path = "./bike-sharing-dataset/hour-fixed.csv"
bikes_numpy = np.loadtxt(path, dtype=np.float32, delimiter=",", skiprows=1, converters={1: lambda x: float(x[8:10])})
bikes = torch.from_numpy(bikes_numpy)

print(f"bikes = {bikes} bikes.shape = {bikes.shape}")

daily_bikes = bikes.view(-1, 24, bikes.shape[1])

print(f"daily_bikes = {bikes} daily_bikes.shape = {daily_bikes.shape}")

daily_bikes.transpose_(1, 2)

print(f"daily_bikes = {daily_bikes} daily_bikes.shape = {daily_bikes.shape}")

first_day = bikes[:24].long()
weather_onehot = torch.zeros(first_day.shape[0], 4)
print(f"first_day = {first_day} first_day.shape = {first_day.shape}")
print(f"first_day[:, 9] = {first_day[:, 9]} first_day[:, 9].shape = {first_day[:, 9].shape} first_day[:, 9].unsqueeze(1).shape = {first_day[:, 9].unsqueeze(1).shape}")

weather_onehot.scatter_(dim=1, index=first_day[:, 9].unsqueeze(1).long() - 1, value=1.0)

print(f"weather_onehot = {weather_onehot}")

after_cat = torch.cat((bikes[:24], weather_onehot), 1)

print(f"after_cat = {after_cat} after_cat.shape = {after_cat.shape}")

daily_weather_onehot = torch.zeros(daily_bikes.shape[0], 4, daily_bikes.shape[2])

print(f"daily_weather_onehot = {daily_weather_onehot} daily_weather_onehot.shape = {daily_weather_onehot.shape}")

daily_weather_onehot.scatter_(1, daily_bikes[:, 9, :].long().unsqueeze(1) - 1, 1.0)

print(f"daily_weather_onehot = {daily_weather_onehot} daily_weather_onehot.shape = {daily_weather_onehot.shape}")

after_cat = torch.cat((daily_bikes, daily_weather_onehot), dim=1)

print(f"after_cat = {after_cat} after_cat.shape = {after_cat.shape}")
