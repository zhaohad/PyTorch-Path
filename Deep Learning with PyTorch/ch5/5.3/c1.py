import torch
import numpy as np
import sys
from matplotlib import pyplot as plt

sys.path.append("../..")

from hwutil.utils import dump
import model

t_c = [0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)

dump(t_c)
dump(t_u)

w = torch.ones(())
b = torch.zeros(())

dump(w)
dump(b)

t_p = model.model(t_u, w, b)
dump(t_p)

loss = model.loss_fn(t_p, t_c)
dump(loss)

# wb = model.training_loop(n_epochs=100, learning_rate=1e-2, params=torch.tensor([1.0, 0.0]), t_u=t_u, t_c=t_c)
# wb = model.training_loop(n_epochs=100, learning_rate=1e-3, params=torch.tensor([1.0, 0.0]), t_u=t_u, t_c=t_c)
# wb = model.training_loop(n_epochs=1000000, learning_rate=1e-4, params=torch.tensor([1.0, 0.0]), t_u=t_u, t_c=t_c)
# w, b = 0.5358, -17.2503
# wb = model.training_loop(n_epochs=100, learning_rate=1e-4, params=torch.tensor([1.0, 0.0]), t_u=t_u, t_c=t_c)
t_un = t_u * 0.1
# wb = model.training_loop(n_epochs=500000, learning_rate=1e-4, params=torch.tensor([1.0, 0.0]), t_u=t_un, t_c=t_c)
w, b = 5.3577, -17.2479

# print(f"wb = {wb}")

print(f"wb = {w, b}")

params = (w, b)

t_p = model.model(t_un, *params)
fig = plt.figure(dpi=200)
plt.xlabel("Temperature (Fahrenheit)")
plt.ylabel("Temperature (Celsius)")
plt.plot(t_u.numpy(), t_p.detach().numpy())
plt.plot(t_u.numpy(), t_c.numpy(), 'o')
plt.show()
