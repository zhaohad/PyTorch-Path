import sys
import torch.nn as nn
import torch.optim as optim
import model
from collections import OrderedDict
from matplotlib import pyplot as plt

sys.path.append("../..")

from hwutil.utils import dump

import torch.nn.modules.module

t_c = [0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c).unsqueeze(1)
t_u = torch.tensor(t_u).unsqueeze(1)

dump(t_c)
dump(t_u)

seq_model = nn.Sequential(OrderedDict([
    ('hidden_linear', nn.Linear(1, 8)),
    ('hidden_activation', nn.Tanh()),
    ('output_linear', nn.Linear(8, 1))
]))
optimizer = optim.SGD(seq_model.parameters(), lr=1e-3)

n_samples = t_u.shape[0]
n_val = int(0.2 * n_samples)

# shuffled_indices = torch.randperm(n_samples)
# train_indices = shuffled_indices[:-n_val]
# val_indices = shuffled_indices[-n_val:]

train_indices = torch.tensor([5, 6, 1, 4, 9, 0, 3, 2, 8])
val_indices = torch.tensor([7, 10])

dump(train_indices)
dump(val_indices)

t_u_train = t_u[train_indices]
t_c_train = t_c[train_indices]

t_u_val = t_u[val_indices]
t_c_val = t_c[val_indices]

t_un_train = 0.1 * t_u_train
t_un_val = 0.1 * t_u_val

model.training_loop_linear_model(n_epochs=5000, optimizer=optimizer, model=seq_model, loss_fn=nn.MSELoss(), t_u_train=t_un_train, t_u_val=t_un_val, t_c_train=t_c_train, t_c_val=t_c_val)

print('output', seq_model(t_un_val))
print('answer', t_c_val)
print('hidden', seq_model.hidden_linear.weight.grad)

t_range = torch.arange(20., 90.).unsqueeze(1)

fig = plt.figure(dpi=200)
plt.xlabel("Fahrenheit")
plt.ylabel("Celsius")
plt.plot(t_u.numpy(), t_c.numpy(), 'o')
plt.plot(t_range.numpy(), seq_model(0.1 * t_range).detach().numpy(), 'c-')
plt.plot(t_u.numpy(), seq_model(0.1 * t_u).detach().numpy(), 'kx')
plt.show()
