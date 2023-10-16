import sys
import torch
import torch.optim as optim
import model

sys.path.append("../..")

from hwutil.utils import dump

diroptim = dir(optim)

print(diroptim)

t_c = [0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)
# t_un = t_u * 0.1

n_samples = t_u.shape[0]
n_val = int(0.2 * n_samples)

shuffled_indices = torch.randperm(n_samples)
# train_indices = shuffled_indices[:-n_val]
# val_indices = shuffled_indices[-n_val:]

train_indices = torch.tensor([9, 6, 5, 8, 4, 7, 0, 1, 3])
val_indices = torch.tensor([2, 10])

dump(train_indices)
dump(val_indices)

train_t_u = t_u[train_indices]
train_t_c = t_c[train_indices]

val_t_u = t_u[val_indices]
val_t_c = t_c[val_indices]

train_t_un = 0.1 * train_t_u
val_t_un = 0.1 * val_t_u

params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-2
optimizer = optim.SGD([params], lr=learning_rate)

res = model.train_loop_train_and_val(n_ephochs=3000, optimizer=optimizer, params=params, train_t_u=train_t_un, val_t_u=val_t_un, train_t_c=train_t_c, val_t_c=val_t_c)

dump(res)
