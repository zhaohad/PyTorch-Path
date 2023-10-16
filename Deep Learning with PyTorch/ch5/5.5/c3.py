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
t_un = t_u * 0.1

params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-2
optimizer = optim.SGD([params], lr=learning_rate)

t_p = model.model(t_un, *params)
loss = model.loss_fn(t_p, t_c)

optimizer.zero_grad()
loss.backward()
optimizer.step()

dump(params)

