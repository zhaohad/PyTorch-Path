import sys

import torch

sys.path.append("../..")

from hwutil.utils import dump
import model

t_c = [0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)

dump(t_c)
dump(t_u)

params = torch.tensor([1.0, 0.0], requires_grad=True)

print(f"params.grad = {params.grad}")

# loss = model.loss_fn(model.model(t_u, * params), t_c)
loss = model.loss_fn(model.model(t_u, params[0], params[1]), t_c)
loss.backward()

print(f"params.grad = {params.grad} t_u.grad = {t_u.grad}")

t_un = t_u * 0.1
wb = model.training_loop_auto_grad(n_epochs=5000, learning_rate=1e-2, params=torch.tensor([1.0, 0.0], requires_grad=True), t_u=t_un, t_c=t_c)

dump(wb)
