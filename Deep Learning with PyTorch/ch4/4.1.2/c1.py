import imageio
import torch

img_arr = imageio.imread('./img.png')

print(f'img_arr.shape = {img_arr.shape}')

img = torch.from_numpy(img_arr)

print(torch.__version__)

print(f'img.shape = {img.shape}')

out = img.permute(2, 0, 1)

print(f'out.shape = {out.shape}')

print(f"id((img.storage()) = {id(img.storage())}  id(out.storage()) = {id(out.storage())}")

img[0][0][0] = 253

print(f"out[0][0][0] = {out[0][0][0]} {id(img.storage()) == id(out.storage())}")
