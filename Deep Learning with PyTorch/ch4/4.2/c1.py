import imageio
import torch

dir_path = "2-LUNG 3.0  B70f-04083"
vol_arr = imageio.volread(dir_path, "DICOM")
print(f"vol_arr.shape = {vol_arr.shape}")

vol = torch.from_numpy(vol_arr).float()
vol = torch.unsqueeze(vol, 0)

print(f"vol.shape = {vol.size()}")
