from datasets import load_dataset

import numpy as np
import pickle
import os
import shutil
from tqdm import tqdm
import torch
#%%
x, y = np.meshgrid( np.linspace(-250, 249, 500, dtype=np.float32), 
                   np.linspace(-50, 49, 100, dtype=np.float32) )
def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

_TORCH_MESH = {}

def _get_torch_mesh(device):
    key = str(device)
    if key not in _TORCH_MESH:
        _TORCH_MESH[key] = (
            torch.from_numpy(x).to(device=device),
            torch.from_numpy(y).to(device=device),
        )
    return _TORCH_MESH[key]

def gaussian_torch(x, y, X, Y, sigma):
    return 1/(2*torch.pi*sigma**2) * torch.exp(-((x-X)**2 + (y-Y)**2)/(2*sigma**2))

def XYZtoImg_XY_batch_torch(XYZ_batch, device="cuda"):
    with torch.no_grad():
        XYZ = torch.as_tensor(XYZ_batch, dtype=torch.float32, device=device)
        x_t, y_t = _get_torch_mesh(device)
        z = XYZ[:, 2, :] - torch.mean(XYZ[:, 2, :], dim=1, keepdim=True)
        # sigmaz = torch.clamp(torch.abs(z) / 5.0, min=2.0)
        sigmaz = torch.clamp(torch.abs(z) / 5.0, min=1.0)
        visible = torch.randint(0, 10, (XYZ.shape[0], XYZ.shape[2]), device=device) > 1
        Ixy_temp = gaussian_torch(
            x_t[None, :, :, None],
            y_t[None, :, :, None],
            XYZ[:, 0, :][:, None, None, :],
            XYZ[:, 1, :][:, None, None, :],
            sigmaz[:, None, None, :],
        )
        # Ixy_temp *= visible[:, None, None, :]
        Ixy = torch.sum(Ixy_temp, dim=3)
        max_vals = torch.amax(Ixy, dim=(1, 2), keepdim=True)
        Ixy = Ixy / max_vals
        Ixy = torch.round(Ixy * 255.0).to(torch.uint8)
        return Ixy.cpu().numpy()

def Traj2Imgarray(traj, batch_size=50, device="cuda"):
    # traj is a list of M states (3N coordinates of the beads)
    # N is the number of beads
    # returns a array of images, shape (M, 100, 500)
    traj_arr = np.asarray(traj, dtype=np.float32)
    if traj_arr.ndim == 1:
        traj_arr = traj_arr[None, :]
    traj_xyz = traj_arr.reshape(traj_arr.shape[0], -1, 3).transpose(0, 2, 1)
    images = np.empty((traj_xyz.shape[0], x.shape[0], x.shape[1]), dtype=np.uint8)
    for start in range(0, traj_xyz.shape[0], batch_size):
        end = min(start + batch_size, traj_xyz.shape[0])
        images[start:end] = XYZtoImg_XY_batch_torch(traj_xyz[start:end], device=device)
    return images

def Traj2Length(traj):
    # traj is a list of M states (3N coordinates of the beads)
    # N is the number of beads
    # returns a array of lenght, shape (M)
    traj_arr = np.asarray(traj, dtype=np.float32)
    if traj_arr.ndim == 1:
        traj_arr = traj_arr[None, :]
    x_coords = traj_arr[:, ::3]
    length = np.ptp(x_coords, axis=1)
    return length.reshape(-1, 1)

def make_data_main(path='Data/TrainingData', split='train', number=-1, device_id=None):
    print('Making data for', split)
    dataset = load_dataset("MLDS-NUS/polymer-dynamics", split=split)
    ImageData=[]
    ImageLengthData=[]
    number=len(dataset['x']) if number<0 else number
    device = "cuda" if device_id is None else f"cuda:{device_id}"
    for i in tqdm(range(number)):
        # print(type(dataset['x'][i]), len(dataset['x'][i])) #<class 'list'> 1001
        # print(type(dataset['x'][i][0]))# <class 'list'>
        # print(len(dataset['x'][i][0])) # 900        
        traj_arr = np.asarray(dataset['x'][i], dtype=np.float32)
        combined_images = Traj2Imgarray(traj_arr, device=device)
        combined_length = Traj2Length(traj_arr)
        ImageData.append(combined_images)
        ImageLengthData.append(combined_length)

    
    with open(path+'/' + split +'_image_data.pkl', 'wb') as file:
        pickle.dump(ImageData, file)
    print('Image Data Done')
    with open(path+'/' + split +'_image_length_data.pkl', 'wb') as file:
        pickle.dump(ImageLengthData, file)
    print('Image length Data Done')
    return ImageData, ImageLengthData


if __name__ == '__main__':
    
    cuda_id = 3
    number = -1  # set to -1 to use all data

    mkdir('Data/TrainData')
    make_data_main(path='Data/TrainData', split='train', number=number, device_id=cuda_id)
    
    mkdir('Data/ValidData')
    make_data_main(path='Data/ValidData', split='valid', number=number, device_id=cuda_id)
    
    mkdir('Data/TestData')
    make_data_main(path='Data/TestData', split='test_fast', number=number, device_id=cuda_id)
    make_data_main(path='Data/TestData', split='test_medium', number=number, device_id=cuda_id)
    make_data_main(path='Data/TestData', split='test_slow', number=number, device_id=cuda_id)