#!/usr/bin/env python3
"""Precompute pixel sets (coords, weights) for every frame in local .npy
image trajectories and save to a single .npz for fast training.

Output arrays in the NPZ:
  coords    : (P, 2) float32   concatenated pixel coordinates per point
  weights   : (P,)   float32   concatenated weights per point
  frame_ptr : (F+1,) int64     prefix sum offsets into coords/weights for each frame
  traj_uid  : (F,)   int32     unique trajectory id (across all trajectories)
  t         : (F,)   int32     timestep within trajectory
  H, W      : int32 scalars
  input_dir : (1,)   U         source directory
  traj_files: (N,)   U         sorted .npy filenames

This matches your current extraction logic in train_image.py.
"""
import os
import argparse
import numpy as np
import pickle as pkl

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input_pkl', type=str, default="../DNA_supervised-main/Data/ValidData/valid_image_length_data.pkl" )
    p.add_argument('--out_npy', type=str, default="dataset/valid_average_length.npy")
    args = p.parse_args()


    with open(args.input_pkl, 'rb') as f:
        length_data = pkl.load(f)
    print(type(length_data), len(length_data))
    print(type(length_data[0]), length_data[0].shape)
    length_array = np.array(length_data) # (N, T, 1)
    length_array = length_array[:, :, 0]  # (N, T)
    length_array = length_array - length_array[:, 0:1]  # normalize by initial length, compute the change
    print(length_array.shape)
    average_length = np.mean(length_array, axis=1)  # (N,)
    print("average_length shape:", average_length.shape)
        

    os.makedirs(os.path.dirname(args.out_npy) or '.', exist_ok=True)
    with open(args.out_npy, 'wb') as f:
        np.save(f, average_length)

    

if __name__ == '__main__':
    main()
