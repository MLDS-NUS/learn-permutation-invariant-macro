# Polymer Length Prediction

This folder contains the Section `4.4. Polymer Extension` in the paper. 
## 1. Generate pixel data

```bash
cd data_generation/DNA_sim_data
python make_data.py # download the polymer simulation data from Hugging Face, and convert to video input. Need the "datasets" library.
python cmpt_x_span.py # compute the targeted polymer stretch length from video input

# compute the pixel info from images
cd ..
python precompute_pixel_sets_npz.py # chanage the input_pkl and out_npz CLI variables to generate valid / test pixel input

cd ../ # change to root dir
```

Above data preparation creates:

- `data_generation/dataset/train_pixel_sets.npz`
- `data_generation/dataset/valid_pixel_sets.npz`
- `data_generation/dataset/test_slow_pixel_sets.npz`
- `data_generation/dataset/test_medium_pixel_sets.npz`
- `data_generation/dataset/test_fast_pixel_sets.npz`

## 2. Train

```bash
python train_nflow_on_pixels.py # train encoder-decoder. Very z_dim to try different latent dimensions.
```
The checkpoint is saved in `trained_nflow_images/`. 

## 3. Generate latent data
```bash
python generate_Z_data.py --data_split train # generate the [bar z, hat z], as the input of the dynamics model. Change --data_split to generate valid and test data as well
```

## 4. Learn macro dynamics (predict stretch length)


```bash
cd learn_sde
python train_nflow_on_pixels.py # learn the macro dynamics
python plot_macro.py # simulate the dynamics on test data using the learned dynamics model, and compare the simulated trajectories to the ground-truth test trajectories
```