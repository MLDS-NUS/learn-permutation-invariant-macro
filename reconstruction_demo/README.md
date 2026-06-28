# Reconstruction Demo

This folder contains the demonstration to reconstruct the target density in one-dimension, with varying kernel scale $\epsilon$ and the learned latent variable dimension $\hat{z}_{dim}$. It corresponds to Section `4.1. Visualizing Distributional Reconstruction` in the paper. 
## 1. Generate data

From `reconstruction_demo/`:

```bash
cd data_generation
python generate_data.py
cd ..
```

This creates:

- `data_generation/gaussian_mixture_data_train.pt`
- `data_generation/gaussian_mixture_data_test.pt`

## 2. Train

Train one model with a chosen kernel scale and latent dimension, e.g., $\epsilon=0.1$, $\hat{z}_{dim} = 4$:

```bash
python train.py --epsilon 0.1 --z_dim 4
```

The checkpoint is saved to `results_0.1/best_conditional_flow_Z4.pt`. 



## 3. Test

Run:

```bash
python test.py --epsilon 0.1 --z_dim 4
```

This loads checkpoints from `results_0.1/`, evaluates the test set, and saves reconstruction plot.

You can also test other test samples, e.g., 
```bash
python test.py --epsilon 0.1 --z_dim 4 --exp_ids 20 
```
