#!/usr/bin/env python3
"""Train the conditional normalizing flow from a precomputed .npz cache."""
import os
import logging
import argparse
from typing import Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from models_image import AnchorGaussianMixtureND, SetEncoderND
from nflows import transforms as nf_transforms
from nflows import distributions as nf_distributions
from nflows import flows as nf_flows

def compute_feature_minmax(
    coords_px: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    coords_px = np.asarray(coords_px, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    x = coords_px[:, 0]
    y = coords_px[:, 1]
    feature_min = np.array([x.min(), y.min(), weights.min()], dtype=np.float32)
    feature_max = np.array([x.max(), y.max(), weights.max()], dtype=np.float32)
    return feature_min, feature_max

class NPZPixelFramesDataset(Dataset):
    def __init__(
        self,
        npz_path: str,
        frame_indices: Optional[np.ndarray] = None,
        max_t: Optional[float] = None,
        min_t: Optional[float] = None,
        equilibrium_timestep: Optional[np.ndarray] = None,
    ):
        npz = np.load(npz_path)
        self.coords = np.array(npz['coords'], copy=False)
        self.weights = np.array(npz['weights'], copy=False)
        self.frame_ptr = np.array(npz['frame_ptr'], copy=False)
        self.traj_uid = np.array(npz['traj_uid'], copy=False)
        self.t = np.array(npz['t'], copy=False)
        self.H = int(npz['H'])
        self.W = int(npz['W'])
        self._F = self.traj_uid.shape[0]

        if frame_indices is None:
            frame_indices = np.arange(self._F, dtype=np.int64)
        else:
            frame_indices = np.asarray(frame_indices, dtype=np.int64)

        if max_t is not None or min_t is not None or equilibrium_timestep is not None:
            valid = np.ones_like(self.t, dtype=bool)
            if max_t is not None:
                valid &= self.t <= max_t
            if min_t is not None:
                valid &= self.t >= min_t
            if equilibrium_timestep is not None:
                equilibrium_timestep = np.asarray(equilibrium_timestep)
                if equilibrium_timestep.ndim != 1:
                    raise ValueError('equilibrium_timestep must be a 1D array')
                num_traj = int(self.traj_uid.max()) + 1
                if equilibrium_timestep.shape[0] != num_traj:
                    raise ValueError(
                        f'equilibrium_timestep length {equilibrium_timestep.shape[0]} '
                        f'does not match num_traj {num_traj}'
                    )
                traj_uid = self.traj_uid.astype(np.int64, copy=False)
                eq_for_frame = equilibrium_timestep[traj_uid]
                # Use timesteps in [0, equilibrium_timestep) for each trajectory.
                valid &= self.t < eq_for_frame
            frame_indices = frame_indices[valid[frame_indices]]

        self.frame_indices = frame_indices

    def __len__(self):
        return int(self.frame_indices.shape[0])

    def __getitem__(self, i: int):
        f = int(self.frame_indices[i])
        a = int(self.frame_ptr[f])
        b = int(self.frame_ptr[f + 1])
        coords = self.coords[a:b]
        w = self.weights[a:b]
        return {
            'coords_px': torch.from_numpy(coords).float(),
            'weights': torch.from_numpy(w).float(),
        }

def pad_collate_pixel_sets(batch):
    sizes = [b['coords_px'].shape[0] for b in batch]
    Mmax = max(sizes)
    E = len(batch)

    coords_px = torch.zeros((E, Mmax, 2), dtype=torch.float32)
    weights = torch.zeros((E, Mmax), dtype=torch.float32)
    mask = torch.zeros((E, Mmax), dtype=torch.bool)

    for i, b in enumerate(batch):
        m = b['coords_px'].shape[0]
        coords_px[i, :m] = b['coords_px']
        weights[i, :m] = b['weights']
        mask[i, :m] = True

    return coords_px, weights, mask

class NFlowsConditionalARQSPixels(torch.nn.Module):
    def __init__(self, dim, z_dim, hidden_dim, n_layers, K, B, set_encoder):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.K = K
        self.B = B
        self.set_encoder = set_encoder

        transforms = []
        for _ in range(n_layers):
            transforms.append(
                nf_transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=dim,
                    hidden_features=hidden_dim,
                    context_features=z_dim,
                    num_bins=K,
                    tails='linear',
                    tail_bound=B,
                )
            )
            transforms.append(nf_transforms.RandomPermutation(features=dim))
        transform = nf_transforms.CompositeTransform(transforms)
        base_dist = nf_distributions.StandardNormal(shape=[dim])
        self.flow = nf_flows.Flow(transform=transform, distribution=base_dist)

    def log_prob(self, x, features, mask=None):
        E, B, D = x.shape
        z_ctx = self.set_encoder(features, mask=mask)
        x_flat = x.reshape(E * B, D)
        context_flat = z_ctx.unsqueeze(1).expand(E, B, self.z_dim).reshape(E * B, self.z_dim)
        log_p_flat = self.flow.log_prob(x_flat, context=context_flat)
        return log_p_flat.reshape(E, B)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train_npz', type=str, default='data_generation/dataset/train_pixel_sets.npz')
    p.add_argument('--val_npz', type=str, default='data_generation/dataset/valid_pixel_sets.npz')
    p.add_argument('--max_t', type=int, default=300)
    p.add_argument('--min_t', type=int, default=0) 
    p.add_argument(
        '--equilibrium_timestep_dir',
        type=str,
        default=None,
        help=(
            'If set, load train_equilibrium_timestep.npy and '
            'valid_equilibrium_timestep.npy from this directory.'
        ),
    )

    p.add_argument('--bs', type=int, default=512)
    p.add_argument('--samples_per_experiment', type=int, default=1000)
    p.add_argument('--val_samples_per_experiment', type=int, default=1000)

    p.add_argument('--n_layers', type=int, default=8)
    p.add_argument('--rqs_K', type=int, default=32)
    p.add_argument('--rqs_B', type=float, default=5.0)
    p.add_argument('--epsilon', type=float, default=0.005)
    p.add_argument('--z_dim', type=int, default=2)
    p.add_argument('--hidden_dim', type=int, default=128)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--num_epochs', type=int, default=200)

    p.add_argument('--device', type=int, default=0)
    p.add_argument('--save_dir', type=str, default='trained_nflow_images')


    
    args = p.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device >= 0 else 'cpu')

    # Compute feature min/max from the training cache
    npz = np.load(args.train_npz)
    H = int(npz['H']); W = int(npz['W'])
    coords_all = np.array(npz['coords'], copy=False)
    weights_all = np.array(npz['weights'], copy=False)
    feature_min, feature_max = compute_feature_minmax(coords_all, weights_all)
    del npz, coords_all, weights_all

    train_eq = None
    val_eq = None
    if args.equilibrium_timestep_dir:
        train_eq_path = os.path.join(args.equilibrium_timestep_dir, 'train_equilibrium_timestep.npy')
        val_eq_path = os.path.join(args.equilibrium_timestep_dir, 'valid_equilibrium_timestep.npy')
        train_eq = np.load(train_eq_path)
        val_eq = np.load(val_eq_path)

    max_t = args.max_t
    min_t = args.min_t
    if args.equilibrium_timestep_dir:
        max_t = None
        min_t = None

    ds_train = NPZPixelFramesDataset(
        args.train_npz,
        max_t=max_t,
        min_t=min_t,
        equilibrium_timestep=train_eq,
    )
    ds_val = NPZPixelFramesDataset(
        args.val_npz,
        max_t=max_t,
        min_t=min_t,
        equilibrium_timestep=val_eq,
    )
    if ds_val.H != ds_train.H or ds_val.W != ds_train.W:
        raise ValueError(
            f"Train/val H,W mismatch: train=({ds_train.H},{ds_train.W}) "
            f"val=({ds_val.H},{ds_val.W})"
        )

    train_loader = DataLoader(
        ds_train, batch_size=args.bs, shuffle=True, drop_last=False,
        collate_fn=pad_collate_pixel_sets,
        pin_memory=True,
    )
    val_loader = DataLoader(
        ds_val, batch_size=args.bs, shuffle=False, drop_last=False,
        collate_fn=pad_collate_pixel_sets,
        pin_memory=True,
    )

    # Logging / save
    save_dir = os.path.join(
        args.save_dir,
        f'Z{args.z_dim}',
    )
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'best_model.pth')
    log_path = os.path.join(save_dir, 'train.log')

    logger = logging.getLogger(f'TrainFlowNPZ:{log_path}')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fh = logging.FileHandler(log_path, mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)

    logger.info(
        f'Train cache={args.train_npz} Val cache={args.val_npz} H={H} W={W} '
        f'epsilon={args.epsilon}'
    )
    logger.info(
        f'min_t={min_t} max_t={max_t} equilibrium_timestep_dir={args.equilibrium_timestep_dir}'
    )
    train_used = len(ds_train)
    val_used = len(ds_val)
    train_total = ds_train._F
    val_total = ds_val._F
    train_pct = 100.0 * train_used / max(1, train_total)
    val_pct = 100.0 * val_used / max(1, val_total)
    usage_msg = (
        f'Train frames used: {train_used}/{train_total} ({train_pct:.2f}%) '
        f'Val frames used: {val_used}/{val_total} ({val_pct:.2f}%)'
    )
    print(usage_msg)
    logger.info(usage_msg)
    logger.info(f'Train frames={len(ds_train)} Val frames={len(ds_val)}')
    logger.info(f'Feature min={feature_min.tolist()} max={feature_max.tolist()}')

    feature_min_t = torch.as_tensor(feature_min, dtype=torch.float32, device=device)
    feature_max_t = torch.as_tensor(feature_max, dtype=torch.float32, device=device)
    feature_range_t = (feature_max_t - feature_min_t).clamp_min(1e-6)

    dim = 3
    set_encoder = SetEncoderND(
        in_dim=dim,
        hidden_dim=args.hidden_dim,
        z_dim=args.z_dim,
        pool='mean',
    ).to(device)
    model = NFlowsConditionalARQSPixels(
        dim=dim, z_dim=args.z_dim, hidden_dim=args.hidden_dim,
        n_layers=args.n_layers, K=args.rqs_K, B=args.rqs_B,
        set_encoder=set_encoder
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val = float('inf')
    train_hist, val_hist, epochs = [], [], []

    for epoch in tqdm(range(1, args.num_epochs + 1)):
        model.train()
        train_losses = []
        for coords_px, weights, mask in train_loader:
            coords_px = coords_px.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            features = torch.cat([coords_px, weights.unsqueeze(-1)], dim=-1)
            features = (features - feature_min_t) / feature_range_t * 2.0 - 1.0
            target = AnchorGaussianMixtureND(
                anchors=features,
                epsilon=args.epsilon,
                mask=mask,
            )
            x = target.sample(args.samples_per_experiment)

            log_p = model.log_prob(x, features, mask=mask)
            loss = -log_p.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        mean_train = float(np.mean(train_losses))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for coords_px, weights, mask in val_loader:
                coords_px = coords_px.to(device, non_blocking=True)
                weights = weights.to(device, non_blocking=True)
                # print(weights); exit() # weights in [0, 255]
                mask = mask.to(device, non_blocking=True)

                features = torch.cat([coords_px, weights.unsqueeze(-1)], dim=-1)
                features = (features - feature_min_t) / feature_range_t * 2.0 - 1.0
                target = AnchorGaussianMixtureND(
                    anchors=features,
                    epsilon=args.epsilon,
                    mask=mask,
                )
                x = target.sample(args.val_samples_per_experiment)

                log_p = model.log_prob(x, features, mask=mask)
                val_losses.append((-log_p.mean()).item())

        mean_val = float(np.mean(val_losses))
        logger.info(f'Epoch {epoch:4d} | Train NLL: {mean_train:8.4f} | Val NLL: {mean_val:8.4f}')

        train_hist.append(mean_train)
        val_hist.append(mean_val)
        epochs.append(epoch)

        if mean_val < best_val:
            best_val = mean_val
            ckpt = {
                'model_state_dict': model.state_dict(),
                'H': H,
                'W': W,
                'model_hparams': {
                    'dim': dim,
                    'z_dim': args.z_dim,
                    'hidden_dim': args.hidden_dim,
                    'n_layers': args.n_layers,
                    'rqs_K': args.rqs_K,
                    'rqs_B': args.rqs_B,
                    'feature_dim': dim,
                    'deepset_pool': 'mean',
                    'epsilon': args.epsilon,
                },
                'normalization_info': {
                    'feature_min': feature_min.tolist(),
                    'feature_max': feature_max.tolist(),
                },
            }
            torch.save(ckpt, save_path)
            logger.info(f'Saved checkpoint to {save_path}')

        if epoch % 5 == 0:
            plt.figure()
            plt.plot(epochs, train_hist, label='Train NLL')
            plt.plot(epochs, val_hist, label='Val NLL')
            plt.xlabel('Epoch')
            plt.ylabel('NLL')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
            plt.close()

if __name__ == '__main__':
    main()
