import torch


def compute_mmd_gpu(pred_tra, true_tra, sigmas=[0.01, 0.05, 0.25, 1.25]):
    """
    GPU-accelerated MMD computation across all time steps using PyTorch.
    pred_tra, true_tra: [N, T, D] tensors (already on GPU)
    Returns: mmd_per_timestep [T] as numpy array
    """
    

    # pred_tra: [N, T, D], true_tra: [N, T, D]
    N, T, D = pred_tra.shape
    
    # Compute squared norms: [N, T]
    pred_sq = (pred_tra ** 2).sum(dim=2)
    true_sq = (true_tra ** 2).sum(dim=2)
    
    # Compute pairwise dot products using einsum: [N, N, T]
    xx = torch.einsum('itd,jtd->ijt', pred_tra, pred_tra)
    yy = torch.einsum('itd,jtd->ijt', true_tra, true_tra)
    xy = torch.einsum('itd,jtd->ijt', pred_tra, true_tra)
    
    # Compute squared distances: [N, N, T]
    dxx = pred_sq[:, None, :] + pred_sq[None, :, :] - 2 * xx
    dyy = true_sq[:, None, :] + true_sq[None, :, :] - 2 * yy
    dxy = pred_sq[:, None, :] + true_sq[None, :, :] - 2 * xy
    
    # Compute MMD with multi-scale RBF kernels
    mmd_per_timestep = torch.zeros(T, device=pred_tra.device)
    for sigma in sigmas:
        factor = -1.0 / (2 * sigma ** 2)
        XX = torch.exp(dxx * factor)
        YY = torch.exp(dyy * factor)
        XY = torch.exp(dxy * factor)
        
        # Mean over sample pairs for each timestep
        mmd_per_timestep += XX.mean(dim=(0, 1)) + YY.mean(dim=(0, 1)) - 2 * XY.mean(dim=(0, 1))
    
    mmd_per_timestep /= len(sigmas)
    return mmd_per_timestep.cpu().numpy()
