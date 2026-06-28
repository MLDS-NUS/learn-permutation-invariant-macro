import torch

def sample_gaussian_mixture(num_samples, mean_list, std_list):
    # sample num_samples from 1D Gaussian mixture from given means and stds
    assert len(mean_list) == len(std_list), "Mean and std lists must be of same length"
    num_components = len(mean_list)

    # Convert parameters to tensors on a common device/dtype
    means = torch.as_tensor(mean_list, dtype=torch.get_default_dtype())
    stds = torch.as_tensor(std_list, dtype=torch.get_default_dtype())

    device = means.device

    # Choose component indices uniformly
    comp_indices = torch.randint(low=0, high=num_components, size=(num_samples,), device=device)

    # Gather corresponding means/stds
    chosen_means = means[comp_indices]
    chosen_stds = stds[comp_indices]

    # Sample from standard normal and scale/shift
    z = torch.randn(num_samples, device=device)
    samples = chosen_means + chosen_stds * z

    return samples


def main(train_or_test):
    if train_or_test == "train":
        num_exp = 3000
    else:
        num_exp = 300

    num_samples = 200

    all_samples = []
    mean_list = []
    std_list = []

    for _ in range(num_exp):
        # Sample means from the specified intervals
        mu1 = torch.empty(1).uniform_(-1.0, -0.5).item()
        mu2 = torch.empty(1).uniform_(-0.5, 0.0).item()
        mu3 = torch.empty(1).uniform_(0.0, 0.5).item()
        mu4 = torch.empty(1).uniform_(0.5, 1.0).item()
        mean = [mu1, mu2, mu3, mu4]

        # Sample stds uniformly in [0.1, 0.2]
        stds = torch.empty(4).uniform_(0.1, 0.2).tolist()

        samples = sample_gaussian_mixture(num_samples, mean, stds)
        all_samples.append(samples)
        mean_list.append(torch.tensor(mean))
        std_list.append(torch.tensor(stds))

    # Stack into a tensor of shape (num_exp, num_samples)
    all_samples = torch.stack(all_samples, dim=0)
    all_samples = all_samples.view(num_exp, num_samples, 1)
    mean_list = torch.stack(mean_list, dim=0)
    std_list = torch.stack(std_list, dim=0)

    # For now just print shape as a simple sanity check
    print("Generated samples shape:", all_samples.shape)
    print("Mean list shape:", mean_list.shape)
    print("Std list shape:", std_list.shape)

    # save all to a file
    torch.save({
        "samples": all_samples, # shape: (num_exp, num_samples, 1)
        "means": mean_list,     # shape: (num_exp, 4)
        "stds": std_list        # shape: (num_exp, 4)
    }, f"gaussian_mixture_data_{train_or_test}.pt")


if __name__ == "__main__":
    main("train")
    main("test")
    