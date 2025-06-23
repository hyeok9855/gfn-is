import numpy as np
import torch

from torch_geometric.data import Data, Batch


def scale_rewards(
    rewards: np.ndarray, beta: float, min_reward: float, max_reward: float
) -> np.ndarray:
    """
    Apply the inverse temperature (beta) scaling to rewards and
    scale them to [min_reward, max_reward].
    """
    assert min_reward > 0, "Minimum reward must be greater than 0"

    rewards = np.log(1 + np.exp(rewards))  # SoftPlus to ensure positivity
    scaled_rewards = rewards**beta
    _min, _max = np.min(scaled_rewards), np.max(scaled_rewards)
    # normalize to [0, 1]
    scaled_rewards = (scaled_rewards - _min) / (_max - _min)
    # scale to [min_reward, max_reward]
    scaled_rewards = scaled_rewards * (max_reward - min_reward) + min_reward
    return scaled_rewards


def tensor_to_np(tensor, reduce_singleton=True) -> np.ndarray | float:
    """Casts torch.tensor into np.array.
    Slow - requires GPU/CPU sync. Try to use infrequently."""
    if type(tensor) == np.ndarray:
        return tensor
    x = tensor.to("cpu").detach().numpy()
    if reduce_singleton and x.shape == (1,):
        return float(x)
    return x


def batch(list):
    if type(list[0]) is torch.Tensor:
        batch = torch.stack(list)
    elif type(list[0]) is Data:
        # Handle batching for torch geometric data (graphs)
        batch = Batch.from_data_list(list)
    return batch
