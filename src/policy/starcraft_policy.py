import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class StarcraftPolicy(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

        # 计算参数量并作为属性
        self.param_dim = sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_flat_params(self) -> np.ndarray:
        return torch.cat([p.data.view(-1) for p in self.parameters()]).cpu().numpy()

    def load_flat_params(self, flat_params: np.ndarray):
        idx = 0
        for p in self.parameters():
            size = p.numel()
            p.data = torch.from_numpy(flat_params[idx:idx+size]).reshape(p.shape).float()
            idx += size

    def act(self, obs: np.ndarray, avail_actions: list = None) -> int:
        with torch.no_grad():
            logits = self.forward(torch.from_numpy(obs).float().unsqueeze(0))
            if avail_actions is not None:
                mask = torch.full_like(logits, -1e9)
                mask[0, avail_actions] = 0
                logits = logits + mask
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
        return action