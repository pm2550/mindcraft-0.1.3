"""
Minecraft Micro Policy — PyTorch model.
CNN + Attention + LSTM + PPO heads.

Architecture:
  Terrain CNN(21x21) → 256
  Threat Bahdanau Attention(self_state × threats) → 64
  Goal-Terrain Cross Attention(goal × terrain_feat) → 64
  Other features(temporal + baritone + task) → 64
  concat → 448 → LSTM(256) → MLP(256) → action heads + value
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TerrainCNN(nn.Module):
    """Process 21x21 terrain grid with spatial convolutions.
    Input: 3 channels — terrain(2D passability) + heightmap(ground height) + ceilmap(ceiling).
    Falls back to 1-channel if heightmap/ceilmap not available (backward compat).
    """

    def __init__(self, out_dim=256, in_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),   # 21x21 → 21x21x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),  # → 21x21x64
            nn.ReLU(),
            nn.MaxPool2d(2),                   # → 10x10x64
            nn.Conv2d(64, 64, 3, padding=1),  # → 10x10x64
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),           # → 4x4x64 = 1024
        )
        self.fc = nn.Linear(1024, out_dim)

    def forward(self, terrain):
        """
        Args:
            terrain: (batch, 441) flattened 21x21 grid, values 0-3
        Returns:
            features: (batch, out_dim)
            spatial: (batch, 64, 4, 4) for cross-attention
        """
        # terrain shape: (batch, C*441) or (batch, C, 21, 21) where C=1 or 3
        B = terrain.size(0)
        if terrain.dim() == 2:
            total = terrain.size(1)
            if total == 441:
                # Legacy 1-channel: pad to 3 channels with zeros
                x = terrain.view(B, 1, 21, 21) / 3.0
                if self.in_channels == 3:
                    x = torch.cat([x, torch.zeros(B, 2, 21, 21, device=x.device)], dim=1)
            elif total == 3 * 441:
                x = terrain.view(B, 3, 21, 21)
                # Normalize each channel: terrain/3, heightmap/12, ceilmap/8
                x[:, 0] /= 3.0
                x[:, 1] /= 12.0
                x[:, 2] /= 8.0
            else:
                x = torch.zeros(B, self.in_channels, 21, 21, device=terrain.device)
        else:
            x = terrain  # already (B, C, 21, 21)
        spatial = self.conv(x)                    # (B, 64, 4, 4)
        flat = spatial.view(spatial.size(0), -1)  # (B, 1024)
        features = self.fc(flat)                  # (B, out_dim)
        return features, spatial


class ThreatAttention(nn.Module):
    """Bahdanau additive attention: self_state attends to threat tokens."""

    def __init__(self, self_dim=12, threat_dim=4, hidden=32, out_dim=64):
        super().__init__()
        self.W_q = nn.Linear(self_dim, hidden)
        self.W_k = nn.Linear(threat_dim, hidden)
        self.v = nn.Linear(hidden, 1)
        self.out_proj = nn.Linear(threat_dim, out_dim)

    def forward(self, self_state, threats, threat_mask):
        """
        Args:
            self_state: (batch, 12)
            threats: (batch, 4, 4) — 4 threat slots × 4 features
            threat_mask: (batch, 4) — 1 if threat present, 0 if empty
        Returns:
            (batch, out_dim)
        """
        q = self.W_q(self_state).unsqueeze(1)          # (B, 1, hidden)
        k = self.W_k(threats)                           # (B, 4, hidden)
        energy = self.v(torch.tanh(q + k)).squeeze(-1)  # (B, 4)

        # Mask empty threat slots
        energy = energy.masked_fill(threat_mask == 0, -1e9)
        weights = F.softmax(energy, dim=-1)              # (B, 4)

        context = (weights.unsqueeze(-1) * threats).sum(dim=1)  # (B, 4)
        return self.out_proj(context)                            # (B, out_dim)


class GoalTerrainCrossAttention(nn.Module):
    """Cross attention: goal queries terrain spatial features."""

    def __init__(self, goal_dim=6, terrain_channels=64, out_dim=64, n_heads=4):
        super().__init__()
        self.goal_proj = nn.Linear(goal_dim, terrain_channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=terrain_channels, num_heads=n_heads, batch_first=True
        )
        self.out_proj = nn.Linear(terrain_channels, out_dim)

    def forward(self, goal, terrain_spatial):
        """
        Args:
            goal: (batch, 6)
            terrain_spatial: (batch, 64, 4, 4) from TerrainCNN
        Returns:
            (batch, out_dim)
        """
        B, C, H, W = terrain_spatial.shape
        kv = terrain_spatial.view(B, C, H * W).permute(0, 2, 1)  # (B, 16, 64)
        q = self.goal_proj(goal).unsqueeze(1)                      # (B, 1, 64)
        out, _ = self.attn(q, kv, kv)                              # (B, 1, 64)
        return self.out_proj(out.squeeze(1))                        # (B, out_dim)


class MicroPolicy(nn.Module):
    """
    Full micro policy: perceive → remember → act.

    Input obs fields (from rl_obs):
      terrain: 441 floats (21x21 grid, 0-3)
      self_state: 12 floats (hp, food, armor, onGround, sprint, vx,vy,vz, slot, cooldown, los, canDig)
      threats: 4×4 floats (dx/25, dz/25, dist/32, present)
      goal: 6 floats (anchorDist, anchorDx, anchorDz, maxRetreat, returnReq, mode)
      temporal: 5 floats (threat_count, recentDmg, timeInMode, lastDyaw, speed)
      baritone: 20 floats (pathing, process, goalDx/Dz/Dy/Dist, estTicks, progress, reserved, 9 inputs)
      task_id: int 0-5 (idle/fight/flee/survive/navigate/mine/build)

    Output:
      Continuous: move_forward, move_strafe, yaw, pitch (Gaussian μ,σ)
      Discrete: jump, sprint, attack, use, sneak (Bernoulli)
      Discrete: hotbar (Categorical 9)
      Value: V(s) scalar
    """

    # Observation field sizes
    TERRAIN_DIM = 441
    SELF_DIM = 12
    THREAT_DIM = 4 * 4  # 4 threats × 4 features
    GOAL_DIM = 6
    TEMPORAL_DIM = 5
    BARITONE_DIM = 20
    NUM_TASKS = 7  # idle, fight, flee, survive, navigate, mine, build

    # LSTM
    LSTM_DIM = 256

    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device

        # === Perception ===
        self.terrain_cnn = TerrainCNN(out_dim=256)
        self.threat_attn = ThreatAttention(self_dim=self.SELF_DIM, threat_dim=4, out_dim=64)
        self.goal_attn = GoalTerrainCrossAttention(goal_dim=self.GOAL_DIM, out_dim=64)
        self.other_mlp = nn.Sequential(
            nn.Linear(self.TEMPORAL_DIM + self.BARITONE_DIM, 64),
            nn.ReLU(),
        )
        self.task_embed = nn.Embedding(self.NUM_TASKS, 32)

        # === Memory ===
        feature_dim = 256 + 64 + 64 + 64 + 32  # terrain + threat + goal + other + task = 480
        self.lstm = nn.LSTM(feature_dim, self.LSTM_DIM, num_layers=1, batch_first=True)

        # === Actor MLP ===
        self.actor_mlp = nn.Sequential(
            nn.Linear(self.LSTM_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # === Continuous action heads (Gaussian: μ, log_σ) ===
        self.move_fwd_head = nn.Linear(256, 2)    # μ, log_σ for [-1,1]
        self.move_strafe_head = nn.Linear(256, 2)  # μ, log_σ for [-1,1]
        self.yaw_head = nn.Linear(256, 2)          # μ, log_σ for [-180,180]
        self.pitch_head = nn.Linear(256, 2)        # μ, log_σ for [-90,90]

        # === Discrete action heads (Bernoulli logits) ===
        self.jump_head = nn.Linear(256, 1)
        self.sprint_head = nn.Linear(256, 1)
        self.attack_head = nn.Linear(256, 1)
        self.use_head = nn.Linear(256, 1)
        self.sneak_head = nn.Linear(256, 1)

        # === Hotbar head (Categorical 9) ===
        self.hotbar_head = nn.Linear(256, 9)

        # === Tactical head (what mode should we be in?) ===
        self.tactical_head = nn.Linear(256, self.NUM_TASKS)

        # === Value head ===
        self.value_head = nn.Sequential(
            nn.Linear(self.LSTM_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.to(device)

    def parse_obs(self, obs_dict):
        """Convert obs dict from Node.js into tensor components."""
        terrain = torch.tensor(obs_dict.get('terrain', [0] * 441), dtype=torch.float32)
        self_state = torch.tensor(obs_dict.get('self_state', [0] * 12), dtype=torch.float32)
        threats_raw = obs_dict.get('threats', [])
        threats = torch.zeros(4, 4)
        threat_mask = torch.zeros(4)
        for i, t in enumerate(threats_raw[:4]):
            if t:
                threats[i] = torch.tensor([t.get('dx', 0)/25, t.get('dz', 0)/25,
                                           min(t.get('dist', 32), 32)/32,
                                           1.0])
                threat_mask[i] = 1.0
        goal = torch.tensor(obs_dict.get('goal', [0] * 6), dtype=torch.float32)
        temporal = torch.tensor(obs_dict.get('temporal', [0] * 5), dtype=torch.float32)
        baritone = torch.tensor(obs_dict.get('baritone_vec', [0] * 20), dtype=torch.float32)
        task_id = torch.tensor(obs_dict.get('task_id', 0), dtype=torch.long)

        return terrain, self_state, threats, threat_mask, goal, temporal, baritone, task_id

    def forward(self, terrain, self_state, threats, threat_mask, goal,
                temporal, baritone, task_id, hidden=None):
        """
        Args: all tensors with batch dim
        Returns: action_dict, value, new_hidden
        """
        B = terrain.shape[0]

        # Perception
        terrain_feat, terrain_spatial = self.terrain_cnn(terrain)       # (B,256), (B,64,4,4)
        threat_feat = self.threat_attn(self_state, threats, threat_mask) # (B, 64)
        goal_feat = self.goal_attn(goal, terrain_spatial)               # (B, 64)
        other_feat = self.other_mlp(torch.cat([temporal, baritone], -1)) # (B, 64)
        task_feat = self.task_embed(task_id)                             # (B, 32)

        # Concat all features
        features = torch.cat([terrain_feat, threat_feat, goal_feat, other_feat, task_feat], dim=-1)  # (B, 480)

        # LSTM
        if hidden is None:
            hidden = self.init_hidden(B)
        lstm_in = features.unsqueeze(1)  # (B, 1, 480) — single timestep
        lstm_out, new_hidden = self.lstm(lstm_in, hidden)
        lstm_out = lstm_out.squeeze(1)   # (B, 256)

        # Actor
        actor_feat = self.actor_mlp(lstm_out)  # (B, 256)

        # Continuous heads
        fwd_params = self.move_fwd_head(actor_feat)
        strafe_params = self.move_strafe_head(actor_feat)
        yaw_params = self.yaw_head(actor_feat)
        pitch_params = self.pitch_head(actor_feat)

        # Discrete heads
        jump_logit = self.jump_head(actor_feat)
        sprint_logit = self.sprint_head(actor_feat)
        attack_logit = self.attack_head(actor_feat)
        use_logit = self.use_head(actor_feat)
        sneak_logit = self.sneak_head(actor_feat)
        hotbar_logits = self.hotbar_head(actor_feat)

        # Tactical head
        tactical_logits = self.tactical_head(actor_feat)

        # Value
        value = self.value_head(lstm_out).squeeze(-1)

        return {
            'move_fwd': fwd_params,       # (B, 2) → μ, log_σ
            'move_strafe': strafe_params,
            'yaw': yaw_params,
            'pitch': pitch_params,
            'jump': jump_logit,            # (B, 1) → logit
            'sprint': sprint_logit,
            'attack': attack_logit,
            'use': use_logit,
            'sneak': sneak_logit,
            'hotbar': hotbar_logits,        # (B, 9)
            'tactical': tactical_logits,    # (B, 7)
            'value': value,                 # (B,)
        }, new_hidden

    def init_hidden(self, batch_size=1):
        """Initialize LSTM hidden state."""
        h = torch.zeros(1, batch_size, self.LSTM_DIM, device=self.device)
        c = torch.zeros(1, batch_size, self.LSTM_DIM, device=self.device)
        return (h, c)

    def sample_actions(self, output, deterministic=False):
        """Sample actions from policy distributions."""
        actions = {}

        # Continuous: Gaussian
        for name, scale, clamp in [
            ('move_fwd', 1.0, (-1, 1)),
            ('move_strafe', 1.0, (-1, 1)),
            ('yaw', 30.0, (-180, 180)),
            ('pitch', 15.0, (-90, 90)),
        ]:
            params = output[name]
            mu = params[:, 0] * scale
            log_sigma = params[:, 1].clamp(-5, 2)
            sigma = log_sigma.exp()
            if deterministic:
                actions[name] = mu.clamp(*clamp)
            else:
                dist = torch.distributions.Normal(mu, sigma)
                actions[name] = dist.sample().clamp(*clamp)

        # Discrete: Bernoulli
        for name in ['jump', 'sprint', 'attack', 'use', 'sneak']:
            logit = output[name].squeeze(-1)
            if deterministic:
                actions[name] = (logit > 0).float()
            else:
                actions[name] = torch.distributions.Bernoulli(logits=logit).sample()

        # Hotbar: Categorical
        if deterministic:
            actions['hotbar'] = output['hotbar'].argmax(dim=-1)
        else:
            actions['hotbar'] = torch.distributions.Categorical(logits=output['hotbar']).sample()

        # Tactical
        if deterministic:
            actions['tactical'] = output['tactical'].argmax(dim=-1)
        else:
            actions['tactical'] = torch.distributions.Categorical(logits=output['tactical']).sample()

        return actions

    def get_param_count(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == '__main__':
    # Quick test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MicroPolicy(device=device)
    print(f"Model parameters: {model.get_param_count():,}")
    print(f"Device: {device}")

    # Test forward pass
    B = 1
    terrain = torch.randn(B, 441, device=device)
    self_state = torch.randn(B, 12, device=device)
    threats = torch.randn(B, 4, 4, device=device)
    threat_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.float32, device=device)
    goal = torch.randn(B, 6, device=device)
    temporal = torch.randn(B, 5, device=device)
    baritone = torch.randn(B, 20, device=device)
    task_id = torch.tensor([1], dtype=torch.long, device=device)

    output, hidden = model(terrain, self_state, threats, threat_mask,
                           goal, temporal, baritone, task_id)
    actions = model.sample_actions(output)

    print("\nAction outputs:")
    for k, v in actions.items():
        print(f"  {k}: {v.item() if v.numel() == 1 else v.tolist()}")
    print(f"  value: {output['value'].item():.3f}")

    # Timing
    import time
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(1000):
        output, hidden = model(terrain, self_state, threats, threat_mask,
                               goal, temporal, baritone, task_id, hidden)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    print(f"\n1000 forward passes: {elapsed:.1f}ms ({elapsed/1000:.2f}ms per pass)")
