"""
Minimal training stack for MicroPolicy.

Supports:
  - Supervised tactical bootstrap from survival_episodes.jsonl
  - Supervised imitation from skill_demos.jsonl (when raw key states exist)
  - PPO-style rollout updates for future online training
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from model import MicroPolicy
from obs_codec import (
    TACTICAL_MAP,
    TASK_MAP,
    extract_action_labels_from_obs,
    parse_obs_to_tensors,
    survival_transition_to_policy_obs,
)


def _cat_tensors(items):
    return [torch.cat(xs, dim=0) for xs in zip(*items)]


def _policy_stats(output):
    cont_specs = {
        'move_fwd': (1.0, (-1.0, 1.0)),
        'move_strafe': (1.0, (-1.0, 1.0)),
        'yaw': (30.0, (-180.0, 180.0)),
        'pitch': (15.0, (-90.0, 90.0)),
    }
    dists = {}
    for name, (scale, _) in cont_specs.items():
        params = output[name]
        mu = params[:, 0] * scale
        log_sigma = params[:, 1].clamp(-5, 2)
        sigma = log_sigma.exp()
        dists[name] = torch.distributions.Normal(mu, sigma)
    for name in ['jump', 'sprint', 'attack', 'use', 'sneak']:
        dists[name] = torch.distributions.Bernoulli(logits=output[name].squeeze(-1))
    dists['hotbar'] = torch.distributions.Categorical(logits=output['hotbar'])
    dists['tactical'] = torch.distributions.Categorical(logits=output['tactical'])
    return dists


def mixed_logprob(output, actions: Dict[str, torch.Tensor], include_tactical=False):
    dists = _policy_stats(output)
    logp = 0.0
    for name in ['move_fwd', 'move_strafe', 'yaw', 'pitch']:
        if name in actions:
            logp = logp + dists[name].log_prob(actions[name])
    for name in ['jump', 'sprint', 'attack', 'use', 'sneak']:
        if name in actions:
            logp = logp + dists[name].log_prob(actions[name])
    if 'hotbar' in actions:
        logp = logp + dists['hotbar'].log_prob(actions['hotbar'])
    if include_tactical and 'tactical' in actions:
        logp = logp + dists['tactical'].log_prob(actions['tactical'])
    return logp


def mixed_entropy(output, include_tactical=False):
    dists = _policy_stats(output)
    ent = 0.0
    for name in ['move_fwd', 'move_strafe', 'yaw', 'pitch', 'jump', 'sprint', 'attack', 'use', 'sneak', 'hotbar']:
        ent = ent + dists[name].entropy()
    if include_tactical:
        ent = ent + dists['tactical'].entropy()
    return ent


@dataclass
class RolloutBuffer:
    obs_list: List[dict] = field(default_factory=list)
    action_list: List[dict] = field(default_factory=list)
    reward_list: List[float] = field(default_factory=list)
    done_list: List[bool] = field(default_factory=list)
    logp_list: List[float] = field(default_factory=list)
    value_list: List[float] = field(default_factory=list)
    adv_list: List[float] = field(default_factory=list)
    ret_list: List[float] = field(default_factory=list)

    def add(self, obs, action, reward, done, logp, value):
        self.obs_list.append(obs)
        self.action_list.append(action)
        self.reward_list.append(float(reward))
        self.done_list.append(bool(done))
        self.logp_list.append(float(logp))
        self.value_list.append(float(value))

    def finish(self, last_value=0.0, gamma=0.99, lam=0.95):
        values = self.value_list + [float(last_value)]
        gae = 0.0
        self.adv_list = [0.0] * len(self.reward_list)
        self.ret_list = [0.0] * len(self.reward_list)
        for t in reversed(range(len(self.reward_list))):
            nonterminal = 0.0 if self.done_list[t] else 1.0
            delta = self.reward_list[t] + gamma * values[t + 1] * nonterminal - values[t]
            gae = delta + gamma * lam * nonterminal * gae
            self.adv_list[t] = gae
            self.ret_list[t] = gae + values[t]

    def clear(self):
        self.obs_list.clear()
        self.action_list.clear()
        self.reward_list.clear()
        self.done_list.clear()
        self.logp_list.clear()
        self.value_list.clear()
        self.adv_list.clear()
        self.ret_list.clear()


class MicroPolicyTrainer:
    def __init__(self, model: MicroPolicy, device='cuda', lr=3e-4):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _encode_obs_batch(self, obs_batch):
        return _cat_tensors([parse_obs_to_tensors(obs, self.device) for obs in obs_batch])

    def load_survival_samples(self, jsonl_path, max_samples=50000):
        path = Path(jsonl_path)
        if not path.exists():
            return []
        samples = []
        with path.open('r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if len(samples) >= max_samples:
                    break
                try:
                    rec = json.loads(line)
                    obs = survival_transition_to_policy_obs(rec.get('s', {}), rec.get('tactical', 'continue'))
                    tactical = TACTICAL_MAP.get(rec.get('tactical', 'continue'), 0)
                    reward = float(rec.get('r', 0.0))
                    samples.append({'obs': obs, 'tactical': tactical, 'value': reward})
                except Exception:
                    continue
        return samples

    def load_skill_demo_samples(self, jsonl_path, max_samples=50000):
        path = Path(jsonl_path)
        if not path.exists():
            return []
        skill_to_task = {1: TASK_MAP['build'], 2: TASK_MAP['build'], 3: TASK_MAP['survive'], 4: TASK_MAP['fight']}
        samples = []
        with path.open('r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if len(samples) >= max_samples:
                    break
                try:
                    ep = json.loads(line)
                    frames = ep.get('frames', [])
                    for i in range(len(frames) - 1):
                        if len(samples) >= max_samples:
                            break
                        cur = frames[i]
                        nxt = frames[i + 1]
                        obs = cur.get('obs') or {}
                        nxt_obs = nxt.get('obs') or {}
                        labels = extract_action_labels_from_obs(obs, nxt_obs)
                        if not labels:
                            continue
                        obs = dict(obs)
                        obs['task_id'] = skill_to_task.get(cur.get('behavior_id', 0), TASK_MAP['survive'])
                        samples.append({
                            'obs': obs,
                            'actions': labels,
                            'task_id': obs['task_id'],
                        })
                except Exception:
                    continue
        return samples

    def load_combat_demo_samples(self, jsonl_path, max_samples=50000):
        """Load combat demo data as individual frames (backward compat)."""
        seqs = self.load_combat_demo_sequences(jsonl_path, max_samples=max_samples, seq_len=1)
        # Flatten sequences of length 1 into individual samples
        return [frame for seq in seqs for frame in seq]

    def load_combat_demo_sequences(self, jsonl_path, max_samples=100000, seq_len=32):
        """Load combat demo data as sequences for LSTM training.
        Returns list of sequences, each sequence is a list of {obs, actions, task_id} dicts.
        Consecutive ticks within same mode form sequences."""
        p = Path(jsonl_path)
        if not p.exists():
            return []
        mode_to_task = {'flee': TASK_MAP['flee'], 'fight': TASK_MAP['fight'],
                        'survive': TASK_MAP['survive'], 'navigate': TASK_MAP['navigate']}

        # First pass: load frames (take last max_samples lines for recency)
        # Count total lines first, then skip to the tail
        total_lines = 0
        with p.open('r', encoding='utf-8', errors='ignore') as f:
            for _ in f:
                total_lines += 1
        skip = max(0, total_lines - max_samples)
        frames = []
        nav_count = 0
        combat_count = 0
        import random
        with p.open('r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if i < skip:
                    continue
                if len(frames) >= max_samples:
                    break
                try:
                    d = json.loads(line)
                    obs = d.get('obs', {})
                    act = d.get('action', {})
                    if not act:
                        continue
                    mode = d.get('mode', 'survive')
                    tick = d.get('tick', 0)
                    task_id = mode_to_task.get(mode, TASK_MAP['survive'])
                    # Downsample navigate SEGMENTS (not individual frames) to prevent
                    # dominating training. Random per-frame drop broke LSTM sequences
                    # (loss 1.98→3.59). Now: decide per-segment (keep/drop entire runs).
                    if mode == 'navigate':
                        nav_count += 1
                        # Start of new navigate segment? Decide once per segment.
                        if not hasattr(self, '_nav_seg_keep') or mode != getattr(self, '_nav_last_mode', ''):
                            self._nav_seg_keep = random.random() < 0.4  # keep 40% of navigate SEGMENTS
                        self._nav_last_mode = mode
                        if not self._nav_seg_keep:
                            continue
                    else:
                        combat_count += 1
                        self._nav_last_mode = mode  # reset segment tracking
                    act.setdefault('hotbar', 0)
                    obs['task_id'] = task_id
                    frames.append({
                        'obs': obs, 'actions': act, 'task_id': task_id,
                        'tick': tick, 'mode': mode,
                    })
                except Exception:
                    continue
        print(f"[train_offline] Data balance: combat={combat_count} nav_total={nav_count} nav_kept={sum(1 for f in frames if f['mode']=='navigate')} ratio={sum(1 for f in frames if f['mode']!='navigate')*100//max(len(frames),1)}% combat")

        # Second pass: cut into sequences of consecutive ticks (same mode, tick gap <= 5)
        sequences = []
        current_seq = []
        for i, frame in enumerate(frames):
            if current_seq:
                prev = current_seq[-1]
                tick_gap = abs(frame['tick'] - prev['tick'])
                mode_changed = frame['mode'] != prev['mode']
                if tick_gap > 5 or mode_changed or len(current_seq) >= seq_len:
                    # End current sequence
                    if len(current_seq) >= 4:  # min 4 frames to be useful
                        sequences.append(current_seq)
                    current_seq = []
            current_seq.append(frame)
        if len(current_seq) >= 4:
            sequences.append(current_seq)

        return sequences

    def train_tactical_supervised(self, samples, epochs=3, batch_size=256):
        if not samples:
            return {'loss': 0.0, 'n': 0}
        self.model.train()
        total_loss = 0.0
        total_n = 0
        for _ in range(epochs):
            for start in range(0, len(samples), batch_size):
                batch = samples[start:start + batch_size]
                encoded = self._encode_obs_batch([x['obs'] for x in batch])
                output, _ = self.model(*encoded, hidden=None)
                target_tactical = torch.tensor([x['tactical'] for x in batch], dtype=torch.long, device=self.device)
                target_value = torch.tensor([x['value'] for x in batch], dtype=torch.float32, device=self.device)
                loss_cls = F.cross_entropy(output['tactical'], target_tactical)
                loss_val = F.mse_loss(output['value'], target_value)
                loss = loss_cls + 0.25 * loss_val
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                total_loss += loss.item() * len(batch)
                total_n += len(batch)
        return {'loss': total_loss / max(total_n, 1), 'n': total_n}

    def train_imitation_supervised(self, samples, epochs=3, batch_size=128):
        if not samples:
            return {'loss': 0.0, 'n': 0}
        self.model.train()
        total_loss = 0.0
        total_n = 0
        for _ in range(epochs):
            for start in range(0, len(samples), batch_size):
                batch = samples[start:start + batch_size]
                encoded = self._encode_obs_batch([x['obs'] for x in batch])
                output, _ = self.model(*encoded, hidden=None)

                tgt_fwd = torch.tensor([x['actions']['move_fwd'] for x in batch], dtype=torch.float32, device=self.device)
                tgt_strafe = torch.tensor([x['actions']['move_strafe'] for x in batch], dtype=torch.float32, device=self.device)
                tgt_yaw = torch.tensor([x['actions']['yaw'] for x in batch], dtype=torch.float32, device=self.device)
                tgt_pitch = torch.tensor([x['actions']['pitch'] for x in batch], dtype=torch.float32, device=self.device)
                tgt_jump = torch.tensor([x['actions']['jump'] for x in batch], dtype=torch.float32, device=self.device)
                tgt_sprint = torch.tensor([x['actions']['sprint'] for x in batch], dtype=torch.float32, device=self.device)
                tgt_attack = torch.tensor([x['actions']['attack'] for x in batch], dtype=torch.float32, device=self.device)
                tgt_use = torch.tensor([x['actions']['use'] for x in batch], dtype=torch.float32, device=self.device)
                tgt_sneak = torch.tensor([x['actions']['sneak'] for x in batch], dtype=torch.float32, device=self.device)
                tgt_hotbar = torch.tensor([x['actions']['hotbar'] for x in batch], dtype=torch.long, device=self.device)
                tgt_task = torch.tensor([x['task_id'] for x in batch], dtype=torch.long, device=self.device)

                loss = 0.0
                loss = loss + F.mse_loss(output['move_fwd'][:, 0], tgt_fwd)
                loss = loss + F.mse_loss(output['move_strafe'][:, 0], tgt_strafe)
                loss = loss + 0.02 * F.mse_loss(output['yaw'][:, 0] * 30.0, tgt_yaw)
                loss = loss + 0.02 * F.mse_loss(output['pitch'][:, 0] * 15.0, tgt_pitch)
                loss = loss + F.binary_cross_entropy_with_logits(output['jump'].squeeze(-1), tgt_jump)
                loss = loss + F.binary_cross_entropy_with_logits(output['sprint'].squeeze(-1), tgt_sprint)
                loss = loss + F.binary_cross_entropy_with_logits(output['attack'].squeeze(-1), tgt_attack)
                loss = loss + F.binary_cross_entropy_with_logits(output['use'].squeeze(-1), tgt_use)
                loss = loss + F.binary_cross_entropy_with_logits(output['sneak'].squeeze(-1), tgt_sneak)
                loss = loss + F.cross_entropy(output['hotbar'], tgt_hotbar)
                loss = loss + 0.2 * F.cross_entropy(output['tactical'], tgt_task)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                total_loss += loss.item() * len(batch)
                total_n += len(batch)
        return {'loss': total_loss / max(total_n, 1), 'n': total_n}

    def _imitation_loss(self, output, frame, value_target=None, outcome_weight=1.0):
        """Compute imitation loss for a single frame.
        Optional: value_target for value head training, outcome_weight for survival weighting."""
        d = self.device
        tgt_fwd = torch.tensor([frame['actions']['move_fwd']], dtype=torch.float32, device=d)
        tgt_strafe = torch.tensor([frame['actions']['move_strafe']], dtype=torch.float32, device=d)
        tgt_yaw = torch.tensor([frame['actions']['yaw']], dtype=torch.float32, device=d)
        tgt_pitch = torch.tensor([frame['actions']['pitch']], dtype=torch.float32, device=d)
        tgt_jump = torch.tensor([frame['actions']['jump']], dtype=torch.float32, device=d)
        tgt_sprint = torch.tensor([frame['actions']['sprint']], dtype=torch.float32, device=d)
        tgt_attack = torch.tensor([frame['actions']['attack']], dtype=torch.float32, device=d)
        tgt_use = torch.tensor([frame['actions']['use']], dtype=torch.float32, device=d)
        tgt_sneak = torch.tensor([frame['actions']['sneak']], dtype=torch.float32, device=d)
        tgt_hotbar = torch.tensor([frame['actions']['hotbar']], dtype=torch.long, device=d)
        tgt_task = torch.tensor([frame['task_id']], dtype=torch.long, device=d)
        fl = F.mse_loss(output['move_fwd'][:, 0], tgt_fwd)
        fl = fl + F.mse_loss(output['move_strafe'][:, 0], tgt_strafe)
        fl = fl + 0.02 * F.mse_loss(output['yaw'][:, 0] * 30.0, tgt_yaw)
        fl = fl + 0.02 * F.mse_loss(output['pitch'][:, 0] * 15.0, tgt_pitch)
        fl = fl + F.binary_cross_entropy_with_logits(output['jump'].squeeze(-1), tgt_jump)
        fl = fl + F.binary_cross_entropy_with_logits(output['sprint'].squeeze(-1), tgt_sprint)
        fl = fl + F.binary_cross_entropy_with_logits(output['attack'].squeeze(-1), tgt_attack)
        fl = fl + F.binary_cross_entropy_with_logits(output['use'].squeeze(-1), tgt_use)
        fl = fl + F.binary_cross_entropy_with_logits(output['sneak'].squeeze(-1), tgt_sneak)
        fl = fl + F.cross_entropy(output['hotbar'], tgt_hotbar)
        fl = fl + 0.2 * F.cross_entropy(output['tactical'], tgt_task)
        # P1: Value head training during hindsight
        if value_target is not None:
            vt = torch.tensor([value_target], dtype=torch.float32, device=d)
            fl = fl + 0.1 * F.mse_loss(output['value'], vt)
        # P2: Outcome-weighted loss
        fl = fl * outcome_weight
        return fl

    def _relabel_goal_from_endpoint(self, frame, end_px, end_pz):
        """Replace obs goal with local goal: direction vector to trajectory endpoint."""
        obs = frame['obs']
        cur_px = obs.get('px', 0) or 0
        cur_pz = obs.get('pz', 0) or 0
        dx = end_px - cur_px
        dz = end_pz - cur_pz
        dist = (dx**2 + dz**2) ** 0.5
        # Write direction vector into goal fields (not baritone fields)
        obs['goal_dx'] = dx
        obs['goal_dz'] = dz
        obs['goal_dist'] = dist
        return frame

    def _sample_trajectories(self, all_frames, num_trajs=2000, min_len=3, max_len=50):
        """Sample random-length trajectories from continuous frame data.
        Each trajectory ends at a random point; the endpoint's position becomes the local goal.
        Samples backwards: pick an endpoint, then take the preceding min_len..max_len frames."""
        import random
        trajs = []
        if len(all_frames) < min_len + 1:
            return trajs
        for _ in range(num_trajs):
            traj_len = random.randint(min_len, max_len)
            # Pick a random endpoint (must have room for traj_len frames before it)
            end_idx = random.randint(traj_len, len(all_frames) - 1)
            start_idx = end_idx - traj_len
            # Check continuity: ticks should be roughly consecutive (gap <= 5 each step)
            segment = all_frames[start_idx:end_idx + 1]
            continuous = True
            for j in range(1, len(segment)):
                if abs(segment[j]['tick'] - segment[j-1]['tick']) > 5:
                    continuous = False
                    break
                if segment[j]['mode'] != segment[j-1]['mode']:
                    continuous = False
                    break
            if not continuous:
                continue
            trajs.append(segment)
        return trajs

    def train_imitation_hindsight(self, all_frames, epochs=2, trajs_per_epoch=2000,
                                   min_len=3, max_len=50, seqs_per_batch=8):
        """Hindsight goal relabeling: sample trajectories, use endpoint as local goal.
        Frame-level (short traj 1-5) = micro decisions: 'how to reach this block'
        Sequence-level (long traj 5-50) = path planning: 'how to navigate to that spot'
        LSTM hidden carried across trajectory for temporal learning."""
        if not all_frames:
            return {'loss': 0.0, 'n': 0}
        import random, copy
        self.model.train()
        total_loss = 0.0
        total_n = 0

        for _ in range(epochs):
            trajs = self._sample_trajectories(all_frames, num_trajs=trajs_per_epoch,
                                               min_len=min_len, max_len=max_len)
            random.shuffle(trajs)

            for si in range(0, len(trajs), seqs_per_batch):
                batch_trajs = trajs[si:si + seqs_per_batch]
                batch_loss = 0.0
                batch_frames = 0

                for traj in batch_trajs:
                    # Endpoint position = local goal
                    end_frame = traj[-1]
                    end_px = end_frame['obs'].get('px', 0) or 0
                    end_pz = end_frame['obs'].get('pz', 0) or 0

                    hidden = self.model.init_hidden(1)
                    traj_loss = 0.0

                    # Compute initial distance for value target
                    first_px = traj[0]['obs'].get('px', 0) or 0
                    first_pz = traj[0]['obs'].get('pz', 0) or 0
                    initial_dist = max(((end_px - first_px)**2 + (end_pz - first_pz)**2)**0.5, 1.0)

                    # P2: Compute outcome weight for this trajectory
                    final_hp = traj[-1]['obs'].get('hp', 20) or 20
                    initial_hp = traj[0]['obs'].get('hp', 20) or 20
                    died = final_hp <= 0
                    if died:
                        traj_outcome_weight = 0.3  # learn something from failures, but down-weighted
                    else:
                        hp_ratio = final_hp / max(initial_hp, 1)
                        traj_outcome_weight = 0.5 + 0.5 * min(hp_ratio, 1.0)  # 0.5 to 1.0

                    traj_len = len(traj) - 1
                    # Process frames in order (forward), but goal is the endpoint (backward logic)
                    for fi, frame in enumerate(traj[:-1]):  # exclude last frame (it IS the goal)
                        # Deep copy obs to avoid mutating original data
                        relabeled = copy.deepcopy(frame)
                        relabeled = self._relabel_goal_from_endpoint(relabeled, end_px, end_pz)

                        encoded = self._encode_obs_batch([relabeled['obs']])
                        output, hidden = self.model(*encoded, hidden=hidden)
                        hidden = (hidden[0].detach(), hidden[1].detach())

                        # P1: Value target = progress toward goal (time + distance)
                        cur_px = frame['obs'].get('px', 0) or 0
                        cur_pz = frame['obs'].get('pz', 0) or 0
                        remaining = ((end_px - cur_px)**2 + (end_pz - cur_pz)**2)**0.5
                        progress = fi / max(traj_len, 1)
                        dist_value = max(0, 1 - remaining / initial_dist)
                        value_target = 0.5 * progress + 0.5 * dist_value

                        traj_loss = traj_loss + self._imitation_loss(
                            output, relabeled,
                            value_target=value_target,
                            outcome_weight=traj_outcome_weight,
                        )
                        batch_frames += 1

                    if len(traj) > 1:
                        batch_loss = batch_loss + traj_loss / (len(traj) - 1)

                if batch_frames > 0:
                    avg_loss = batch_loss / len(batch_trajs)
                    self.optimizer.zero_grad(set_to_none=True)
                    avg_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    total_loss += avg_loss.item() * batch_frames
                    total_n += batch_frames

        return {'loss': total_loss / max(total_n, 1), 'n': total_n}

    def train_ppo(self, rollout: RolloutBuffer, epochs=4, batch_size=64, clip_ratio=0.2, vf_coef=0.5, ent_coef=0.01, freeze_shared=False):
        if not rollout.obs_list:
            return {'loss': 0.0, 'n': 0}
        self.model.train()

        # Freeze shared perception layers — PPO only updates decision layers
        frozen_modules = []
        if freeze_shared:
            frozen_modules = [self.model.terrain_cnn, self.model.threat_attn,
                              self.model.goal_attn, self.model.other_mlp,
                              self.model.task_embed, self.model.lstm]
            for m in frozen_modules:
                for p in m.parameters():
                    p.requires_grad = False

        rollout.finish()
        adv = torch.tensor(rollout.adv_list, dtype=torch.float32, device=self.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        total_loss = 0.0
        total_n = 0
        for _ in range(epochs):
            for start in range(0, len(rollout.obs_list), batch_size):
                end = start + batch_size
                idx = slice(start, end)
                encoded = self._encode_obs_batch(rollout.obs_list[idx])
                output, _ = self.model(*encoded, hidden=None)
                batch_actions = {}
                keys = ['move_fwd', 'move_strafe', 'yaw', 'pitch', 'jump', 'sprint', 'attack', 'use', 'sneak', 'hotbar']
                for key in keys:
                    vals = [rollout.action_list[i][key] for i in range(start, min(end, len(rollout.action_list))) if key in rollout.action_list[i]]
                    if not vals:
                        continue
                    dtype = torch.long if key == 'hotbar' else torch.float32
                    batch_actions[key] = torch.tensor(vals, dtype=dtype, device=self.device)
                old_logp = torch.tensor(rollout.logp_list[idx], dtype=torch.float32, device=self.device)
                batch_adv = adv[idx]
                batch_ret = torch.tensor(rollout.ret_list[idx], dtype=torch.float32, device=self.device)

                logp = mixed_logprob(output, batch_actions, include_tactical=False)
                ratio = torch.exp(logp - old_logp)
                clip_adv = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * batch_adv
                loss_pi = -(torch.min(ratio * batch_adv, clip_adv)).mean()
                loss_v = F.mse_loss(output['value'], batch_ret)
                entropy = mixed_entropy(output, include_tactical=False).mean()
                loss = loss_pi + vf_coef * loss_v - ent_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                total_loss += loss.item() * len(batch_ret)
                total_n += len(batch_ret)

        # Unfreeze shared layers after PPO
        if freeze_shared:
            for m in frozen_modules:
                for p in m.parameters():
                    p.requires_grad = True

        return {'loss': total_loss / max(total_n, 1), 'n': total_n}

    def save_checkpoint(self, path):
        ckpt = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(ckpt, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer'])


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--survival-jsonl', default=None)
    parser.add_argument('--skill-jsonl', default=None)
    parser.add_argument('--out', default='micro_policy.pt')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    model = MicroPolicy(device=args.device)
    trainer = MicroPolicyTrainer(model, device=args.device)
    if args.survival_jsonl:
        stats = trainer.train_tactical_supervised(trainer.load_survival_samples(args.survival_jsonl))
        print(f"tactical: n={stats['n']} loss={stats['loss']:.4f}")
    if args.skill_jsonl:
        stats = trainer.train_imitation_supervised(trainer.load_skill_demo_samples(args.skill_jsonl))
        print(f"imitation: n={stats['n']} loss={stats['loss']:.4f}")
    trainer.save_checkpoint(args.out)
    print(f"saved: {args.out}")


if __name__ == '__main__':
    main()
