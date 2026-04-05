"""
Observation/action encoding shared by the policy server and training code.
"""

from __future__ import annotations

import torch


TASK_MAP = {
    'idle': 0,
    'fight': 1,
    'flee': 2,
    'survive': 3,
    'navigate': 4,
    'mine': 5,
    'build': 6,
    'continue': 0,
    'hide': 3,
}

TACTICAL_MAP = {
    'continue': 0,
    'fight': 1,
    'flee': 2,
    'hide': 3,
    'eat': 3,
    'equip_weapon': 1,
    'equip_armor': 3,
    'abort_task': 0,
    'hunt_food': 5,
    'sleep': 3,
}


def _extract_terrain_window(terrain_raw):
    if len(terrain_raw) == 2601:
        terrain = []
        for dz in range(-10, 11):
            for dx in range(-10, 11):
                idx = (dz + 25) * 51 + (dx + 25)
                terrain.append(terrain_raw[idx] if idx < len(terrain_raw) else 0)
        return terrain
    if len(terrain_raw) == 441:
        return terrain_raw
    return [0] * 441


def parse_obs_to_tensors(obs, device):
    """
    Convert one obs dict into the batched tensor tuple expected by MicroPolicy.
    """
    d = device
    terrain_raw = obs.get('terrain', [])
    terrain_2d = _extract_terrain_window(terrain_raw)  # 441 values

    # 3D terrain: heightmap + ceilmap (each 51x51 or 441 from 21x21)
    hm_raw = obs.get('heightmap', [])
    hm_21 = _extract_terrain_window(hm_raw) if len(hm_raw) > 0 else [0] * 441
    cm_raw = obs.get('ceilmap', [])
    cm_21 = _extract_terrain_window(cm_raw) if len(cm_raw) > 0 else [0] * 441

    # Stack 3 channels: terrain(0-3) + heightmap(-12..12) + ceilmap(0..8) → (1, 3*441)
    terrain = torch.tensor([terrain_2d + hm_21 + cm_21], dtype=torch.float32, device=d)

    self_state = torch.tensor([[
        (obs.get('hp', 20)) / 20,
        (obs.get('food', 20)) / 20,
        (obs.get('armor', 0)) / 20,
        1.0 if obs.get('onGround') else 0.0,
        1.0 if obs.get('sprinting') else 0.0,
        obs.get('vx', 0),
        obs.get('vy', 0),
        obs.get('vz', 0),
        (obs.get('selectedSlot', 0)) / 8,
        obs.get('attackCooldown', 0),
        1.0 if obs.get('los') else 0.0,
        1.0 if obs.get('canDigDown') else 0.0,
    ]], dtype=torch.float32, device=d)

    threats_raw = obs.get('threats', [])
    threats = torch.zeros(1, 4, 4, device=d)
    threat_mask = torch.zeros(1, 4, device=d)
    for i, t in enumerate(threats_raw[:4]):
        if t and isinstance(t, dict):
            threats[0, i] = torch.tensor([
                (t.get('dx', 0)) / 25,
                (t.get('dz', 0)) / 25,
                min(t.get('dist', 32), 32) / 32,
                1.0,
            ], dtype=torch.float32, device=d)
            threat_mask[0, i] = 1.0

    b = obs.get('baritone', {})
    # Goal direction vector: filled by hindsight relabel during training, 0 during inference
    # PPO learns which direction to go autonomously
    goal = torch.tensor([[
        min(obs.get('_anchorDist', 0), 48) / 48,
        (obs.get('_anchorDx', 0)) / 48,
        (obs.get('_anchorDz', 0)) / 48,
        (obs.get('goal_dx', 0)) / 48,
        (obs.get('goal_dz', 0)) / 48,
        min(abs(obs.get('goal_dist', 0)), 96) / 96,
    ]], dtype=torch.float32, device=d)

    temporal = torch.tensor([[
        min(len(threats_raw), 5) / 5,
        1.0 if obs.get('_recentDamage') else 0.0,
        min(obs.get('_timeInMode', 0), 100) / 100,
        max(-1, min(1, (obs.get('_lastDyaw', 0)) / 30)),
        obs.get('_speed', 0),
    ]], dtype=torch.float32, device=d)

    # Baritone state: only pathing flag + key inputs (no goal fields)
    baritone = torch.zeros(1, 20, device=d)
    baritone[0, 0] = 1.0 if b.get('pathing') else 0.0
    # slots 1-9 reserved (were goal fields, now unused — kept for tensor shape compat)
    baritone[0, 10] = 1.0 if obs.get('keyFwd') else 0.0
    baritone[0, 11] = 1.0 if obs.get('keyBack') else 0.0
    baritone[0, 12] = 1.0 if obs.get('keyLeft') else 0.0
    baritone[0, 13] = 1.0 if obs.get('keyRight') else 0.0
    baritone[0, 14] = 1.0 if obs.get('keyJump') else 0.0
    baritone[0, 15] = 1.0 if obs.get('keySneak') else 0.0
    baritone[0, 16] = 1.0 if obs.get('keySprint') else 0.0
    baritone[0, 17] = 1.0 if obs.get('keyAttack') else 0.0
    baritone[0, 18] = 1.0 if obs.get('keyUse') else 0.0
    baritone[0, 19] = (obs.get('hotbar', 0)) / 8

    task_id = torch.tensor([obs.get('task_id', 0)], dtype=torch.long, device=d)
    return terrain, self_state, threats, threat_mask, goal, temporal, baritone, task_id


def survival_transition_to_policy_obs(state_dict, tactical='continue'):
    """
    Expand a compact survival transition state into a sparse policy obs.
    This is enough to bootstrap the tactical head from existing logs.
    """
    state = state_dict or {}
    return {
        'hp': state.get('hp', 20),
        'food': state.get('hg', 20),
        'armor': state.get('arm', 0),
        'onGround': True,
        'sprinting': False,
        'vx': 0.0,
        'vy': 0.0,
        'vz': 0.0,
        'selectedSlot': 0,
        'attackCooldown': 0.0,
        'los': state.get('mc', 0) > 0,
        'canDigDown': state.get('blk', 0) > 0,
        'terrain': [0] * 441,
        'threats': ([] if state.get('mc', 0) <= 0 else [{
            'dx': 0.0,
            'dz': min(state.get('md', 32), 32),
            'dist': min(state.get('md', 32), 32),
            'type': state.get('mt') or 'unknown',
        }]),
        'baritone': {},
        'keyFwd': False,
        'keyBack': False,
        'keyLeft': False,
        'keyRight': False,
        'keyJump': False,
        'keySneak': False,
        'keySprint': False,
        'keyAttack': False,
        'keyUse': False,
        'hotbar': 0,
        '_anchorDist': 0.0,
        '_anchorDx': 0.0,
        '_anchorDz': 0.0,
        '_recentDamage': False,
        '_timeInMode': 0.0,
        '_lastDyaw': 0.0,
        '_speed': 0.0,
        'task_id': TASK_MAP.get(tactical, 0),
    }


def extract_action_labels_from_obs(prev_obs, next_obs=None):
    """
    Build low-level action labels from actual key states. Returns None if missing.
    """
    if not prev_obs:
        return None
    has_keys = any(k in prev_obs for k in (
        'keyFwd', 'keyBack', 'keyLeft', 'keyRight',
        'keyJump', 'keySprint', 'keyAttack', 'keyUse', 'keySneak'
    ))
    if not has_keys:
        return None
    move_fwd = 1.0 if prev_obs.get('keyFwd') else (-1.0 if prev_obs.get('keyBack') else 0.0)
    move_strafe = 1.0 if prev_obs.get('keyRight') else (-1.0 if prev_obs.get('keyLeft') else 0.0)
    yaw = 0.0
    pitch = 0.0
    if next_obs:
        yaw = (next_obs.get('yaw', prev_obs.get('yaw', 0)) - prev_obs.get('yaw', 0))
        pitch = (next_obs.get('pitch', prev_obs.get('pitch', 0)) - prev_obs.get('pitch', 0))
    return {
        'move_fwd': move_fwd,
        'move_strafe': move_strafe,
        'yaw': yaw,
        'pitch': pitch,
        'jump': 1.0 if prev_obs.get('keyJump') else 0.0,
        'sprint': 1.0 if prev_obs.get('keySprint') else 0.0,
        'attack': 1.0 if prev_obs.get('keyAttack') else 0.0,
        'use': 1.0 if prev_obs.get('keyUse') else 0.0,
        'sneak': 1.0 if prev_obs.get('keySneak') else 0.0,
        'hotbar': int(prev_obs.get('hotbar', 0) or 0),
    }
