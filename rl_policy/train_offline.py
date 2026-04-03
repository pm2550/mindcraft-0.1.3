"""Offline training: load checkpoint, train, save. No server needed."""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import torch
from model import MicroPolicy
from trainer import MicroPolicyTrainer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
base_dir = os.path.join(os.path.dirname(__file__), '..', 'bots', 'GaoBot1')

# Paths
ckpt_in = os.path.join(base_dir, 'micro_policy_v2.pt')
ckpt_out = ckpt_in  # overwrite in-place
survival_jsonl = os.path.join(base_dir, 'survival_episodes.jsonl')
skill_jsonl = os.path.join(base_dir, 'skill_demos.jsonl')
# Use recent subset if available (pre-extracted by tail), else full file
combat_recent = os.path.join(base_dir, 'combat_demos_recent.jsonl')
combat_jsonl = combat_recent if os.path.exists(combat_recent) else os.path.join(base_dir, 'combat_demos.jsonl')

print(f"[train_offline] device={device}")
model = MicroPolicy(device=device)
trainer = MicroPolicyTrainer(model, device=device)

# Load existing checkpoint
if os.path.exists(ckpt_in):
    print(f"[train_offline] Loading {ckpt_in}")
    trainer.load_checkpoint(ckpt_in)
else:
    print(f"[train_offline] No checkpoint found, training from scratch")

# Tactical training
if os.path.exists(survival_jsonl):
    print(f"[train_offline] Training tactical...")
    t0 = time.time()
    samples = trainer.load_survival_samples(survival_jsonl, max_samples=50000)
    stats = trainer.train_tactical_supervised(samples, epochs=3, batch_size=256)
    print(f"[train_offline] tactical: n={stats['n']} loss={stats['loss']:.4f} ({time.time()-t0:.1f}s)")
else:
    print(f"[train_offline] No survival data at {survival_jsonl}")

# Imitation training (skill demos) — DISABLED: skill_demos.jsonl has action=None
# due to freezeMicro=true during skill execution. Training on this data
# corrupts shared model parameters (loss explodes to 90-147).
# TODO: re-enable when skill demo collection records real actions.
print(f"[train_offline] Skipping skill imitation (data quality too poor, action=None)")

# Combat imitation training — two phases:
# Phase 1: Frame-level with hindsight goal (short traj 1-5): micro decisions
# Phase 2: Sequence-level with hindsight goal (long traj 5-50): path planning + LSTM
if os.path.exists(combat_jsonl):
    # Load all frames in order (needed for trajectory sampling)
    print(f"[train_offline] Loading combat demos...")
    t0 = time.time()
    all_frames = []
    sequences = trainer.load_combat_demo_sequences(combat_jsonl, max_samples=400000, seq_len=9999)
    for seq in sequences:
        all_frames.extend(seq)
    print(f"[train_offline] Loaded {len(all_frames)} frames in {len(sequences)} continuous segments ({time.time()-t0:.1f}s)")

    # Phase 1: Frame-level (short trajectories 1-5 frames, micro decisions)
    # "How do I get to this nearby block?" — immediate obstacle avoidance
    print(f"[train_offline] Phase 1: Frame-level hindsight (traj 1-5)...")
    t0 = time.time()
    stats = trainer.train_imitation_hindsight(all_frames, epochs=2, trajs_per_epoch=3000,
                                              min_len=1, max_len=5, seqs_per_batch=16)
    print(f"[train_offline] frame hindsight: n={stats['n']} loss={stats['loss']:.4f} ({time.time()-t0:.1f}s)")

    # Phase 2: Sequence-level (long trajectories 5-50 frames, path planning)
    # "How do I navigate to that spot 20 blocks away?" — LSTM temporal patterns
    print(f"[train_offline] Phase 2: Sequence-level hindsight (traj 5-50, LSTM)...")
    t0 = time.time()
    stats = trainer.train_imitation_hindsight(all_frames, epochs=2, trajs_per_epoch=2000,
                                              min_len=5, max_len=50, seqs_per_batch=8)
    print(f"[train_offline] seq hindsight: n={stats['n']} loss={stats['loss']:.4f} ({time.time()-t0:.1f}s)")
else:
    print(f"[train_offline] No combat demos yet at {combat_jsonl}")

# Save
trainer.save_checkpoint(ckpt_out)
print(f"[train_offline] Saved to {ckpt_out}")

# Hot-reload into running server via TCP
import socket, json
try:
    s = socket.socket()
    s.settimeout(5)
    s.connect(('127.0.0.1', 7860))
    s.setsockopt(6, 1, 1)
    s.sendall((json.dumps({'type':'load','data':{'path': os.path.relpath(ckpt_out, os.path.dirname(__file__))}}) + '\n').encode())
    data = b''
    while b'\n' not in data:
        data += s.recv(65536)
    print(f"[train_offline] Hot-reloaded into server: {data.decode().strip()[:100]}")
    s.close()
except Exception as e:
    print(f"[train_offline] Hot-reload failed (server busy?): {e}")

print("[train_offline] Done!")
