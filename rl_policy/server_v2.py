"""Start server with v2 checkpoint pre-loaded."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from server import PolicyServer
import torch

server = PolicyServer(host='127.0.0.1', port=7860, device='cuda:0')

# Load trained v2 checkpoint
ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'bots', 'GaoBot1', 'micro_policy_v2.pt')
if os.path.exists(ckpt_path):
    print(f"[server_v2] Loading checkpoint: {ckpt_path}")
    server.trainer.load_checkpoint(ckpt_path)
    server.hidden = server.model.init_hidden(1)
    print("[server_v2] Checkpoint loaded!")
else:
    print(f"[server_v2] No checkpoint found at {ckpt_path}, using fresh model")

server.start()
