/**
 * PolicyClient — Async streaming client for PyTorch policy server.
 *
 * Design: fire-and-forget sends + cached latest action.
 *   - sendObs(): non-blocking, sends obs immediately, does NOT await response
 *   - getLatestAction(): returns most recent valid action if fresh enough
 *   - Server responses arrive asynchronously, update cached action
 *   - seq numbers prevent stale action from overriding newer state
 */

import net from 'net';

export class PolicyClient {
    constructor(host = '127.0.0.1', port = 7860) {
        this.host = host;
        this.port = port;
        this.socket = null;
        this.connected = false;
        this._buffer = '';
        this._reconnectTimer = null;

        // Sequence tracking
        this._sendSeq = 0;       // incremented on each send
        this._recvSeq = 0;       // seq of latest received action
        this._lastAction = null;  // cached action from server
        this._lastActionAt = 0;   // timestamp of last valid action

        // Stats
        this._obsCount = 0;
        this._actionCount = 0;
        this._avgLatencyMs = 0;
        this._lastSendAt = 0;
    }

    connect() {
        if (this.socket) return;
        this.socket = new net.Socket();
        this.socket.setNoDelay(true);

        this.socket.on('connect', () => {
            this.connected = true;
            this._buffer = '';
            console.log(`[PolicyClient] Connected to ${this.host}:${this.port}`);
        });

        this.socket.on('data', (data) => {
            this._buffer += data.toString();
            while (this._buffer.includes('\n')) {
                const idx = this._buffer.indexOf('\n');
                const line = this._buffer.slice(0, idx);
                this._buffer = this._buffer.slice(idx + 1);
                if (!line.trim()) continue;
                try {
                    const msg = JSON.parse(line);
                    this._onResponse(msg);
                } catch (e) { /* ignore parse errors */ }
            }
        });

        this.socket.on('error', (err) => {
            if (err.code !== 'ECONNREFUSED') {
                console.log(`[PolicyClient] Error: ${err.message}`);
            }
        });

        this.socket.on('close', () => {
            this.connected = false;
            this.socket = null;
            if (!this._reconnectTimer) {
                this._reconnectTimer = setTimeout(() => {
                    this._reconnectTimer = null;
                    this.connect();
                }, 3000);
            }
        });

        this.socket.connect(this.port, this.host);
    }

    /**
     * Handle response from server. Updates cached action if seq is fresh.
     */
    _onResponse(msg) {
        if (msg.type === 'action' && msg.data) {
            const seq = msg.seq || 0;
            // Only accept if this response is for a recent request (not stale)
            if (seq >= this._recvSeq) {
                this._recvSeq = seq;
                this._lastAction = msg.data;
                this._lastActionAt = Date.now();
                this._actionCount++;

                // Latency tracking
                if (this._lastSendAt > 0) {
                    const lat = Date.now() - this._lastSendAt;
                    this._avgLatencyMs = this._avgLatencyMs * 0.9 + lat * 0.1;
                }
            }
        }
    }

    /**
     * Fire-and-forget: send obs to server, don't wait for response.
     * Non-blocking. Call this every tick.
     */
    sendObs(obs) {
        if (!this.connected || !this.socket) return;
        this._sendSeq++;
        this._obsCount++;
        this._lastSendAt = Date.now();
        try {
            const msg = JSON.stringify({ type: 'obs', seq: this._sendSeq, data: obs }) + '\n';
            this.socket.write(msg);
        } catch (e) { /* ignore write errors */ }
    }

    /**
     * Get the most recent valid action from server.
     * Returns null if no fresh action available (caller should fallback to JS micro).
     *
     * @param maxAgeMs - max age of cached action to consider valid (default 100ms)
     */
    getLatestAction(maxAgeMs = 100) {
        if (!this._lastAction) return null;
        const age = Date.now() - this._lastActionAt;
        if (age > maxAgeMs) return null;
        return this._lastAction;
    }

    /**
     * Reset LSTM hidden state (on death, respawn, mode change).
     */
    resetHidden() {
        if (!this.connected || !this.socket) return;
        try {
            this.socket.write(JSON.stringify({ type: 'reset_hidden' }) + '\n');
        } catch (e) { /* ignore */ }
    }

    /**
     * Send rollout buffer to server for PPO training.
     */
    sendRollout(rolloutBuffer) {
        if (!this.connected || !this.socket || !rolloutBuffer || rolloutBuffer.length < 10) return;
        try {
            const msg = JSON.stringify({ type: 'train_ppo', data: rolloutBuffer }) + '\n';
            this.socket.write(msg);
            console.log(`[PolicyClient] Sent rollout: ${rolloutBuffer.length} steps for PPO`);
        } catch (e) { /* ignore write errors */ }
    }

    /**
     * Convert raw rl_obs from Java into the format expected by policy server.
     */
    static formatObs(obs, constraints, mode) {
        if (!obs) return null;

        const b = obs.baritone || {};
        const threats = obs.threats || [];

        return {
            hp: obs.hp || 20,
            food: obs.food || 20,
            armor: obs.armor || 0,
            onGround: !!obs.onGround,
            sprinting: !!obs.sprinting,
            vx: obs.vx || 0,
            vy: obs.vy || 0,
            vz: obs.vz || 0,
            selectedSlot: obs.selectedSlot || 0,
            attackCooldown: obs.attackCooldown || 0,
            los: !!obs.los,
            canDigDown: !!obs.canDigDown,
            inWater: !!obs.inWater,
            yaw: obs.yaw || 0,
            pitch: obs.pitch || 0,
            px: obs.px,
            pz: obs.pz,
            terrain: obs.terrain || [],
            threats: threats.slice(0, 4).map(t => t ? {
                dx: t.dx || 0, dz: t.dz || 0, dy: t.dy || 0, dist: t.dist || 32,
                hp: t.hp ?? -1, type: t.type || 'unknown',
            } : null).filter(Boolean),
            // Goal direction: filled by hindsight during training, 0 during inference/collection
            // PPO learns which direction to go on its own
            goal_dx: 0,
            goal_dz: 0,
            goal_dist: 0,
            // Baritone state: only pathing flag + key inputs (no goal — goal comes from hindsight)
            baritone: {
                pathing: !!b.pathing,
            },
            keyFwd: !!obs.keyFwd, keyBack: !!obs.keyBack,
            keyLeft: !!obs.keyLeft, keyRight: !!obs.keyRight,
            keyJump: !!obs.keyJump, keySneak: !!obs.keySneak,
            keySprint: !!obs.keySprint, keyAttack: !!obs.keyAttack,
            keyUse: !!obs.keyUse, hotbar: obs.hotbar || 0,
            _anchorDist: constraints?._anchorDist || 0,
            _anchorDx: constraints?._anchorDx || 0,
            _anchorDz: constraints?._anchorDz || 0,
            _recentDamage: !!constraints?._recentDamage,
            _timeInMode: constraints?._timeInMode || 0,
            _lastDyaw: constraints?._lastDyaw || 0,
            _speed: constraints?._speed || 0,
            task_id: { 'idle': 0, 'fight': 1, 'flee': 2, 'survive': 3,
                       'navigate': 4, 'mine': 5, 'build': 6 }[mode] || 0,
        };
    }

    /**
     * Convert policy server action dict into rl_action format for Java.
     */
    static actionToRlAction(action) {
        if (!action) return null;
        // Clamp yaw/pitch to prevent wild spinning from untrained model
        const dyaw = Math.max(-15, Math.min(15, action.yaw || 0));
        const dpitch = Math.max(-10, Math.min(10, action.pitch || 0));
        const result = {
            forward: (action.move_fwd || 0) > 0.3,
            back: (action.move_fwd || 0) < -0.3,
            left: (action.move_strafe || 0) < -0.3,
            right: (action.move_strafe || 0) > 0.3,
            jump: !!(action.jump),
            sprint: !!(action.sprint),
            attack: !!(action.attack),
            dyaw,
            dpitch,
            // Disable untrained heads to prevent random spam:
            // use/sneak/hotbar will be enabled once imitation training is done
            // use: !!(action.use),
            // sneak: !!(action.sneak),
            // selectSlot: action.hotbar != null ? Math.round(action.hotbar) : undefined,
        };
        return result;
    }

    get stats() {
        return {
            connected: this.connected,
            obsCount: this._obsCount,
            actionCount: this._actionCount,
            avgLatencyMs: Math.round(this._avgLatencyMs * 10) / 10,
            hitRate: this._obsCount > 0 ? Math.round(this._actionCount / this._obsCount * 100) : 0,
        };
    }

    destroy() {
        if (this._reconnectTimer) clearTimeout(this._reconnectTimer);
        if (this.socket) {
            this.socket.destroy();
            this.socket = null;
        }
        this.connected = false;
    }
}
