/**
 * Survival System: Two-layer architecture
 *
 * Layer 1 - Tactical Policy:  fight / flee / hide / eat / equip / abort_task / resume_task
 * Layer 2 - Escape Micro-Policy: sprint_forward / strafe / dig_in / pillar_up / seek_cover / consume_now
 *
 * Both layers share one unified log format for training.
 * Rule-based bootstrap → later replaced by trained policies.
 */

import fs from 'fs';
import path from 'path';
import * as world from './library/world.js';

// ============================================================
// STATE CAPTURE (shared by both layers)
// ============================================================
function captureState(bot) {
    const pos = bot.entity?.position;
    const inv = world.getInventoryCounts(bot);
    const items = bot.inventory?.items() || [];

    // Hostile mob detection
    const HOSTILES = ['zombie', 'skeleton', 'spider', 'creeper', 'enderman', 'witch', 'drowned', 'husk', 'stray', 'phantom', 'pillager', 'vindicator', 'ravager', 'warden'];
    const nearbyMobs = [];
    for (const entity of Object.values(bot.entities || {})) {
        if (entity === bot.entity || !entity?.position || !pos) continue;
        const dist = pos.distanceTo(entity.position);
        const NON_HOSTILE = ['skeleton_horse', 'spider_jockey']; // passive mobs with hostile-sounding names
        if (dist < 32 && entity.name && HOSTILES.some(h => entity.name.includes(h)) && !NON_HOSTILE.includes(entity.name)) {
            nearbyMobs.push({
                type: entity.name,
                distance: Math.round(dist * 10) / 10,
                isRanged: ['skeleton', 'stray', 'pillager'].some(r => entity.name.includes(r)),
                isExplosive: entity.name.includes('creeper'),
                position: { x: entity.position.x, y: entity.position.y, z: entity.position.z },
            });
        }
    }
    nearbyMobs.sort((a, b) => a.distance - b.distance);

    // Weapon analysis
    const swords = items.filter(i => i.name?.includes('sword'));
    const tierOrder = ['netherite', 'diamond', 'iron', 'stone', 'wooden', 'golden'];
    swords.sort((a, b) => tierOrder.indexOf(a.name.split('_')[0]) - tierOrder.indexOf(b.name.split('_')[0]));
    const bestSword = swords[0]?.name || null;

    // Armor
    const armorPieces = items.filter(i =>
        i.name?.includes('helmet') || i.name?.includes('chestplate') ||
        i.name?.includes('leggings') || i.name?.includes('boots')
    );

    // Food
    const EDIBLE = ['cooked_beef','cooked_porkchop','cooked_chicken','cooked_mutton','bread','golden_apple',
        'cooked_salmon','baked_potato','cooked_rabbit','cooked_cod','apple','carrot','potato',
        'sweet_berries','rotten_flesh','mushroom_stew','dried_kelp','melon_slice','pumpkin_pie'];
    const foodItems = Object.entries(inv).filter(([k]) => EDIBLE.includes(k));
    const totalFood = foodItems.reduce((s, [_, c]) => s + c, 0);
    // Sort by nutrition value (rough)
    const foodPriority = ['golden_apple','cooked_beef','cooked_porkchop','cooked_mutton','bread','cooked_chicken','baked_potato','cooked_salmon','cooked_cod','cooked_rabbit','apple','carrot','potato','melon_slice','sweet_berries','dried_kelp','rotten_flesh'];
    const bestFood = foodItems.sort((a, b) => foodPriority.indexOf(a[0]) - foodPriority.indexOf(b[0]))[0]?.[0] || null;

    // Placeable blocks for shelter
    const placeableBlocks = Object.entries(inv).filter(([k]) =>
        k.includes('dirt') || k.includes('cobblestone') || k.includes('planks') || k.includes('stone') || k.includes('log') || k.includes('sand') || k === 'netherrack' || k === 'cobbled_deepslate'
    ).reduce((s, [_, c]) => s + c, 0);

    const time = bot.time?.timeOfDay || 0;

    return {
        // Vitals
        health: Math.round(bot.health || 0),
        hunger: Math.round(bot.food || 0),
        armorValue: bot.entity?.attributes?.['minecraft:armor']?.value || 0,

        // Equipment
        heldItem: bot.heldItem?.name || 'empty',
        bestSword,
        hasSword: !!bestSword,
        armorPieceCount: armorPieces.length,
        hasShield: items.some(i => i.name?.includes('shield')),

        // Threats
        nearbyMobs: nearbyMobs.slice(0, 5),
        mobCount: nearbyMobs.length,
        closestMobDist: nearbyMobs[0]?.distance || 999,
        closestMobType: nearbyMobs[0]?.type || null,
        closestMobIsRanged: nearbyMobs[0]?.isRanged || false,
        closestMobIsExplosive: nearbyMobs[0]?.isExplosive || false,
        closestMobPos: nearbyMobs[0]?.position || null,

        // Environment
        isNight: time > 13000 || time < 200,
        onSurface: (pos?.y || 64) > 50,
        y: Math.round(pos?.y || 0),
        position: pos ? { x: Math.round(pos.x), y: Math.round(pos.y), z: Math.round(pos.z) } : null,

        // Resources
        totalFood,
        bestFood,
        placeableBlocks,
        hasBed: items.some(i => i.name?.includes('bed')),

        // Threat awareness (for hide gating)
        // closestMobDist < 8 is a proxy for "mob can see/reach you"
        threatVisible: nearbyMobs.length > 0 && nearbyMobs[0].distance < 8,
        recentDamage: bot.lastDamageTime ? (Date.now() - bot.lastDamageTime < 5000) : false,
    };
}

// ============================================================
// TACTICAL ACTIONS (Layer 1)
// ============================================================
const TACTICAL = {
    CONTINUE: 'continue',
    FIGHT: 'fight',
    FLEE: 'flee',
    HIDE: 'hide',
    EAT: 'eat',
    EQUIP_WEAPON: 'equip_weapon',
    EQUIP_ARMOR: 'equip_armor',
    ABORT_TASK: 'abort_task',
    HUNT_FOOD: 'hunt_food',
    SLEEP: 'sleep',
};

// ============================================================
// ESCAPE MICRO-ACTIONS (Layer 2, used when tactical = FLEE or HIDE)
// ============================================================
const ESCAPE = {
    SPRINT_AWAY: 'sprint_away',
    STRAFE_LEFT: 'strafe_left',
    STRAFE_RIGHT: 'strafe_right',
    DIG_IN: 'dig_in',
    SEEK_COVER: 'seek_cover',
    CONSUME_WHILE_FLEE: 'consume_while_flee',
};

// ============================================================
// LEARNED TACTICAL POLICY (loaded from trained weights)
// Falls back to rules if no trained model available.
// ============================================================
class TacticalModel {
    constructor() {
        this.loaded = false;
        this.w1 = null; this.b1 = null;
        this.w2 = null; this.b2 = null;
        this.w3 = null; this.b3 = null;
        this.actions = null;
    }

    load(filepath) {
        try {
            if (!fs.existsSync(filepath)) return false;
            const data = JSON.parse(fs.readFileSync(filepath, 'utf8'));
            this.w1 = data.w1; this.b1 = data.b1;
            this.w2 = data.w2; this.b2 = data.b2;
            this.w3 = data.w3; this.b3 = data.b3;
            this.actions = data.actions;
            this.loaded = true;
            console.log(`[TacticalModel] Loaded policy (${data.trained_on} episodes, v${data.version})`);
            return true;
        } catch (e) {
            console.log(`[TacticalModel] No trained model found, using rules`);
            return false;
        }
    }

    predict(state) {
        if (!this.loaded) return null;
        // State to feature vector (must match train_offline.py parse_state)
        const x = [
            state.health / 20, state.hunger / 20, state.armorValue / 20,
            state.hasSword ? 1 : 0, state.hasShield ? 1 : 0,
            state.mobCount / 5, Math.min(state.closestMobDist, 32) / 32,
            state.closestMobIsRanged ? 1 : 0, state.closestMobIsExplosive ? 1 : 0,
            state.isNight ? 1 : 0, state.onSurface ? 1 : 0,
            state.totalFood / 20, Math.min(state.placeableBlocks, 64) / 64,
            state.hasBed ? 1 : 0,
            Math.min(state.hideSteps || 0, 20) / 20,
            Math.min(state.idleSteps || 0, 20) / 20,
        ];
        // Forward pass: 3-layer MLP with ReLU
        let h = this._relu(this._matmul(x, this.w1, this.b1));
        h = this._relu(this._matmul(h, this.w2, this.b2));
        const logits = this._matmul(h, this.w3, this.b3);
        const bestIdx = logits.indexOf(Math.max(...logits));
        return this.actions[bestIdx];
    }

    _matmul(x, w, b) {
        const out = new Array(b.length).fill(0);
        for (let j = 0; j < b.length; j++) {
            let sum = b[j];
            for (let i = 0; i < x.length; i++) sum += x[i] * w[i][j];
            out[j] = sum;
        }
        return out;
    }

    _relu(x) { return x.map(v => Math.max(0, v)); }
}

const _tacticalModel = new TacticalModel();

// ============================================================
// ONLINE Q-MODEL: learns in background, predicts but doesn't control
// ============================================================
const TACTICAL_ACTIONS_LIST = Object.values(TACTICAL);

class OnlineQModel {
    constructor(inputDim = 16, hiddenDim = 24, nActions = TACTICAL_ACTIONS_LIST.length) {
        this.inputDim = inputDim;
        this.hiddenDim = hiddenDim;
        this.nActions = nActions;
        this.lr = 0.001;
        this.gamma = 0.95;
        this.totalUpdates = 0;
        this.targetUpdateFreq = 50;

        // Q-network weights (small 2-layer MLP)
        this.w1 = this._xavier(inputDim, hiddenDim);
        this.b1 = new Array(hiddenDim).fill(0);
        this.w2 = this._xavier(hiddenDim, nActions);
        this.b2 = new Array(nActions).fill(0);

        // Target network (copy)
        this.tw1 = this.w1.map(r => [...r]);
        this.tb1 = [...this.b1];
        this.tw2 = this.w2.map(r => [...r]);
        this.tb2 = [...this.b2];

        // Simple replay buffer
        this.buffer = [];
        this.bufferMax = 2000;

        // Stats
        this.losses = [];
    }

    _xavier(rows, cols) {
        const scale = Math.sqrt(2 / rows);
        return Array.from({ length: rows }, () =>
            Array.from({ length: cols }, () => (Math.random() * 2 - 1) * scale)
        );
    }

    stateToVec(state) {
        return [
            state.health / 20, state.hunger / 20, (state.armorValue || 0) / 20,
            state.hasSword ? 1 : 0, state.hasShield ? 1 : 0,
            state.mobCount / 5, Math.min(state.closestMobDist, 32) / 32,
            state.closestMobIsRanged ? 1 : 0, state.closestMobIsExplosive ? 1 : 0,
            state.isNight ? 1 : 0, state.onSurface ? 1 : 0,
            state.totalFood / 20, Math.min(state.placeableBlocks || 0, 64) / 64,
            state.hasBed ? 1 : 0,
            Math.min(state.hideSteps || 0, 20) / 20,
            Math.min(state.idleSteps || 0, 20) / 20,
        ];
    }

    actionToIdx(action) {
        return TACTICAL_ACTIONS_LIST.indexOf(action);
    }

    // Forward pass: returns Q values for all actions
    forward(x, useTarget = false) {
        const w1 = useTarget ? this.tw1 : this.w1;
        const b1 = useTarget ? this.tb1 : this.b1;
        const w2 = useTarget ? this.tw2 : this.w2;
        const b2 = useTarget ? this.tb2 : this.b2;

        // Hidden layer + ReLU
        const h = new Array(this.hiddenDim);
        for (let j = 0; j < this.hiddenDim; j++) {
            let sum = b1[j];
            for (let i = 0; i < this.inputDim; i++) sum += x[i] * w1[i][j];
            h[j] = Math.max(0, sum);
        }
        // Output layer
        const q = new Array(this.nActions);
        for (let j = 0; j < this.nActions; j++) {
            let sum = b2[j];
            for (let i = 0; i < this.hiddenDim; i++) sum += h[i] * w2[i][j];
            q[j] = sum;
        }
        return { q, h };
    }

    // Predict best action (for logging/shadow, not control)
    predict(state) {
        const x = this.stateToVec(state);
        const { q } = this.forward(x);
        const bestIdx = q.indexOf(Math.max(...q));
        return TACTICAL_ACTIONS_LIST[bestIdx];
    }

    // Store transition in replay buffer
    store(state, action, reward, nextState, done) {
        this.buffer.push({
            s: this.stateToVec(state),
            a: this.actionToIdx(action),
            r: reward,
            ns: this.stateToVec(nextState),
            done: done ? 1 : 0,
        });
        if (this.buffer.length > this.bufferMax) {
            this.buffer.shift();
        }
    }

    // One step of online TD update (sample mini-batch from buffer)
    update(batchSize = 8) {
        if (this.buffer.length < batchSize) return;

        let totalLoss = 0;
        for (let b = 0; b < batchSize; b++) {
            const idx = Math.floor(Math.random() * this.buffer.length);
            const { s, a, r, ns, done } = this.buffer[idx];

            // Current Q
            const { q, h } = this.forward(s);
            const qCurrent = q[a];

            // Target Q (from target network)
            const { q: qNext } = this.forward(ns, true);
            const maxQNext = Math.max(...qNext);
            const target = r + this.gamma * maxQNext * (1 - done);

            // TD error
            const tdError = qCurrent - target;
            totalLoss += tdError * tdError;

            // Backprop: dL/dq[a] = 2 * tdError
            const dq = new Array(this.nActions).fill(0);
            dq[a] = 2 * tdError / batchSize;

            // Update w2, b2
            for (let j = 0; j < this.nActions; j++) {
                this.b2[j] -= this.lr * dq[j];
                for (let i = 0; i < this.hiddenDim; i++) {
                    this.w2[i][j] -= this.lr * dq[j] * h[i];
                }
            }

            // Backprop through ReLU to w1, b1
            for (let i = 0; i < this.hiddenDim; i++) {
                if (h[i] <= 0) continue; // ReLU gate
                let grad = 0;
                for (let j = 0; j < this.nActions; j++) grad += dq[j] * this.w2[i][j];
                this.b1[i] -= this.lr * grad;
                for (let k = 0; k < this.inputDim; k++) {
                    this.w1[k][i] -= this.lr * grad * s[k];
                }
            }
        }

        this.totalUpdates++;

        // Sync target network
        if (this.totalUpdates % this.targetUpdateFreq === 0) {
            this.tw1 = this.w1.map(r => [...r]);
            this.tb1 = [...this.b1];
            this.tw2 = this.w2.map(r => [...r]);
            this.tb2 = [...this.b2];
        }

        this.losses.push(totalLoss / batchSize);
        if (this.losses.length > 100) this.losses.shift();

        // Log every 100 updates
        if (this.totalUpdates % 100 === 0) {
            const avgLoss = this.losses.reduce((a, b) => a + b, 0) / this.losses.length;
            console.log(`[OnlineQ] updates=${this.totalUpdates} avgLoss=${avgLoss.toFixed(4)} buffer=${this.buffer.length}`);
        }
    }

    // Save weights to JSON
    save(filepath) {
        fs.writeFileSync(filepath, JSON.stringify({
            w1: this.w1, b1: this.b1, w2: this.w2, b2: this.b2,
            tw1: this.tw1, tb1: this.tb1, tw2: this.tw2, tb2: this.tb2,
            totalUpdates: this.totalUpdates,
            inputDim: this.inputDim, hiddenDim: this.hiddenDim, nActions: this.nActions,
        }));
    }

    load(filepath) {
        try {
            if (!fs.existsSync(filepath)) return false;
            const d = JSON.parse(fs.readFileSync(filepath, 'utf8'));
            const savedIn = d.inputDim || d.w1?.length || 0;
            const savedHid = d.hiddenDim || d.w1?.[0]?.length || 0;

            // Verify output dim matches
            const savedActions = d.nActions || d.w2?.[0]?.length || 0;
            if (savedActions > 0 && savedActions !== this.nActions) {
                console.log(`[OnlineQ] Action space changed (${savedActions}->${this.nActions}), fresh start`);
                return false;
            }

            if (savedIn === this.inputDim && savedHid === this.hiddenDim) {
                this.w1 = d.w1; this.b1 = d.b1; this.w2 = d.w2; this.b2 = d.b2;
                this.tw1 = d.tw1 || this.w1.map(r => [...r]);
                this.tb1 = d.tb1 || [...this.b1];
                this.tw2 = d.tw2 || this.w2.map(r => [...r]);
                this.tb2 = d.tb2 || [...this.b2];
                this.totalUpdates = d.totalUpdates || 0;
                console.log(`[OnlineQ] Loaded (${this.totalUpdates} updates)`);
            } else if (savedHid === this.hiddenDim && savedIn > 0) {
                const overlap = Math.min(savedIn, this.inputDim);
                this.w1 = this._xavier(this.inputDim, this.hiddenDim);
                for (let i = 0; i < overlap; i++) this.w1[i] = d.w1[i];
                this.b1 = d.b1; this.w2 = d.w2; this.b2 = d.b2;
                this.tw1 = this.w1.map(r => [...r]); this.tb1 = [...this.b1];
                this.tw2 = this.w2.map(r => [...r]); this.tb2 = [...this.b2];
                this.totalUpdates = Math.floor((d.totalUpdates || 0) * 0.3);
                console.log(`[OnlineQ] Dimension-adapted: ${savedIn}->${this.inputDim}, updates=${this.totalUpdates}`);
            } else {
                console.log(`[OnlineQ] Incompatible dims, fresh start`);
                return false;
            }
            return true;
        } catch (e) { return false; }
    }
}

const _onlineQ = new OnlineQModel();

// ============================================================
// ONLINE PPO (Actor-Critic): learns in background alongside Q-model
// Actor outputs action probabilities, Critic estimates V(s)
// ============================================================
class OnlinePPO {
    constructor(inputDim = 16, hiddenDim = 24, nActions = TACTICAL_ACTIONS_LIST.length) {
        this.inputDim = inputDim;
        this.hiddenDim = hiddenDim;
        this.nActions = nActions;
        this.lr = 0.0005;
        this.gamma = 0.95;
        this.clipEps = 0.2;
        this.totalUpdates = 0;

        // Actor weights
        this.aw1 = this._xavier(inputDim, hiddenDim);
        this.ab1 = new Array(hiddenDim).fill(0);
        this.aw2 = this._xavier(hiddenDim, nActions);
        this.ab2 = new Array(nActions).fill(0);

        // Critic weights
        this.cw1 = this._xavier(inputDim, hiddenDim);
        this.cb1 = new Array(hiddenDim).fill(0);
        this.cw2 = this._xavier(hiddenDim, 1);
        this.cb2 = [0];

        // Rollout buffer (collect then update)
        this.rollout = [];
        this.rolloutSize = 32;
    }

    _xavier(r, c) {
        const s = Math.sqrt(2 / r);
        return Array.from({ length: r }, () => Array.from({ length: c }, () => (Math.random() * 2 - 1) * s));
    }

    stateToVec(state) { return _onlineQ.stateToVec(state); } // reuse same feature extraction

    // Actor forward: returns action probabilities (softmax)
    actorForward(x) {
        const h = new Array(this.hiddenDim);
        for (let j = 0; j < this.hiddenDim; j++) {
            let sum = this.ab1[j];
            for (let i = 0; i < this.inputDim; i++) sum += x[i] * this.aw1[i][j];
            h[j] = Math.max(0, sum);
        }
        const logits = new Array(this.nActions);
        for (let j = 0; j < this.nActions; j++) {
            let sum = this.ab2[j];
            for (let i = 0; i < this.hiddenDim; i++) sum += h[i] * this.aw2[i][j];
            logits[j] = sum;
        }
        // Softmax
        const maxL = Math.max(...logits);
        const exps = logits.map(l => Math.exp(l - maxL));
        const sumExp = exps.reduce((a, b) => a + b, 0);
        const probs = exps.map(e => e / sumExp);
        return { probs, h, logits };
    }

    // Critic forward: returns V(s)
    criticForward(x) {
        const h = new Array(this.hiddenDim);
        for (let j = 0; j < this.hiddenDim; j++) {
            let sum = this.cb1[j];
            for (let i = 0; i < this.inputDim; i++) sum += x[i] * this.cw1[i][j];
            h[j] = Math.max(0, sum);
        }
        let v = this.cb2[0];
        for (let i = 0; i < this.hiddenDim; i++) v += h[i] * this.cw2[i][0];
        return { v, h };
    }

    predict(state) {
        const x = this.stateToVec(state);
        const { probs } = this.actorForward(x);
        const bestIdx = probs.indexOf(Math.max(...probs));
        return TACTICAL_ACTIONS_LIST[bestIdx];
    }

    store(state, action, reward, nextState, done) {
        const x = this.stateToVec(state);
        const { probs } = this.actorForward(x);
        const aIdx = TACTICAL_ACTIONS_LIST.indexOf(action);
        this.rollout.push({ s: x, a: aIdx, r: reward, ns: this.stateToVec(nextState), done: done ? 1 : 0, oldProb: probs[aIdx] });

        if (this.rollout.length >= this.rolloutSize) {
            const batch = [...this.rollout];
            this.rollout = [];
            setImmediate(() => {
                try { this._updatePPOBatch(batch); } catch (e) {}
            });
        }
    }

    _updatePPOBatch(rollout) {
        const n = rollout.length;
        if (n === 0) return;

        // Compute returns and advantages
        const returns = new Array(n);
        const advantages = new Array(n);
        let nextV = 0;
        for (let t = n - 1; t >= 0; t--) {
            const { s, r, ns, done } = rollout[t];
            const { v: vCurr } = this.criticForward(s);
            const { v: vNext } = this.criticForward(ns);
            nextV = done ? 0 : vNext;
            returns[t] = r + this.gamma * nextV;
            advantages[t] = returns[t] - vCurr;
        }

        // Normalize advantages
        const meanAdv = advantages.reduce((a, b) => a + b, 0) / n;
        const stdAdv = Math.sqrt(advantages.reduce((a, b) => a + (b - meanAdv) ** 2, 0) / n + 1e-8);
        for (let t = 0; t < n; t++) advantages[t] = (advantages[t] - meanAdv) / stdAdv;

        // PPO update (single pass)
        for (let t = 0; t < n; t++) {
            const { s, a, oldProb } = rollout[t];
            const { probs, h } = this.actorForward(s);
            const newProb = probs[a];
            const ratio = newProb / (oldProb + 1e-8);
            const clipped = Math.max(1 - this.clipEps, Math.min(1 + this.clipEps, ratio));
            const policyLoss = -Math.min(ratio * advantages[t], clipped * advantages[t]);

            // Actor gradient: dL/dlogits[a] approximation
            const dlogits = new Array(this.nActions).fill(0);
            dlogits[a] = policyLoss * (1 - newProb); // simplified policy gradient
            for (let j = 0; j < this.nActions; j++) {
                this.ab2[j] -= this.lr * dlogits[j];
                for (let i = 0; i < this.hiddenDim; i++) {
                    this.aw2[i][j] -= this.lr * dlogits[j] * h[i];
                }
            }

            // Critic update: MSE on returns
            const { v, h: ch } = this.criticForward(s);
            const vError = v - returns[t];
            this.cb2[0] -= this.lr * 2 * vError;
            for (let i = 0; i < this.hiddenDim; i++) {
                if (ch[i] <= 0) continue;
                this.cw2[i][0] -= this.lr * 2 * vError * ch[i];
            }
        }

        this.totalUpdates++;
        if (this.totalUpdates % 20 === 0) {
            console.log(`[OnlinePPO] updates=${this.totalUpdates} rollouts=${this.totalUpdates * this.rolloutSize}`);
        }
    }

    save(filepath) {
        fs.writeFileSync(filepath, JSON.stringify({
            aw1: this.aw1, ab1: this.ab1, aw2: this.aw2, ab2: this.ab2,
            cw1: this.cw1, cb1: this.cb1, cw2: this.cw2, cb2: this.cb2,
            totalUpdates: this.totalUpdates,
        }));
    }

    load(filepath) {
        try {
            if (!fs.existsSync(filepath)) return false;
            const d = JSON.parse(fs.readFileSync(filepath, 'utf8'));
            this.aw1 = d.aw1; this.ab1 = d.ab1; this.aw2 = d.aw2; this.ab2 = d.ab2;
            this.cw1 = d.cw1; this.cb1 = d.cb1; this.cw2 = d.cw2; this.cb2 = d.cb2;
            this.totalUpdates = d.totalUpdates || 0;
            console.log(`[OnlinePPO] Loaded (${this.totalUpdates} updates)`);
            return true;
        } catch (e) { return false; }
    }
}

const _onlinePPO = new OnlinePPO();

// ============================================================
// TACTICAL POLICY: tries learned model first, falls back to rules
// ============================================================
// ============================================================
// COMBAT ASSESSMENT — hardcoded MC damage model, no RL needed
// ============================================================
const WEAPON_DPS = {
    'netherite_sword': 12.8, 'diamond_sword': 11.2, 'iron_sword': 9.6,
    'stone_sword': 8.0, 'wooden_sword': 6.4, 'golden_sword': 6.4,
    'netherite_axe': 10.0, 'diamond_axe': 9.0, 'iron_axe': 9.0,
    'stone_axe': 7.0, 'iron_pickaxe': 3.0, 'stone_pickaxe': 2.5,
    'fist': 2.0,
};
const MOB_STATS = {
    'zombie': { hp: 20, dps: 3.0 }, 'husk': { hp: 20, dps: 3.0 },
    'skeleton': { hp: 20, dps: 4.0, ranged: true }, 'stray': { hp: 20, dps: 4.0, ranged: true },
    'spider': { hp: 16, dps: 2.0 }, 'cave_spider': { hp: 12, dps: 2.0 },
    'creeper': { hp: 20, dps: 32, explosive: true },
    'drowned': { hp: 20, dps: 3.5 }, 'phantom': { hp: 20, dps: 2.5 },
    'enderman': { hp: 40, dps: 7.0 }, 'witch': { hp: 26, dps: 6.0, ranged: true },
    'pillager': { hp: 24, dps: 4.0, ranged: true }, 'vindicator': { hp: 24, dps: 6.0 },
    'ravager': { hp: 100, dps: 12.0 }, 'warden': { hp: 500, dps: 30.0 },
};

function assessCombat(state) {
    if (state.mobCount === 0 || state.closestMobDist > 24) {
        return { canWin: true, confidence: 1.0, expectedHpLoss: 0, threatLevel: 'none' };
    }

    // My DPS: check held item first, then best sword in inventory
    const heldDps = WEAPON_DPS[state.heldItem] || 0;
    const swordDps = WEAPON_DPS[state.bestSword] || 0;
    const myDps = Math.max(heldDps, swordDps, WEAPON_DPS['fist']);

    // Armor damage reduction: each armor point = 4% reduction
    const armorReduction = Math.min(0.8, (state.armorValue || 0) * 0.04);

    // Sum threat from nearby hostile mobs (cap at 4 for calculation)
    // Separate melee and ranged DPS — dodge factor only applies to melee.
    let meleeThreatDps = 0, rangedThreatDps = 0;
    let totalThreatHp = 0;
    let closestExplosiveDist = 99;
    const mobsToConsider = Math.min(state.mobCount, 4);
    for (let i = 0; i < mobsToConsider; i++) {
        const mob = state.nearbyMobs?.[i];
        if (!mob) continue;
        const stats = MOB_STATS[mob.type] || { hp: 20, dps: 3.0 };
        if (stats.explosive) {
            closestExplosiveDist = Math.min(closestExplosiveDist, mob.distance || 99);
            continue;
        }
        // Closer mobs contribute more (they'll hit you sooner)
        const distFactor = mob.distance < 6 ? 1.0 : (mob.distance < 12 ? 0.5 : 0.2);
        if (stats.ranged) rangedThreatDps += stats.dps * distFactor;
        else meleeThreatDps += stats.dps * distFactor;
        totalThreatHp += (mob.hp || stats.hp);
    }

    // Creeper = always flee, but only if the CREEPER ITSELF is close (<8 blocks)
    // Old bug: used closestMobDist which could be a zombie → creeper at 20 blocks triggered lethal
    if (closestExplosiveDist < 8) {
        return { canWin: false, confidence: 0.95, expectedHpLoss: 999, threatLevel: 'lethal' };
    }

    if (totalThreatHp === 0) {
        return { canWin: true, confidence: 1.0, expectedHpLoss: 0, threatLevel: 'none' };
    }

    // Dodge factor: sprint-knockback lets a skilled player avoid much melee damage.
    // Ranged mobs (skeleton, pillager) are NOT dodgeable — arrows track the player.
    const meleeCount = (state.nearbyMobs || []).filter(m => {
        const s = MOB_STATS[m.type]; return s && !s.explosive && !s.ranged;
    }).length;
    const dodgeFactor = meleeCount <= 1 ? 0.35 : (meleeCount <= 2 ? 0.6 : 0.85);
    const effectiveThreatDps = meleeThreatDps * (1 - armorReduction) * dodgeFactor
                             + rangedThreatDps * (1 - armorReduction); // ranged: no dodge
    // Cap timeToKill: bot won't stand and fight for 20+ seconds.
    // Sprint-knockback kiting means actual engagement is shorter.
    const rawTimeToKill = totalThreatHp / myDps;
    const timeToKill = Math.min(rawTimeToKill, 15);
    const expectedHpLoss = effectiveThreatDps * timeToKill;
    const safetyMargin = state.health - expectedHpLoss;

    let threatLevel, canWin;
    if (safetyMargin > 12) { threatLevel = 'trivial'; canWin = true; }
    else if (safetyMargin > 4) { threatLevel = 'manageable'; canWin = true; }
    else if (safetyMargin > -4) { threatLevel = 'dangerous'; canWin = false; }
    else { threatLevel = 'lethal'; canWin = false; }

    // Courage override: full HP + single melee mob (not ranged/explosive) → promote to manageable
    // A 20hp player CAN kill a single zombie with fists even if the math says "dangerous"
    const singleMeleeMob = state.mobCount === 1 && !state.closestMobIsRanged && !state.closestMobIsExplosive;
    if (threatLevel === 'dangerous' && state.health >= 18 && singleMeleeMob) {
        threatLevel = 'manageable';
        canWin = true;
    }
    // Lethal → dangerous override for single melee mobs at full HP.
    // Bot has no weapon → hpLoss is huge (50-100) → always lethal → always flee → never fights.
    // But counterattack PROVED bot can hit (spider hp 13→11). With attackEntity fix,
    // bot can damage mobs. A full-HP player CAN survive a single zombie/spider fight.
    // Remove safetyMargin gate — let brave fight (line ~780) handle the decision.
    if (threatLevel === 'lethal' && state.health >= 18 && singleMeleeMob) {
        threatLevel = 'dangerous';
    }

    const confidence = Math.min(1, Math.max(0, Math.abs(safetyMargin) / 20));

    return { canWin, confidence, expectedHpLoss: Math.round(expectedHpLoss * 10) / 10, threatLevel };
}

function tacticalDecide(state) {
    if (_tacticalModel.loaded) {
        const modelAction = _tacticalModel.predict(state);
        const actionMap = {
            'fight': TACTICAL.FIGHT, 'flee': TACTICAL.FLEE, 'hide': TACTICAL.HIDE,
            'eat': TACTICAL.EAT, 'equip_weapon': TACTICAL.EQUIP_WEAPON,
            'equip_armor': TACTICAL.EQUIP_ARMOR, 'abort_task': TACTICAL.ABORT_TASK,
            'hunt_food': TACTICAL.HUNT_FOOD, 'sleep': TACTICAL.SLEEP, 'continue': TACTICAL.CONTINUE,
        };
        return { action: actionMap[modelAction] || TACTICAL.CONTINUE, reason: `[RL] ${modelAction}` };
    }

    const combat = assessCombat(state);

    // === CRITICAL: flee if dying + mobs very close ===
    if (state.health <= 4 && state.mobCount > 0 && state.closestMobDist < 16) {
        return { action: TACTICAL.FLEE, reason: `dying hp=${state.health}` };
    }

    // === CRITICAL: eat if about to die (only if safe enough to eat) ===
    if (state.health <= 6 && state.totalFood > 0 && state.closestMobDist > 4) {
        return { action: TACTICAL.EAT, reason: `critical hp=${state.health}` };
    }

    // === CREEPER: always flee (hardcoded, RL cannot override) ===
    if (state.closestMobIsExplosive && state.closestMobDist < 8) {
        return { action: TACTICAL.FLEE, reason: `creeper@${state.closestMobDist}!` };
    }

    // === EAT: hungry and safe ===
    if (state.hunger <= 14 && state.totalFood > 0 && state.closestMobDist > 8) {
        return { action: TACTICAL.EAT, reason: `hunger=${state.hunger}` };
    }

    // === HUNT: starving, no food ===
    if (state.hunger <= 10 && state.totalFood === 0 && state.hasSword) {
        return { action: TACTICAL.HUNT_FOOD, reason: `starving, hunting` };
    }

    // === EQUIP: mob approaching, have sword but not holding ===
    if (state.mobCount > 0 && state.closestMobDist < 10 && state.hasSword && !state.heldItem?.includes('sword')) {
        return { action: TACTICAL.EQUIP_WEAPON, reason: `equipping ${state.bestSword}` };
    }

    // === COMBAT DECISION based on assessCombat ===
    if (state.mobCount > 0 && state.closestMobDist < 16) {
        if (combat.threatLevel === 'none') {
            // No real threat, continue task
            return { action: TACTICAL.CONTINUE, reason: `${combat.threatLevel} threat, ignore` };
        }
        if (combat.threatLevel === 'trivial') {
            // Easy fight — only engage if mob is close enough to be worth it
            if (state.closestMobDist < 8) {
                return { action: TACTICAL.FIGHT, reason: `trivial ${state.closestMobType}@${state.closestMobDist} (hpLoss~${combat.expectedHpLoss})` };
            }
            return { action: TACTICAL.CONTINUE, reason: `trivial but far@${state.closestMobDist}, ignore` };
        }
        if (combat.threatLevel === 'manageable') {
            // Can win but will take some hits — fight if close, prepare if far
            if (state.closestMobDist < 10) {
                return { action: TACTICAL.FIGHT, reason: `manageable ${state.closestMobType}@${state.closestMobDist} (hpLoss~${combat.expectedHpLoss})` };
            }
            return { action: TACTICAL.CONTINUE, reason: `manageable but far, continue` };
        }
        if (combat.threatLevel === 'dangerous') {
            // Risky — but if full HP + single melee mob, try hit_and_run instead of fleeing
            if (state.closestMobDist < 10) {
                if (state.health >= 16 && state.mobCount === 1 && !state.closestMobIsRanged && !state.closestMobIsExplosive) {
                    return { action: TACTICAL.FIGHT, reason: `dangerous-but-brave ${state.closestMobType}@${state.closestMobDist} (hpLoss~${combat.expectedHpLoss}) hp=${state.health}` };
                }
                return { action: TACTICAL.FLEE, reason: `dangerous ${state.closestMobType}@${state.closestMobDist} (hpLoss~${combat.expectedHpLoss})` };
            }
            return { action: TACTICAL.CONTINUE, reason: `dangerous but far@${state.closestMobDist}` };
        }
        // lethal: hide (dig shelter) when HP is low — flee doesn't work in badlands (19% gaining).
        // Bot died 6/8 times in flee mode at low HP. Hide = digDown = survive.
        if (state.health <= 8) {
            return { action: TACTICAL.HIDE, reason: `lethal+lowHP hide ${state.closestMobType}@${state.closestMobDist} hp=${state.health}` };
        }
        return { action: TACTICAL.FLEE, reason: `lethal ${state.closestMobType}@${state.closestMobDist} (hpLoss~${combat.expectedHpLoss})` };
    }

    // === No combat — utility actions ===
    if (state.health < 14 && state.totalFood > 0 && state.closestMobDist > 16) {
        return { action: TACTICAL.EAT, reason: `safe to eat, hp=${state.health}` };
    }
    if (state.isNight && state.hasBed && state.closestMobDist > 8) {
        return { action: TACTICAL.SLEEP, reason: 'night, sleeping' };
    }

    return { action: TACTICAL.CONTINUE, reason: 'safe' };
}

// ============================================================
// RULE-BASED ESCAPE MICRO-POLICY (bootstrap, replaceable by RL)
// ============================================================
function escapeDecide(state, tacticalAction) {
    if (tacticalAction === TACTICAL.HIDE) {
        if (state.placeableBlocks >= 3) {
            return { micro: ESCAPE.DIG_IN, reason: 'have blocks, digging shelter' };
        }
        return { micro: ESCAPE.SPRINT_AWAY, reason: 'no blocks to hide, sprinting away' };
    }

    if (tacticalAction === TACTICAL.FLEE) {
        // Creeper: just sprint, don't stop
        if (state.closestMobIsExplosive) {
            return { micro: ESCAPE.SPRINT_AWAY, reason: 'creeper — sprint!' };
        }
        // Skeleton: strafe to dodge arrows
        if (state.closestMobIsRanged && state.closestMobDist < 16) {
            return { micro: Math.random() > 0.5 ? ESCAPE.STRAFE_LEFT : ESCAPE.STRAFE_RIGHT, reason: 'ranged mob — strafing' };
        }
        // Low health + food: eat while running
        if (state.health <= 6 && state.totalFood > 0) {
            return { micro: ESCAPE.CONSUME_WHILE_FLEE, reason: 'eating while fleeing' };
        }
        // General flee
        if (state.placeableBlocks >= 5 && state.closestMobDist < 5) {
            return { micro: ESCAPE.DIG_IN, reason: 'mob too close, digging in' };
        }
        return { micro: ESCAPE.SPRINT_AWAY, reason: 'sprinting away' };
    }

    return { micro: null, reason: 'no escape needed' };
}

// ============================================================
// REWARD FUNCTION
// ============================================================
function computeReward(prevState, action, micro, newState) {
    let reward = 0;

    // Tick survival bonus
    reward += 0.01;

    // Health delta
    const hpDelta = newState.health - prevState.health;
    if (hpDelta > 0) reward += 0.1 * hpDelta;
    if (hpDelta < 0) reward -= 0.2 * Math.abs(hpDelta);

    // Death
    if (newState.health <= 0) reward -= 5.0;

    // Ate food
    if (action === TACTICAL.EAT && newState.hunger > prevState.hunger) reward += 0.5;

    // Equipped weapon
    if (action === TACTICAL.EQUIP_WEAPON && newState.heldItem?.includes('sword')) reward += 0.3;

    // Fled successfully
    if (action === TACTICAL.FLEE && newState.closestMobDist > prevState.closestMobDist) reward += 0.4;

    // Hid successfully (mob count dropped or distance increased)
    if (action === TACTICAL.HIDE && (newState.mobCount < prevState.mobCount || newState.closestMobDist > prevState.closestMobDist + 5)) reward += 0.6;

    // Killed mob
    if (action === TACTICAL.FIGHT && newState.mobCount < prevState.mobCount) reward += 1.0;

    // Took heavy damage fighting (should have fled)
    if (action === TACTICAL.FIGHT && hpDelta < -6) reward -= 0.5;

    // Had food, low health, didn't eat
    if (action === TACTICAL.CONTINUE && prevState.health < 8 && prevState.totalFood > 0) reward -= 0.3;

    // Creeper: didn't flee
    if (prevState.closestMobIsExplosive && prevState.closestMobDist < 6 && action !== TACTICAL.FLEE) reward -= 0.8;

    // === PREPARATION REWARDS ===
    if (!newState.onSurface && newState.totalFood < 4) reward -= 0.4;
    if (action === TACTICAL.HUNT_FOOD && newState.totalFood > prevState.totalFood) reward += 0.6;
    if (!newState.onSurface && newState.totalFood >= 8) reward += 0.1;

    // === ANTI-PASSIVITY REWARDS ===
    const hs = prevState.hideSteps || 0;
    const safe = prevState.mobCount === 0 && prevState.health >= 14 && !prevState.isNight;

    if (action === TACTICAL.HIDE) {
        if (hs <= 3) {
            reward += 0.05; // short hide is ok
        } else if (hs <= 8) {
            reward -= 0.03 * (hs - 3); // mild penalty ramp
        } else {
            reward -= 0.08 * (hs - 8); // strong penalty
        }
        // Safe but still hiding = bad
        if (safe) reward -= 0.3;
    }

    // Reward hide -> productive action transition
    if (hs > 0 && action !== TACTICAL.HIDE) {
        if (action === TACTICAL.EAT) reward += 0.3;
        else if (action === TACTICAL.FLEE) reward += 0.2;
        else if (action === TACTICAL.FIGHT) reward += 0.2;
        else if (action === TACTICAL.CONTINUE && safe) reward += 0.2;
        else if (action === TACTICAL.HUNT_FOOD) reward += 0.3;
    }

    // Continue should be judged by short-term consequence, not just by the snapshot.
    if (action === TACTICAL.CONTINUE) {
        const threatened = prevState.mobCount > 0 && prevState.closestMobDist < 10;
        const tookHitSoon = hpDelta < 0;
        const distDelta = (newState.closestMobDist || 999) - (prevState.closestMobDist || 999);

        if (threatened) {
            // Continue was a bad idea if we got tagged shortly after choosing it.
            if (tookHitSoon) reward -= 0.8;
            // Continue is acceptable if we stayed unhit through the next window.
            else reward += 0.25;
            // If "continue" still improved spacing, reward that outcome slightly.
            if (distDelta > 1.5) reward += 0.15;
            if (distDelta < -1.5) reward -= 0.15;
        }

        if (prevState.hunger <= 12 && prevState.totalFood > 0) reward -= 0.3;
        const is = prevState.idleSteps || 0;
        if (is > 5) reward -= 0.1 * (is - 5); // stuck doing nothing
    }

    // Reward sleeping (prevents phantoms)
    if (action === TACTICAL.SLEEP) reward += 0.5;

    return Math.round(reward * 100) / 100;
}

// ============================================================
// SURVIVAL POLICY CLASS
// ============================================================
export class SurvivalPolicy {
    constructor(agent, botDir) {
        this.agent = agent;
        this.bot = agent.bot;
        this.logFile = path.join(botDir, 'survival_episodes.jsonl');
        this.lastState = null;
        this.lastTactical = null;
        this.lastMicro = null;
        this.lastActionTime = 0;
        this.cooldown = 2000;
        this.episodeId = 0;
        this.sheltering = false;
        this.hideSteps = 0;       // how many consecutive ticks in hide
        this.idleSteps = 0;       // how many consecutive ticks in continue with no progress
        this.lastPosition = null;
        this.lastDeathLogAt = 0;
        this.hideBuildActive = false;
        this.lastHideBuildAt = 0;
        this.pillarBuildActive = false;
        this.lastPillarBuildAt = 0;
        this._fleeUntil = 0; // post-flee buffer: keep flee mode for N ms after tactical says "safe"

        // Load trained tactical model (if available)
        _tacticalModel.load(path.join(botDir, 'tactical_policy.json'));

        // Load/init online models (shadow learners)
        this._qModelPath = path.join(botDir, 'online_q_weights.json');
        this._ppoModelPath = path.join(botDir, 'online_ppo_weights.json');
        _onlineQ.load(this._qModelPath);
        // PPO disabled (off-policy data invalid for PPO ratio)
        // Auto-save Q weights every 5 minutes
        this._modelSaveInterval = setInterval(() => {
            try { _onlineQ.save(this._qModelPath); } catch (e) {}
        }, 300000);

        // Load episode count from existing log
        try {
            if (fs.existsSync(this.logFile)) {
                const lines = fs.readFileSync(this.logFile, 'utf8').trim().split('\n');
                this.episodeId = lines.length;
            }
        } catch (e) {}
    }

    /**
     * Fast lightweight state check — no logging, no decisions, just state.
     * Used by the 1.5s reactive timer in agent.js.
     */
    _quickCheck() {
        this.bot = this.agent.bot;
        if (!this.bot?.entity) return null;
        return captureState(this.bot);
    }

    assessCombat(state) {
        return assessCombat(state);
    }

    /**
     * Fast Q gate for emergency survival.
     * Only allows Q to choose between immediate reflex actions: eat or flee.
     * Returns null when Q is not ready or not confident enough.
     */
    getFastQDecision(state) {
        if (!state || _onlineQ.totalUpdates < 50) return null;
        const x = _onlineQ.stateToVec(state);
        const { q } = _onlineQ.forward(x);
        const sorted = [...q].sort((a, b) => b - a);
        const confidence = sorted[0] - sorted[1];
        const bestIdx = q.indexOf(sorted[0]);
        const bestAction = TACTICAL_ACTIONS_LIST[bestIdx];
        if (confidence < 0.2) return null;
        if (bestAction !== TACTICAL.EAT && bestAction !== TACTICAL.FLEE) return null;
        return { action: bestAction, confidence };
    }

    /**
     * Called by planner loop. Returns {action, micro, reason} if survival needs to act.
     */
    /**
     * Q-model gated tick: Q predicts, rules fallback.
     * Q can override rules for: eat, flee, continue.
     * Rules still control: hide, fight, sleep, hunt_food (until Q is more trusted).
     */
    /**
     * Main entry: Q-model gated + rules fallback.
     * Two-phase: (1) produce decision, (2) commit to lastState/lastTactical.
     * This ensures what we log = what we actually execute.
     */
    async tickWithQ() {
        const now = Date.now();
        this.bot = this.agent.bot;
        if (!this.bot?.entity) return null;
        const state = captureState(this.bot);
        state.hideSteps = this.hideSteps;
        state.idleSteps = this.idleSteps;

        // === PHASE 0: Log PREVIOUS transition (uses last committed state/action) ===
        if (this.lastState && this.lastTactical) {
            const reward = computeReward(this.lastState, this.lastTactical, this.lastMicro, state);
            this._log(this.lastState, this.lastTactical, this.lastMicro, reward, state);
            const done = state.health <= 0;
            _onlineQ.store(this.lastState, this.lastTactical, reward, state, done);
            if (!this._qUpdating) {
                this._qUpdating = true;
                setImmediate(() => {
                    try { _onlineQ.update(4); } catch (e) {}
                    this._qUpdating = false;
                });
            }
            if (_onlineQ.totalUpdates % 50 === 0 && _onlineQ.totalUpdates > 0) {
                console.log(`[Shadow] Q=${_onlineQ.predict(this.lastState)} executed=${this.lastTactical}`);
            }
        }

        // === Shelter state ===
        if (this.sheltering) {
            const safe = !state.isNight && state.mobCount === 0 && state.health >= 10;
            const timedOut = this.hideSteps > 30;
            if (safe || timedOut) {
                this.sheltering = false;
                this.hideSteps = 0;
            } else {
                this._commit(state, TACTICAL.HIDE, null, now);
                return null;
            }
        }

        if (now - this.lastActionTime < this.cooldown) {
            this.lastState = state;
            return null;
        }

        // === PHASE 1: Rules produce candidate ===
        const ruleDecision = tacticalDecide(state);
        const escape = escapeDecide(state, ruleDecision.action);
        let finalAction = ruleDecision.action;
        let finalMicro = escape.micro;
        let finalReason = ruleDecision.reason;

        // === PHASE 1.5: Post-flee buffer ===
        // When we were just fleeing, don't drop to CONTINUE instantly.
        // Keep flee mode for 4s after last flee decision if any mob is within 16 blocks.
        // This prevents the bot from turning around and walking back toward danger.
        if (finalAction === TACTICAL.FLEE) {
            this._fleeUntil = now + 4000;
            // Signal scheduler so override timeout respects this buffer
            if (this.agent.scheduler) this.agent.scheduler.postFleeUntil = now + 4000;
        }
        if (finalAction === TACTICAL.CONTINUE && now < this._fleeUntil && state.mobCount > 0 && state.closestMobDist < 16) {
            finalAction = TACTICAL.FLEE;
            finalMicro = escape.micro || null;
            finalReason = `post-flee buffer (mob ${state.closestMobType}@${state.closestMobDist})`;
        }

        // === PHASE 2: Q-model may override (conservative thresholds) ===
        const ALLOWED_Q = ['continue', 'eat', 'flee'];
        const MIN_Q_UPDATES = 500;
        const MIN_Q_CONFIDENCE = 0.3;
        if (_onlineQ.totalUpdates >= MIN_Q_UPDATES) {
            const x = _onlineQ.stateToVec(state);
            const { q } = _onlineQ.forward(x);
            const bestIdx = q.indexOf(Math.max(...q));
            const qAction = TACTICAL_ACTIONS_LIST[bestIdx];
            const sorted = [...q].sort((a, b) => b - a);
            const confidence = sorted[0] - sorted[1];
            const qFleeAllowed = state.mobCount > 0 && state.closestMobDist < 12;
            const qEatAllowed = state.totalFood > 0 && !!state.bestFood;
            const qActionAllowed =
                qAction !== 'flee' || qFleeAllowed
                    ? (qAction !== 'eat' || qEatAllowed)
                    : false;

            // Health veto: Q cannot override rule-flee with "continue" when low HP
            const healthVeto = state.health <= 8 && finalAction === 'flee' && qAction === 'continue';
            // Post-flee veto: Q cannot revert flee back to continue during post-flee buffer
            const postFleeVeto = now < this._fleeUntil && finalAction === 'flee' && qAction === 'continue';

            if (confidence > MIN_Q_CONFIDENCE && ALLOWED_Q.includes(qAction) && qActionAllowed && qAction !== finalAction && !healthVeto && !postFleeVeto) {
                console.log(`[Q-Override] Q=${qAction}(conf=${confidence.toFixed(2)}) over rule=${finalAction}`);
                finalAction = qAction;
                finalMicro = null;
                finalReason = `[Q conf=${confidence.toFixed(2)}] ${qAction}`;
            }
        }

        // === PHASE 3: Track counters based on FINAL action ===
        if (finalAction === TACTICAL.HIDE) {
            this.hideSteps++;
            this.idleSteps = 0;
        } else if (finalAction === TACTICAL.CONTINUE) {
            this.hideSteps = 0;
            const pos = this.bot.entity?.position;
            const moved = this.lastPosition && pos ?
                Math.abs(pos.x - this.lastPosition.x) + Math.abs(pos.z - this.lastPosition.z) > 1 : true;
            this.lastPosition = pos ? { x: pos.x, z: pos.z } : null;
            if (!moved) this.idleSteps++;
            else this.idleSteps = 0;
        } else {
            this.hideSteps = 0;
            this.idleSteps = 0;
        }

        // === PHASE 4: Commit final decision (what we log next tick = what we execute now) ===
        this._commit(state, finalAction, finalMicro, now);

        // === PHASE 5: Publish mode + declare/resolve override ===
        if (this.agent.scheduler) {
            this.agent.scheduler.setMode(finalAction, 'tactical');

            const OVERRIDE_MODES = ['flee', 'fight', 'hide', 'eat', 'sleep'];
            if (OVERRIDE_MODES.includes(finalAction)) {
                if (!this.agent.scheduler.activeOverride || this.agent.scheduler.activeOverride.mode !== finalAction) {
                    this.agent.scheduler.declareOverride(finalAction, finalReason, 15000, 'return_anchor', 'tactical');
                }
            } else if (
                finalAction === TACTICAL.CONTINUE &&
                this.agent.scheduler.activeOverride &&
                this.agent.scheduler.activeOverride.source !== 'reflex'
            ) {
                this.agent.scheduler.resolveOverride('safe');
            }
            this.agent.scheduler.checkOverrideTimeout();
        }

        if (finalAction === TACTICAL.CONTINUE) return null;

        this.lastActionTime = now;
        return {
            action: finalAction,
            micro: finalMicro,
            reason: finalReason + (finalMicro ? ` [${finalMicro}]` : ''),
        };
    }

    _commit(state, action, micro, now) {
        this.lastState = state;
        this.lastTactical = action;
        this.lastMicro = micro;
        if (now) this.lastActionTime = now;
    }

    /**
     * Execute the survival decision.
     */
    async execute(decision) {
        this.bot = this.agent.bot;
        const bot = this.bot;
        if (!bot?.entity) return;
        let skills;
        try { skills = await import('./library/skills.js'); } catch (e) { console.error('[SurvivalPolicy] skills import failed:', e.message); return; }

        switch (decision.action) {
            case TACTICAL.EAT: {
                const state = captureState(bot);
                if (state.bestFood) {
                    try {
                        await skills.consume(bot, state.bestFood);
                        console.log(`[Survival] Ate ${state.bestFood}`);
                    } catch (e) {}
                }
                break;
            }
            case TACTICAL.EQUIP_WEAPON: {
                const state = captureState(bot);
                if (state.bestSword) {
                    try {
                        const sword = bot.inventory.items().find(i => i.name === state.bestSword);
                        if (sword) await bot.equip(sword, 'hand');
                        console.log(`[Survival] Equipped ${state.bestSword}`);
                    } catch (e) {}
                }
                break;
            }
            case TACTICAL.FLEE: {
                console.log(`[Survival] FLEE execute entered`);
                const state = captureState(bot);
                const now = Date.now();
                // Track consecutive pillar failures — disable after 3 consecutive fails
                if (!this._pillarConsecutiveFails) this._pillarConsecutiveFails = 0;
                const shouldTryPillar =
                    !!this.agent.skillRegistry &&
                    !!this.agent.scheduler &&
                    !!this.agent.microCtrl &&
                    !bot.entity?.isInWater &&
                    !state.closestMobIsExplosive &&
                    state.mobCount > 0 &&
                    state.placeableBlocks >= 1 &&
                    state.closestMobDist < 10 &&
                    !this.pillarBuildActive &&
                    this._pillarConsecutiveFails < 3 &&
                    now - this.lastPillarBuildAt >= 8000;

                // Debug: only log every 10th call to avoid spam
                this._pillarCheckCount = (this._pillarCheckCount || 0) + 1;
                if (this._pillarCheckCount % 10 === 0) {
                    console.log(`[Survival] pillar check: mobs=${state.mobCount} blocks=${state.placeableBlocks} dist=${state.closestMobDist?.toFixed(1)} → ${shouldTryPillar}`);
                }
                if (shouldTryPillar) {
                    this.pillarBuildActive = true;
                    console.log(`[Survival] Flee: running registered skill pillar_up vs ${state.closestMobType}@${state.closestMobDist}`);
                    try {
                        const result = await this.agent.skillRegistry.run('pillar_up', {
                            source: 'tactical_flee',
                            reason: decision.reason || 'flee_pillar',
                            startedAt: now,
                        });
                        if (!result?.ok) {
                            this._pillarConsecutiveFails = (this._pillarConsecutiveFails || 0) + 1;
                            console.log(`[Survival] pillar_up skill failed: ${result?.reason || 'unknown'} (consecutiveFails=${this._pillarConsecutiveFails})`);
                            if (this._pillarConsecutiveFails >= 3) {
                                console.log(`[Survival] pillar_up disabled after ${this._pillarConsecutiveFails} consecutive failures`);
                            }
                        } else {
                            this._pillarConsecutiveFails = 0; // reset on success
                        }
                    } catch (e) {
                        this._pillarConsecutiveFails = (this._pillarConsecutiveFails || 0) + 1;
                        console.log(`[Survival] pillar_up error: ${e.message} (consecutiveFails=${this._pillarConsecutiveFails})`);
                    } finally {
                        this.pillarBuildActive = false;
                        this.lastPillarBuildAt = Date.now();
                    }
                    break;
                }

                // In RT scheduler mode, flee is handled continuously by microCtrl.
                if (this.agent.scheduler && this.agent.microCtrl) {
                    console.log('[Survival] Flee delegated to scheduler + microCtrl');
                } else {
                    await this._executeFlee(decision.micro, skills);
                }
                break;
            }
            case TACTICAL.HIDE: {
                // Hide positioning (movement to safe spot) → micro handles via scheduler
                // Hide building (dig + seal) → discrete skill, quick execute
                if (this.agent.scheduler && this.agent.microCtrl) {
                    const now = Date.now();
                    // Give micro a short window to position first.
                    if (this.hideSteps < 3) {
                        console.log('[Survival] Hide: micro positioning before dig+seal');
                        break;
                    }
                    // Debounce repeated hide-build attempts while tactical stays in hide.
                    if (this.hideBuildActive || now - this.lastHideBuildAt < 4000) {
                        break;
                    }
                    this.hideBuildActive = true;
                    console.log('[Survival] Hide: running registered skill dig_shelter');
                    try {
                        if (this.agent.skillRegistry) {
                            const result = await this.agent.skillRegistry.run('dig_shelter', {
                                source: 'tactical_hide',
                                reason: decision.reason || 'hide',
                                startedAt: now,
                            });
                            if (!result?.ok) {
                                console.log(`[Survival] dig_shelter skill failed: ${result?.reason || 'unknown'}`);
                            }
                        } else {
                            this.agent.scheduler.setConstraints({
                                freezeMicro: true,
                                freezeMicroUntil: now + 3500,
                            });
                            this.agent.scheduler.clearProposal?.('micro');
                            try {
                                await bot._bridge?.sendCommand('rl_action', {
                                    forward: false, back: false, left: false, right: false,
                                    jump: false, sprint: false, attack: false,
                                    dyaw: 0, dpitch: 0,
                                });
                            } catch (e) {}
                            await Promise.race([
                                skills.digDown(bot, 3),
                                new Promise((_, r) => setTimeout(() => r(), 3000)),
                            ]);
                            const pos = bot.entity?.position;
                            const sealBlock = bot.inventory.items().find(i =>
                                i.name.includes('dirt') || i.name.includes('cobblestone') || i.name.includes('planks'));
                            if (sealBlock && pos) {
                                await skills.placeBlock(bot, sealBlock.name, Math.floor(pos.x), Math.floor(pos.y) + 2, Math.floor(pos.z));
                            }
                        }
                    } catch (e) {}
                    finally {
                        this.hideBuildActive = false;
                        this.lastHideBuildAt = Date.now();
                    }
                } else {
                    await this._executeHide(decision.micro, skills);
                }
                break;
            }
            case TACTICAL.FIGHT: {
                // Equip weapon first
                const state = captureState(bot);
                if (state.bestSword) {
                    try {
                        const sword = bot.inventory.items().find(i => i.name === state.bestSword);
                        if (sword) await bot.equip(sword, 'hand');
                    } catch (e) {}
                }
                // Try hit_and_run skill if available (better than raw attackEntity)
                const now_fight = Date.now();
                const shouldHitAndRun =
                    !!this.agent.skillRegistry &&
                    !!this.agent.scheduler &&
                    !!this.agent.microCtrl &&
                    state.mobCount > 0 &&
                    state.closestMobDist < 12 &&
                    !this._hitAndRunActive &&
                    now_fight - (this._lastHitAndRunAt || 0) >= 5000;
                if (shouldHitAndRun) {
                    this._hitAndRunActive = true;
                    console.log(`[Survival] Fight: running hit_and_run vs ${state.closestMobType}@${state.closestMobDist}`);
                    try {
                        const result = await this.agent.skillRegistry.run('hit_and_run', {
                            source: 'tactical_fight',
                            reason: decision.reason || 'fight',
                            startedAt: now_fight,
                        });
                        if (!result?.ok) {
                            console.log(`[Survival] hit_and_run failed: ${result?.reason || 'unknown'}`);
                        }
                    } catch (e) {
                        console.log(`[Survival] hit_and_run error: ${e.message}`);
                    } finally {
                        this._hitAndRunActive = false;
                        this._lastHitAndRunAt = Date.now();
                    }
                    break;
                }
                // Fallback: RT scheduler mode, fight handled by microCtrl
                if (this.agent.scheduler && this.agent.microCtrl) {
                    console.log('[Survival] Fight delegated to scheduler + microCtrl');
                } else if (state.closestMobType) {
                    try {
                        const target = Object.values(bot.entities || {}).find(e =>
                            e.name?.includes(state.closestMobType) && e !== bot.entity
                        );
                        if (target) {
                            await skills.attackEntity(bot, target, false);
                            console.log(`[Survival] Fighting ${state.closestMobType}`);
                        }
                    } catch (e) {}
                }
                break;
            }
            case TACTICAL.SLEEP: {
                // Place bed and sleep
                try {
                    const bedItem = bot.inventory.items().find(i => i.name?.includes('bed'));
                    if (bedItem) {
                        // Place bed nearby
                        await skills.placeBlock(bot, bedItem.name,
                            Math.floor(bot.entity.position.x) + 1,
                            Math.floor(bot.entity.position.y),
                            Math.floor(bot.entity.position.z));
                        // Try to sleep (right-click bed)
                        await new Promise(r => setTimeout(r, 500));
                        // Use !useOn to interact with bed
                        await skills.useToolOn(bot, 'hand', bedItem.name);
                        console.log('[Survival] Placed bed and sleeping');
                        // Wait for morning
                        await new Promise(r => setTimeout(r, 3000));
                    } else {
                        console.log('[Survival] No bed available, need to craft one');
                    }
                } catch (e) {
                    console.log('[Survival] Sleep failed:', e.message);
                }
                break;
            }
            case TACTICAL.HUNT_FOOD: {
                // Try multiple food sources: animals, berries, crops
                const ANIMALS = ['cow', 'pig', 'chicken', 'sheep', 'rabbit'];
                const FOOD_BLOCKS = ['sweet_berry_bush', 'wheat', 'carrots', 'potatoes', 'melon', 'pumpkin'];
                let found = false;

                // First: check for nearby animals
                for (const animal of ANIMALS) {
                    const target = Object.values(bot.entities || {}).find(e =>
                        e.name?.includes(animal) && e !== bot.entity &&
                        bot.entity.position.distanceTo(e.position) < 24
                    );
                    if (target) {
                        try {
                            const sword = bot.inventory.items().find(i => i.name?.includes('sword'));
                            if (sword) await bot.equip(sword, 'hand');
                            await skills.attackEntity(bot, target, true);
                            console.log(`[Survival] Hunting ${animal} for food`);
                            found = true;
                        } catch (e) {}
                        break;
                    }
                }

                // Second: search for food blocks (berries, crops)
                if (!found) {
                    try {
                        // Use baritone to find and collect food blocks
                        if (bot.baritone) {
                            await bot.baritone('mine sweet_berry_bush');
                            console.log('[Survival] Searching for berry bushes');
                            found = true;
                        }
                    } catch (e) {}
                }

                if (!found) {
                    console.log('[Survival] No food sources found nearby');
                }
                break;
            }
            case TACTICAL.ABORT_TASK: {
                try {
                    if (bot._bridge) await bot._bridge.sendCommand('stop');
                    else if (bot.baritone) await bot.baritone('cancel');
                } catch (e) {}
                console.log('[Survival] Aborted current task for safety');
                break;
            }
        }
    }

    async _executeFlee(micro, skills) {
        const bot = this.bot;
        // If in water, skip moveAway (it fails in water) — micro_controller auto-swim handles it
        if (bot.entity?.isInWater) {
            console.log('[Survival] In water — skipping flee execute, auto-swim active');
            return;
        }
        switch (micro) {
            case ESCAPE.SPRINT_AWAY:
                try {
                    const state = captureState(bot);
                    if (state.closestMobPos) await skills.moveAwayFromPosition(bot, state.closestMobPos, 24);
                    else await skills.moveAway(bot, 24);
                } catch (e) {}
                console.log('[Survival] Sprint away');
                break;
            case ESCAPE.STRAFE_LEFT:
            case ESCAPE.STRAFE_RIGHT:
                try {
                    const state = captureState(bot);
                    if (state.closestMobPos) await skills.moveAwayFromPosition(bot, state.closestMobPos, 16);
                    else await skills.moveAway(bot, 16);
                } catch (e) {}
                console.log(`[Survival] ${micro}`);
                break;
            case ESCAPE.CONSUME_WHILE_FLEE:
                try {
                    const state = captureState(bot);
                    if (state.bestFood) await skills.consume(bot, state.bestFood);
                    if (state.closestMobPos) await skills.moveAwayFromPosition(bot, state.closestMobPos, 20);
                    else await skills.moveAway(bot, 20);
                } catch (e) {}
                console.log('[Survival] Eat + flee');
                break;
            case ESCAPE.DIG_IN:
                try { await skills.digDown(bot, 3); } catch (e) {}
                console.log('[Survival] Dug shelter while fleeing');
                break;
            default:
                try {
                    const state = captureState(bot);
                    if (state.closestMobPos) await skills.moveAwayFromPosition(bot, state.closestMobPos, 24);
                    else await skills.moveAway(bot, 24);
                } catch (e) {}
                break;
        }
    }

    async _executeHide(micro, skills) {
        const bot = this.bot;
        this.sheltering = true;
        switch (micro) {
            case null:
                console.log('[Survival] Staying hidden');
                break;
            case ESCAPE.DIG_IN:
            default:
                try {
                    // Classic dig-3-fill-1: dig down 3, seal head with 1 block
                    await skills.digDown(bot, 3);
                    await new Promise(r => setTimeout(r, 300));
                    const pos = bot.entity.position;
                    // After digging 3 down, head is at pos.y+1, seal at pos.y+2 (the block above head)
                    const sealY = Math.floor(pos.y) + 2;
                    const placeableItem = bot.inventory.items().find(i =>
                        i.name.includes('dirt') || i.name.includes('cobblestone') ||
                        i.name.includes('planks') || i.name.includes('stone')
                    );
                    if (placeableItem) {
                        await skills.placeBlock(bot, placeableItem.name, Math.floor(pos.x), sealY, Math.floor(pos.z));
                        console.log(`[Survival] Dig-3-fill-1: sealed at y=${sealY} with ${placeableItem.name}`);
                    } else {
                        console.log('[Survival] Dug down but no blocks to seal');
                    }
                } catch (e) {
                    console.log('[Survival] Dig shelter failed:', e.message);
                }
                break;
        }
    }

    _log(state, tactical, micro, reward, nextState) {
        const entry = {
            id: this.episodeId++,
            t: new Date().toISOString(),
            // Compact state for training
            s: {
                hp: state.health, hg: state.hunger, arm: state.armorValue,
                held: state.heldItem, sw: state.hasSword, shd: state.hasShield,
                mc: state.mobCount, md: state.closestMobDist, mt: state.closestMobType,
                rng: state.closestMobIsRanged, exp: state.closestMobIsExplosive,
                nt: state.isNight, sf: state.onSurface, fd: state.totalFood, blk: state.placeableBlocks,
                bed: state.hasBed, hs: state.hideSteps || 0, is: state.idleSteps || 0,
            },
            // Actions
            tactical,
            micro,
            // Counters for anti-passivity
            hs: this.hideSteps,
            is: this.idleSteps,
            // Reward (includes hide/idle penalties)
            r: reward,
            // Next state (same schema as s)
            ns: {
                hp: nextState.health, hg: nextState.hunger, arm: nextState.armorValue,
                sw: nextState.hasSword, shd: nextState.hasShield,
                mc: nextState.mobCount, md: nextState.closestMobDist, mt: nextState.closestMobType,
                rng: nextState.closestMobIsRanged, exp: nextState.closestMobIsExplosive,
                nt: nextState.isNight, sf: nextState.onSurface, fd: nextState.totalFood, blk: nextState.placeableBlocks,
                bed: nextState.hasBed, hs: nextState.hideSteps || this.hideSteps, is: nextState.idleSteps || this.idleSteps,
            },
            done: nextState.health <= 0,
        };

        try {
            fs.appendFileSync(this.logFile, JSON.stringify(entry) + '\n');
        } catch (e) {}
    }

    /**
     * Log a reflex-triggered event (separate from tactical episodes).
     */
    logReflex(action, state) {
        const entry = {
            id: this.episodeId++,
            t: new Date().toISOString(),
            source: 'reflex',
            tactical: action,
            s: {
                hp: state.health, hg: state.hunger, arm: state.armorValue || 0,
                sw: state.hasSword, mc: state.mobCount,
                md: state.closestMobDist, mt: state.closestMobType,
                nt: state.isNight, sf: state.onSurface, fd: state.totalFood,
            },
            r: 0,
            done: false,
        };
        try { fs.appendFileSync(this.logFile, JSON.stringify(entry) + '\n'); } catch (e) {}
    }

    /**
     * Called on death event. Writes a terminal episode with done=true and -5 reward.
     */
    logDeath() {
        if (!this.lastState) return;
        const now = Date.now();
        if (now - this.lastDeathLogAt < 2000) return;
        this.lastDeathLogAt = now;
        // Reset pillar failure counter on death/respawn
        this._pillarConsecutiveFails = 0;
        const deathState = { ...this.lastState, health: 0 };
        const entry = {
            id: this.episodeId++,
            t: new Date().toISOString(),
            s: {
                hp: this.lastState.health, hg: this.lastState.hunger, arm: this.lastState.armorValue,
                sw: this.lastState.hasSword, shd: this.lastState.hasShield,
                mc: this.lastState.mobCount, md: this.lastState.closestMobDist, mt: this.lastState.closestMobType,
                rng: this.lastState.closestMobIsRanged, exp: this.lastState.closestMobIsExplosive,
                nt: this.lastState.isNight, sf: this.lastState.onSurface, fd: this.lastState.totalFood, blk: this.lastState.placeableBlocks,
                bed: this.lastState.hasBed, hs: this.hideSteps, is: this.idleSteps,
            },
            tactical: this.lastTactical || 'continue',
            micro: this.lastMicro,
            r: -5.0,
            ns: {
                hp: 0, hg: 0, arm: 0,
                sw: false, shd: false,
                mc: 0, md: 999, mt: null,
                rng: false, exp: false,
                nt: this.lastState.isNight, sf: this.lastState.onSurface, fd: 0, blk: 0,
                bed: false, hs: 0, is: 0,
            },
            done: true,
        };
        try {
            fs.appendFileSync(this.logFile, JSON.stringify(entry) + '\n');
            console.log('[Survival] Death episode logged');
        } catch (e) {}

        // Feed terminal transition to online Q-model
        const deathNextState = { health: 0, hunger: 0, armorValue: 0, hasSword: false, hasShield: false, mobCount: 0, closestMobDist: 999, closestMobType: null, closestMobIsRanged: false, closestMobIsExplosive: false, isNight: this.lastState.isNight, onSurface: this.lastState.onSurface, totalFood: 0, placeableBlocks: 0, hasBed: false, hideSteps: 0, idleSteps: 0 };
        _onlineQ.store(this.lastState, this.lastTactical || 'continue', -5.0, deathNextState, true);
        setImmediate(() => { try { _onlineQ.update(8); } catch (e) {} });

        this.lastState = null;
        this.lastTactical = null;
        this.sheltering = false;
        this._fleeUntil = 0;
    }
}
