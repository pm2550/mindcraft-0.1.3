/**
 * Micro Controller: continuous 20Hz tick-level RL controller.
 * Always running when in combat/escape mode.
 * Publishes proposals to RTScheduler, never sends rl_action directly.
 * Reads mode from scheduler, adapts continuously.
 */

import fs from 'fs';
import path from 'path';
import { LocalEscapePlanner } from './local_escape_planner.js';
import { PolicyClient } from './policy_client.js';

const YAW_VALUES = [-30, -10, 0, 10, 30];
const PITCH_VALUES = [-5, 0, 5];

function quantizeYawIndex(deg) {
    let bestIdx = 0;
    let bestErr = Infinity;
    for (let i = 0; i < YAW_VALUES.length; i++) {
        const err = Math.abs(YAW_VALUES[i] - deg);
        if (err < bestErr) {
            bestErr = err;
            bestIdx = i;
        }
    }
    return bestIdx;
}

/**
 * Fight heuristic: approach target, time attacks to cooldown, exploit knockback.
 * Returns same 7-element array as bootstrapEscapeActions for Q-blend compatibility.
 */
function bootstrapFightActions(obs) {
    const threats = obs?.threats || [];
    if (threats.length === 0) {
        return [1, 1, 0, 0, 0, 2, 1]; // idle: no movement, no attack
    }

    // Find actual closest threat (Java does NOT sort by distance)
    const target = threats.reduce((closest, t) => (!closest || (t.dist || 99) < (closest.dist || 99)) ? t : closest, null);
    const dist = target.dist || 5;
    const cooldown = obs.attackCooldown ?? 0;

    // Aim at target: target.dx = right(+), target.dz = front(+) in player-relative
    const aimDeg = Math.atan2(target.dx || 0, target.dz || 0) * (180 / Math.PI);
    // Dampen small angles to prevent oscillation (±30 every tick = spinning)
    const dampedAim = Math.abs(aimDeg) < 12 ? 0 : aimDeg;
    const yawIdx = quantizeYawIndex(dampedAim);

    // Pitch toward target (look down if target is below, up if above)
    const dy = target.dy || 0;
    const pitchDeg = -Math.atan2(dy, dist) * (180 / Math.PI);
    const pitchIdx = pitchDeg < -3 ? 0 : (pitchDeg > 3 ? 2 : 1);

    let fb, lr, jump, sprint, attack;

    // Heading gate: only attack when crosshair roughly on mob (±40°).
    // Without this, bot attacks at heading=130° = punching air.
    const aimAligned = Math.abs(aimDeg) < 40;

    if (dist > 4.0) {
        // Phase: APPROACH — sprint toward target, but only if roughly facing mob.
        // Old code: unconditional forward → bot runs past mob when heading > 60°.
        // 66% of fight frames were dist>=4 because bot never closed the gap.
        if (Math.abs(aimDeg) < 60) {
            fb = 2;      // forward — facing mob enough to close distance
            sprint = 1;
        } else {
            fb = 1;      // stop — turn toward mob first, don't run past
            sprint = 0;
        }
        lr = 1;      // no strafe during approach
        jump = 0;
        attack = 0;
    } else if (dist >= 2.5 && dist <= 4.0) {
        // Phase: STRIKE ZONE — attack when cooldown ready AND facing mob
        if (cooldown >= 0.9 && aimAligned) {
            // Full cooldown + aligned: sprint-attack for max damage + knockback
            fb = 2;      // forward (close gap for hit)
            lr = 1;
            sprint = 1;
            attack = 1;
            jump = obs.onGround ? 1 : 0; // jump-crit if on ground
        } else {
            // Waiting for cooldown or alignment: circle-strafe
            fb = 1;      // no forward/back
            lr = Math.random() > 0.5 ? 0 : 2; // random strafe direction
            sprint = 0;
            attack = 0;
            jump = 0;
        }
    } else {
        // Phase: TOO CLOSE — back up, attack if cooldown ready AND aligned
        fb = 0;      // back
        lr = 1;
        sprint = 0;
        attack = (cooldown >= 0.7 && aimAligned) ? 1 : 0;
        jump = 0;
    }

    return [fb, lr, jump, sprint, attack, yawIdx, pitchIdx, 0, 0]; // +use, +sneak (not used in fight)
}

function bootstrapEscapeActions(mode, obs) {
    const threats = obs?.threats || [];
    if (threats.length === 0) {
        return [2, 1, 0, 1, 0, 2, 1]; // forward + sprint, no turn
    }

    // Multi-threat: compute weighted escape vector (closer threats push harder)
    let escapeDx = 0, escapeDz = 0;
    for (const t of threats) {
        const dist = Math.max(t.dist || 1, 0.5);
        const weight = 1 / (dist * dist); // inverse square: close mobs dominate
        escapeDx += -(t.dx || 0) * weight;
        escapeDz += -(t.dz || 0) * weight;
    }
    const escapeMag = Math.sqrt(escapeDx * escapeDx + escapeDz * escapeDz) || 1;
    escapeDx /= escapeMag;
    escapeDz /= escapeMag;

    // Yaw: turn toward escape direction.
    // escapeDx/escapeDz are player-relative (dx=right+, dz=forward+).
    // atan2(relX, relZ) = 0 when straight ahead, +90 when right, -90 when left.
    const relAngle = Math.atan2(escapeDx, escapeDz) * (180 / Math.PI);
    const headingErrEsc = ((relAngle + 540) % 360) - 180; // normalize [-180,180]
    const yawIdx = quantizeYawIndex(headingErrEsc);

    // If escape direction needs > 60° turn, back up instead of running forward
    const fb = Math.abs(headingErrEsc) > 60 ? 0 : 2; // back up if threat is mostly in front
    const sprint = fb === 2 ? 1 : 0; // only sprint when moving forward

    // Strafe away from threat center of mass
    let lr = 1; // none
    if (escapeDx > 0.3) lr = 2;       // escape is to the right -> strafe right
    else if (escapeDx < -0.3) lr = 0;  // escape is to the left -> strafe left

    const closestDist = threats.reduce((min, t) => Math.min(min, t?.dist ?? 99), 99);
    const jump = closestDist < 2.5 ? 1 : 0;
    const attack = 0;
    const pitchIdx = 1; // 0 pitch

    return [fb, lr, jump, sprint, attack, yawIdx, pitchIdx, 0, 0]; // +use, +sneak
}

/**
 * Generate a synthetic flee observation for imitation pre-training.
 * Produces a realistic-looking obs object the escape planner can process.
 */
function makeSyntheticFleeObs(rng = Math.random) {
    const numThreats = Math.floor(rng() * 3) + 1;
    const threats = Array.from({ length: numThreats }, () => {
        const angle = rng() * Math.PI * 2;
        const dist = 4 + rng() * 16;
        return {
            dx: Math.cos(angle) * dist, dz: Math.sin(angle) * dist,
            dist, type: 'zombie', hp: 20,
        };
    });

    // 51x51 terrain using Java encoding: 0=passable, 1=solid, 2=liquid, 3=void/drop
    const terrain = Array.from({ length: 2601 }, () => {
        const r = rng();
        return r < 0.65 ? 0 : (r < 0.82 ? 1 : (r < 0.93 ? 2 : 3));
        // 65% passable, 17% solid wall, 11% liquid, 7% void
    });
    // Center cell always passable (player's feet position)
    terrain[25 * 51 + 25] = 0;

    const yaw = rng() * 360 - 180;
    return {
        hp: 5 + Math.floor(rng() * 15), food: 10 + Math.floor(rng() * 10),
        armor: 0, onGround: rng() > 0.2, sprinting: false,
        vx: (rng() - 0.5) * 0.4, vy: 0, vz: (rng() - 0.5) * 0.4,
        selectedSlot: 0, attackCooldown: 0, los: rng() > 0.5,
        canDigDown: false, inWater: rng() < 0.1,
        yaw, threats, terrain,
        px: (rng() - 0.5) * 200, pz: (rng() - 0.5) * 200,
    };
}

/** Convert MC yaw (degrees) to world-space direction vector. */
function yawToWorldDir_MC(yawDeg) {
    const rad = yawDeg * Math.PI / 180;
    return { x: -Math.sin(rad), z: Math.cos(rad) };
}

/** Normalize angle to [-180, 180]. */
function normalizeAngle(a) {
    a = a % 360;
    if (a > 180) a -= 360;
    if (a <= -180) a += 360;
    return a;
}

// inputDim = 500 (12 self + 441 terrain 21x21 + 16 threats + 6 goal + 5 temporal + 20 baritone)
const INPUT_DIM = 500;
function obsToVec(obs, constraints) {
    if (!obs) return new Array(INPUT_DIM).fill(0);
    const threats = obs.threats || [];
    const fullTerrain = obs.terrain || [];
    // Extract inner 21x21 from 51x51 grid (offset ±10 from center 25)
    const terrain = [];
    for (let dz = -10; dz <= 10; dz++) {
        for (let dx = -10; dx <= 10; dx++) {
            const idx = (dz + 25) * 51 + (dx + 25);
            terrain.push(fullTerrain[idx] ?? 0);
        }
    }
    const anchorDist = constraints?._anchorDist || 0;
    const anchorDx = constraints?._anchorDx || 0;
    const anchorDz = constraints?._anchorDz || 0;
    const b = obs.baritone || {};

    const v = [
        // === Self [0-11] ===
        (obs.hp || 20) / 20,
        (obs.food || 20) / 20,
        (obs.armor || 0) / 20,
        obs.onGround ? 1 : 0,
        obs.sprinting ? 1 : 0,
        obs.vx || 0,
        obs.vy || 0,
        obs.vz || 0,
        (obs.selectedSlot || 0) / 8,
        (obs.attackCooldown || 0),
        obs.los ? 1 : 0,
        obs.canDigDown ? 1 : 0,

        // === Terrain grid [12-452]: 21x21=441 cells, each 0/1/2/3 normalized ===
        ...terrain.map(t => t / 3),

        // === Threats 0-3 [453-468]: 4 threats × 4 features ===
        ...Array.from({ length: 4 }, (_, i) => {
            const t = threats[i] || {};
            return [
                (t.dx || 0) / 25,
                (t.dz || 0) / 25,
                Math.min(t.dist || 32, 32) / 32,
                threats.length > i ? 1 : 0,
            ];
        }).flat(),

        // === Goal [469-474] ===
        Math.min(anchorDist, 48) / 48,
        anchorDx / 48,
        anchorDz / 48,
        (constraints?.maxRetreatRadius || 24) / 48,
        constraints?.returnRequired ? 1 : 0,
        ({ 'flee': 0, 'fight': 1, 'hide': 2 }[constraints?._mode] ?? 3) / 3,

        // === Temporal [475-479] ===
        Math.min(threats.length, 5) / 5,
        constraints?._recentDamage ? 1 : 0,
        Math.min(constraints?._timeInMode || 0, 100) / 100,
        constraints?._lastDyaw ? Math.max(-1, Math.min(1, (constraints._lastDyaw || 0) / 30)) : 0,
        constraints?._speed || 0,

        // === Baritone state [480-499] ===
        b.pathing ? 1 : 0,                                     // [480] is Baritone pathing
        0,                                                       // [481] reserved (process type)
        (b.goalDx || 0) / 48,                                   // [482] goal direction (player-relative)
        (b.goalDz || 0) / 48,                                   // [483]
        (b.goalDy || 0) / 48,                                   // [484]
        Math.min(b.goalDist || 0, 96) / 96,                     // [485] goal distance
        Math.min(b.estTicks || 500, 500) / 500,                 // [486] estimated ticks
        b.pathProgress || 0,                                     // [487] path progress 0-1
        0, 0, 0,                                                 // [488-490] reserved
        b.inFwd ? 1 : 0,                                        // [491] Baritone input: forward
        b.inBack ? 1 : 0,                                       // [492] back
        b.inLeft ? 1 : 0,                                       // [493] left
        b.inRight ? 1 : 0,                                      // [494] right
        b.inJump ? 1 : 0,                                       // [495] jump
        b.inSneak ? 1 : 0,                                      // [496] sneak
        b.inSprint ? 1 : 0,                                     // [497] sprint
        b.inAttack ? 1 : 0,                                     // [498] attack/dig
        b.inUse ? 1 : 0,                                        // [499] use/place
    ];

    // Pad/truncate to exact INPUT_DIM
    while (v.length < INPUT_DIM) v.push(0);
    return v.slice(0, INPUT_DIM);
}

// ============================================================
// Performance Tracker: rules are shield, Q earns authority
// ============================================================
class PerformanceTracker {
    constructor(windowSize = 100, minUpdatesBeforeBlend = 1000) {
        this.window = []; // { source: 'rule'|'q', reward }
        this.windowSize = windowSize;
        this.minUpdates = minUpdatesBeforeBlend;
        this.vetoCount = 0;
    }

    record(source, reward) {
        this.window.push({ source, reward });
        if (this.window.length > this.windowSize) this.window.shift();
    }

    getAlpha(totalUpdates) {
        if (totalUpdates < this.minUpdates) return 0;
        const rules = this.window.filter(w => w.source === 'rule');
        const qs = this.window.filter(w => w.source === 'q');
        if (rules.length < 10 || qs.length < 10) return 0; // not enough data
        const ruleScore = rules.reduce((s, w) => s + w.reward, 0) / rules.length;
        const qScore = qs.reduce((s, w) => s + w.reward, 0) / qs.length;
        // Q must be BETTER than rules to earn authority. Cap at 0.5.
        const raw = (qScore - ruleScore) / (Math.abs(ruleScore) + 0.01);
        return Math.max(0, Math.min(0.5, raw));
    }

    getStats() {
        const rules = this.window.filter(w => w.source === 'rule');
        const qs = this.window.filter(w => w.source === 'q');
        return {
            ruleScore: rules.length > 0 ? (rules.reduce((s, w) => s + w.reward, 0) / rules.length) : 0,
            qScore: qs.length > 0 ? (qs.reduce((s, w) => s + w.reward, 0) / qs.length) : 0,
            ruleN: rules.length,
            qN: qs.length,
            vetoes: this.vetoCount,
        };
    }
}

class MultiHeadQ {
    constructor(inputDim = INPUT_DIM, hiddenDim = 128) {
        this.inputDim = inputDim;
        this.hiddenDim = hiddenDim;
        this.lr = 0.003;
        this.gamma = 0.95;
        this.totalUpdates = 0;
        this.w1 = this._x(inputDim, hiddenDim); this.b1 = new Array(hiddenDim).fill(0);
        this.heads = [
            { w: this._x(hiddenDim, 3), b: [0,0,0] },    // 0: fb
            { w: this._x(hiddenDim, 3), b: [0,0,0] },    // 1: lr
            { w: this._x(hiddenDim, 2), b: [0,0] },       // 2: jump
            { w: this._x(hiddenDim, 2), b: [0,0] },       // 3: sprint
            { w: this._x(hiddenDim, 2), b: [0,0] },       // 4: attack
            { w: this._x(hiddenDim, 5), b: [0,0,0,0,0] }, // 5: yaw
            { w: this._x(hiddenDim, 3), b: [0,0,0] },     // 6: pitch
            { w: this._x(hiddenDim, 2), b: [0,0] },       // 7: use/place
            { w: this._x(hiddenDim, 2), b: [0,0] },       // 8: sneak
        ];
        this.tw1 = this.w1.map(r=>[...r]); this.tb1 = [...this.b1];
        this.theads = this.heads.map(h=>({w:h.w.map(r=>[...r]),b:[...h.b]}));
        this.buffer = []; this.bufferMax = 2000;
    }
    _x(r,c) { const s=Math.sqrt(2/r); return Array.from({length:r},()=>Array.from({length:c},()=>(Math.random()*2-1)*s)); }

    forward(x, t=false) {
        const w1=t?this.tw1:this.w1, b1=t?this.tb1:this.b1, hs=t?this.theads:this.heads;
        const h=new Array(this.hiddenDim);
        for(let j=0;j<this.hiddenDim;j++){let s=b1[j];for(let i=0;i<this.inputDim;i++)s+=x[i]*w1[i][j];h[j]=Math.max(0,s);}
        return { h, qHeads: hs.map(hd=>{const n=hd.b.length;const q=new Array(n);for(let j=0;j<n;j++){let s=hd.b[j];for(let i=0;i<this.hiddenDim;i++)s+=h[i]*hd.w[i][j];q[j]=s;}return q;}) };
    }

    predict(obs, constraints, eps=0.15) {
        const {qHeads}=this.forward(obsToVec(obs, constraints));
        return qHeads.map(q=>Math.random()<eps?Math.floor(Math.random()*q.length):q.indexOf(Math.max(...q)));
    }

    store(obs,a,r,ns,d,constraints) {
        this.buffer.push({s:obsToVec(obs,constraints),a,r,ns:obsToVec(ns,constraints),d:d?1:0});
        if(this.buffer.length>this.bufferMax) this.buffer.shift();
    }

    update(bs=4) {
        if(this.buffer.length<bs) return;
        for(let b=0;b<bs;b++){
            const{s,a,r,ns,d}=this.buffer[Math.floor(Math.random()*this.buffer.length)];
            const{h,qHeads}=this.forward(s);
            const{qHeads:nq}=this.forward(ns,true);
            const nv=nq.reduce((sum,q)=>sum+Math.max(...q),0)/this.heads.length;
            const tgt=r+this.gamma*nv*(1-d);
            for(let hi=0;hi<this.heads.length;hi++){
                const q=qHeads[hi],ai=a[hi],err=q[ai]-tgt;
                const dq=new Array(q.length).fill(0); dq[ai]=2*err/bs;
                for(let j=0;j<q.length;j++){this.heads[hi].b[j]-=this.lr*dq[j];for(let i=0;i<this.hiddenDim;i++)this.heads[hi].w[i][j]-=this.lr*dq[j]*h[i];}
            }
        }
        this.totalUpdates++;
        if(this.totalUpdates%50===0){this.tw1=this.w1.map(r=>[...r]);this.tb1=[...this.b1];this.theads=this.heads.map(h=>({w:h.w.map(r=>[...r]),b:[...h.b]}));}
    }

    /**
     * Imitation learning step: push Q(s)[expertAction] above all other actions by margin.
     * Only trains heads in `headMask` (default: all 7). For flee pre-training use [0,1,2,3].
     */
    pretrainStep(obsVec, expertActions, headMask = null, margin = 1.5) {
        const { h, qHeads } = this.forward(obsVec);
        for (let hi = 0; hi < this.heads.length; hi++) {
            if (headMask && !headMask.includes(hi)) continue;
            const q = qHeads[hi];
            const ai = expertActions[hi];
            if (ai === undefined || ai < 0 || ai >= q.length) continue;
            // Margin loss: want q[expert] >= max(others) + margin
            const maxOther = q.reduce((m, v, i) => i === ai ? m : Math.max(m, v), -Infinity);
            const violation = maxOther + margin - q[ai]; // > 0 means we need to update
            if (violation <= 0) continue; // already satisfying margin
            const dq = new Array(q.length).fill(0);
            dq[ai] = -2 * violation; // pull expert up
            for (let j = 0; j < q.length; j++) {
                this.heads[hi].b[j] -= this.lr * dq[j];
                for (let i = 0; i < this.hiddenDim; i++) {
                    this.heads[hi].w[i][j] -= this.lr * dq[j] * h[i];
                }
            }
        }
    }

    save(fp) {
        fs.writeFileSync(fp, JSON.stringify({
            w1: this.w1, b1: this.b1, heads: this.heads,
            totalUpdates: this.totalUpdates,
            inputDim: this.inputDim, hiddenDim: this.hiddenDim,
        }));
    }

    load(fp) {
        try {
            if (!fs.existsSync(fp)) return;
            const d = JSON.parse(fs.readFileSync(fp, 'utf8'));
            const savedIn = d.inputDim || d.w1?.length || 0;
            const savedHid = d.hiddenDim || d.w1?.[0]?.length || 0;

            // Verify action heads match current architecture
            const savedHeads = d.heads || [];
            const headsMatch = savedHeads.length === this.heads.length &&
                savedHeads.every((sh, i) => sh.b.length === this.heads[i].b.length);

            if (savedIn === this.inputDim && savedHid === this.hiddenDim && headsMatch) {
                // Perfect match
                this.w1 = d.w1; this.b1 = d.b1; this.heads = d.heads;
                this.totalUpdates = d.totalUpdates || 0;
                console.log(`[MicroQ] Loaded (${this.totalUpdates} updates)`);
            } else if (!headsMatch && savedHid === this.hiddenDim) {
                // Head count/dims changed: recover matching heads, init new ones
                const overlap = Math.min(savedIn, this.inputDim);
                if (savedIn !== this.inputDim) {
                    this.w1 = this._x(this.inputDim, this.hiddenDim);
                    for (let i = 0; i < overlap; i++) this.w1[i] = d.w1[i];
                } else {
                    this.w1 = d.w1;
                }
                this.b1 = d.b1;
                // Copy heads that match dims, keep fresh init for new/changed ones
                for (let i = 0; i < Math.min(savedHeads.length, this.heads.length); i++) {
                    if (savedHeads[i].b.length === this.heads[i].b.length) {
                        this.heads[i] = savedHeads[i];
                    }
                }
                this.totalUpdates = Math.floor((d.totalUpdates || 0) * 0.3);
                console.log(`[MicroQ] Partial head recovery: ${savedHeads.length}->${this.heads.length} heads, input ${savedIn}->${this.inputDim}, updates=${this.totalUpdates}`);
            } else if (savedHid === this.hiddenDim && savedIn > 0) {
                // Input dim changed: copy overlapping rows, xavier-init new ones
                const overlap = Math.min(savedIn, this.inputDim);
                this.w1 = this._x(this.inputDim, this.hiddenDim);
                for (let i = 0; i < overlap; i++) this.w1[i] = d.w1[i];
                this.b1 = d.b1;
                this.heads = d.heads;
                this.totalUpdates = Math.floor((d.totalUpdates || 0) * 0.3);
                console.log(`[MicroQ] Dimension-adapted: ${savedIn}->${this.inputDim}, kept ${overlap} rows, updates=${this.totalUpdates}`);
            } else {
                console.log(`[MicroQ] Incompatible dims (${savedIn}x${savedHid} vs ${this.inputDim}x${this.hiddenDim}), fresh start`);
                return;
            }
            this.tw1 = this.w1.map(r => [...r]); this.tb1 = [...this.b1];
            this.theads = this.heads.map(h => ({ w: h.w.map(r => [...r]), b: [...h.b] }));
        } catch (e) {
            console.log('[MicroQ] Load failed, fresh start');
        }
    }
}

export class MicroController {
    constructor(agent, botDir) {
        this.agent = agent;
        this.qNet = new MultiHeadQ();
        this.weightsPath = path.join(botDir, 'micro_q_weights.json');
        this.qNet.load(this.weightsPath);
        this.tracker = new PerformanceTracker(100, 1000);
        this.escapePlanner = new LocalEscapePlanner();
        this.policyClient = new PolicyClient();
        this.running = false;
        this.lastObs = null;
        this.lastActions = null;
        this._modeStartTime = 0;
        this._lastMode = 'idle';
        this._lastDyaw = 0;
        this._heldActions = null;
        this._holdMode = null;
        this._holdTicksRemaining = 0;
        this._boundBridge = null;
        this._boundWs = null;
        this._bridgeObsEnabled = false;
        this._stallTicks = 0;
        this._lastObsAt = 0;
        this._lastObserveEnableAt = 0;
        this._lastEmergencyAt = 0;
        this._smoothedDyaw = 0;

        // Imitation learning: collect Baritone movement data during idle/continue
        this._imitationBuffer = [];
        this._imitBufMax = 5000;
        this._imitTrainTick = 0;

        // Baritone flee: use Baritone pathfinding for medium-distance escape (dist > 8)
        this._baritoneFleeActive = false;
        this._baritoneFleeIssuedAt = 0;
        this._baritoneFleeTarget = null;

        // Dynamic policy explore rate: multi-indicator composite scoring
        this._exploreRate = 0.25;       // start conservative (was 40%, caused more deaths)
        this._exploreMin = 0.10;        // floor
        this._exploreMax = 0.80;        // ceiling
        this._exploreWindow = [];       // recent composite scores (0-1)
        this._exploreWindowSize = 30;   // sliding window
        // Per-episode tracking (short-term indicators)
        this._exploreHpAtStart = null;
        this._exploreDistAtStart = null; // nearest threat dist at explore start
        this._exploreSpeedSum = 0;       // accumulated speed during explore
        this._exploreTickCount = 0;
        this._exploreModeAtStart = null; // flee or fight
        this._exploreDeathDuringExplore = false;
        // Long-term indicators
        this._deathRateWindow = [];      // last 200 explore episodes: 1=death, 0=survived
        this._totalExploreEpisodes = 0;
        this._lifetimeTicks = 0;         // ticks since last death
        this._lifetimeHistory = [];      // last 10 lifetimes (ticks)
        this._rewardEMA = 0;            // exponential moving average of survival reward
        // PPO rollout collection
        this._rolloutBuffer = [];
        this._rolloutMaxSize = 2000;    // send when buffer reaches this size
        this._lastRolloutSendAt = 0;
        this._rolloutSendIntervalMs = 300000; // every 5 minutes

        // Save weights every 5 min
        setInterval(() => { try { this.qNet.save(this.weightsPath); } catch(e) {} }, 300000);
    }

    /**
     * Called when bot dies. Updates long-term indicators and marks current explore as failed.
     */
    onDeath() {
        // Mark current explore episode as death
        if (this._exploreHpAtStart !== null) {
            this._exploreDeathDuringExplore = true;
            this._endExploreEpisode(null); // force-end with death
        }
        // Long-term: record lifetime
        if (this._lifetimeTicks > 0) {
            this._lifetimeHistory.push(this._lifetimeTicks);
            if (this._lifetimeHistory.length > 10) this._lifetimeHistory.shift();
        }
        this._lifetimeTicks = 0;

        // Death cycle detection: if 3+ deaths in 90s, baritone escape on respawn
        const now = Date.now();
        if (!this._recentDeaths) this._recentDeaths = [];
        this._recentDeaths.push(now);
        // Keep only deaths in last 90 seconds
        this._recentDeaths = this._recentDeaths.filter(t => now - t < 90000);
        if (this._recentDeaths.length >= 3) {
            const px = this.agent.bot?.entity?.position?.x || 0;
            const pz = this.agent.bot?.entity?.position?.z || 0;
            // Escape far away in a random direction after respawn
            const escX = Math.round(px + (Math.random() - 0.5) * 200);
            const escZ = Math.round(pz + (Math.random() - 0.5) * 200);
            console.log(`[MicroCtrl] DEATH CYCLE (${this._recentDeaths.length} deaths in 90s) — baritone escape to ${escX} 70 ${escZ} on respawn`);
            // Delay slightly to let respawn complete
            setTimeout(() => {
                const bridge = this.agent.bot?._bridge;
                if (bridge) {
                    bridge.sendCommand('baritone', { command: `goto ${escX} 70 ${escZ}` }).catch(() => {});
                }
                // Suspend micro control for 20s to let baritone work
                this._stuckEscapeUntil = Date.now() + 20000;
            }, 2000);
            this._recentDeaths = []; // reset after escape
        }
    }

    /**
     * End an explore episode and compute composite score.
     * Called after 20 ticks or on death.
     */
    _endExploreEpisode(obs) {
        const died = this._exploreDeathDuringExplore;
        const mode = this._exploreModeAtStart || 'flee';

        // Short-term indicator 1: HP change (weight 0.3)
        const hpNow = died ? 0 : (obs?.hp || 0);
        const hpDelta = hpNow - (this._exploreHpAtStart || 20);
        const hpScore = Math.max(0, Math.min(1, 1 + hpDelta / 10)); // -10hp=0, 0=1

        // Short-term indicator 2: Distance change (weight 0.25)
        const distNow = obs?.threats?.reduce((m, t) => Math.min(m, t?.dist ?? 99), 99) ?? 99;
        const distDelta = distNow - (this._exploreDistAtStart || 99);
        let distScore;
        if (mode === 'flee') {
            distScore = Math.max(0, Math.min(1, 0.5 + distDelta / 10)); // farther=better
        } else {
            distScore = Math.max(0, Math.min(1, 0.5 - distDelta / 10)); // closer=better for fight
        }

        // Short-term indicator 3: Survived? (weight 0.3)
        const survivalScore = died ? 0 : 1;

        // Short-term indicator 4: Speed maintenance (weight 0.15)
        const avgSpeed = this._exploreTickCount > 0 ? this._exploreSpeedSum / this._exploreTickCount : 0;
        const speedScore = Math.min(avgSpeed / 0.2, 1);

        // Composite
        const composite = 0.3 * hpScore + 0.25 * distScore + 0.3 * survivalScore + 0.15 * speedScore;

        // Push to window
        this._exploreWindow.push(composite);
        if (this._exploreWindow.length > this._exploreWindowSize) this._exploreWindow.shift();

        // Push death/survive to long-term window
        this._deathRateWindow.push(died ? 1 : 0);
        if (this._deathRateWindow.length > 200) this._deathRateWindow.shift();
        this._totalExploreEpisodes++;

        // Adjust rate
        if (this._exploreWindow.length >= 10) {
            const avgScore = this._exploreWindow.reduce((a, b) => a + b, 0) / this._exploreWindow.length;
            const deathRate = this._deathRateWindow.reduce((a, b) => a + b, 0) / this._deathRateWindow.length;
            const oldRate = this._exploreRate;

            let adjustment = 0;
            if (avgScore > 0.7) adjustment = 0.03 * (avgScore - 0.7) / 0.3;       // max +3%
            else if (avgScore < 0.4) adjustment = -0.05 * (0.4 - avgScore) / 0.4;  // max -5% (asymmetric)

            // Death rate brake: if >20% deaths, force decrease (was 5%, too strict for badlands)
            if (deathRate > 0.20) adjustment = Math.min(adjustment, -0.01);

            this._exploreRate = Math.max(this._exploreMin, Math.min(this._exploreMax, this._exploreRate + adjustment));

            if (Math.abs(this._exploreRate - oldRate) > 0.001 || this._totalExploreEpisodes % 50 === 0) {
                console.log(`[Policy] rate ${oldRate.toFixed(2)}→${this._exploreRate.toFixed(2)} score=${avgScore.toFixed(2)} deathRate=${(deathRate*100).toFixed(1)}% hp=${hpScore.toFixed(2)} dist=${distScore.toFixed(2)} alive=${survivalScore} spd=${speedScore.toFixed(2)} (n=${this._totalExploreEpisodes})`);
            }
        }

        // Reset per-episode state
        this._exploreHpAtStart = null;
        this._exploreDistAtStart = null;
        this._exploreSpeedSum = 0;
        this._exploreTickCount = 0;
        this._exploreModeAtStart = null;
        this._exploreDeathDuringExplore = false;
    }

    /**
     * Collect a rollout step for PPO training.
     * Reward is assigned retrospectively from next tick's HP/dist change.
     */
    _collectRolloutStep(obs, policyAction, rawObs) {
        const mode = this._exploreModeAtStart || 'flee';
        const constraints = this.agent.scheduler?.constraints || {};

        // Safety gate: if death rate > 3%, don't collect PPO data
        const deathRate = this._deathRateWindow.length > 0
            ? this._deathRateWindow.reduce((a, b) => a + b, 0) / this._deathRateWindow.length
            : 0;
        if (deathRate > 0.03) {
            this._rolloutBuffer = [];
            return;
        }

        // Assign reward to previous step using full _reward() function
        if (this._rolloutBuffer.length > 0) {
            const prev = this._rolloutBuffer[this._rolloutBuffer.length - 1];
            if (prev.reward === null) {
                prev.reward = this._reward(prev._mode, prev._prevObs, rawObs, constraints);
                prev.done = (rawObs.hp || 20) <= 0;
            }
        }

        // Push new step
        this._rolloutBuffer.push({
            obs: obs,
            action: policyAction,
            logp: policyAction?.logp || 0,
            value: policyAction?.value || 0,
            reward: null,
            done: false,
            _prevObs: rawObs,  // for computing reward next tick
            _mode: mode,
        });

        // Send when buffer reaches 2000 steps or 5 minutes elapsed
        const now = Date.now();
        if (this._rolloutBuffer.length >= this._rolloutMaxSize ||
            (this._rolloutBuffer.length >= 500 && now - this._lastRolloutSendAt > this._rolloutSendIntervalMs)) {
            const last = this._rolloutBuffer[this._rolloutBuffer.length - 1];
            if (last.reward === null) last.reward = 0;
            const sendBuf = this._rolloutBuffer.map(s => ({
                obs: s.obs, action: s.action, logp: s.logp, value: s.value,
                reward: s.reward || 0, done: s.done || false,
            }));
            this.policyClient?.sendRollout?.(sendBuf);
            this._rolloutBuffer = [];
            this._lastRolloutSendAt = now;
        }
    }

    /**
     * Infer expert actions from actual key states when available, otherwise
     * fall back to Baritone's input overrides, then velocity inference.
     */
    _inferActionsFromBaritone(obs) {
        const hasActualKeys = [
            'keyFwd', 'keyBack', 'keyLeft', 'keyRight',
            'keyJump', 'keySprint', 'keyAttack', 'keyUse', 'keySneak'
        ].some(k => obs[k] !== undefined && obs[k] !== null);
        const b = obs.baritone || {};
        const direct = hasActualKeys ? {
            inFwd: !!obs.keyFwd,
            inBack: !!obs.keyBack,
            inLeft: !!obs.keyLeft,
            inRight: !!obs.keyRight,
            inJump: !!obs.keyJump,
            inSprint: !!obs.keySprint,
            inAttack: !!obs.keyAttack,
            inUse: !!obs.keyUse,
            inSneak: !!obs.keySneak,
        } : (
            (b.inFwd !== undefined || b.inBack !== undefined || b.inLeft !== undefined || b.inRight !== undefined ||
             b.inJump !== undefined || b.inSprint !== undefined || b.inAttack !== undefined || b.inUse !== undefined ||
             b.inSneak !== undefined) ? b : null
        );

        if (direct) {
            // Direct input teacher signal from the actual key state or Baritone override.
            const fb = direct.inFwd ? 2 : (direct.inBack ? 0 : 1);
            const lr = direct.inLeft ? 0 : (direct.inRight ? 2 : 1);
            const jump = direct.inJump ? 1 : 0;
            const sprint = direct.inSprint ? 1 : 0;
            const attack = direct.inAttack ? 1 : 0;
            const dyaw = (obs.yaw || 0) - (this.lastObs?.yaw || obs.yaw || 0);
            const yawIdx = quantizeYawIndex(dyaw);
            const dpitch = (obs.pitch || 0) - (this.lastObs?.pitch || obs.pitch || 0);
            const pitchIdx = dpitch < -3 ? 0 : (dpitch > 3 ? 2 : 1);
            const use = direct.inUse ? 1 : 0;
            const sneak = direct.inSneak ? 1 : 0;
            return {
                actions: [fb, lr, jump, sprint, attack, yawIdx, pitchIdx, use, sneak],
                preciseTeacher: true,
                source: hasActualKeys ? 'keys' : 'baritone',
            };
        }
        // Fallback: infer from velocity (no Baritone direct input available)
        const yawRad = (obs.yaw || 0) * Math.PI / 180;
        const vx = obs.vx || 0, vz = obs.vz || 0;
        const speed = Math.sqrt(vx * vx + vz * vz);
        const fwdDot = vx * (-Math.sin(yawRad)) + vz * Math.cos(yawRad);
        const strafeDot = vx * Math.cos(yawRad) + vz * Math.sin(yawRad);
        const fb = fwdDot > 0.08 ? 2 : (fwdDot < -0.05 ? 0 : 1);
        const lr = strafeDot > 0.08 ? 2 : (strafeDot < -0.08 ? 0 : 1);
        const sprint = speed > 0.18 ? 1 : 0;
        const jump = (obs.vy || 0) > 0.3 ? 1 : 0;
        return { actions: [fb, lr, jump, sprint, 0, 2, 1, 0, 0], preciseTeacher: false, source: 'velocity' };
    }

    /**
     * Collect combat demo data for PyTorch imitation training.
     * Writes obs+action pairs to combat_demos.jsonl during flee/fight.
     * Only collects from escape planner / fight heuristic (not policy explore).
     */
    _collectCombatDemo(obs, action, constraints, mode) {
        if (!this._combatDemoStream) {
            const demoPath = path.join('./bots', this.agentName || 'GaoBot1', 'combat_demos.jsonl');
            this._combatDemoStream = fs.createWriteStream(demoPath, { flags: 'a' });
            this._combatDemoCount = 0;
        }
        // Format obs same way as PolicyClient.formatObs for consistency
        const policyObs = PolicyClient.formatObs(obs, constraints, mode);
        if (!policyObs) return;
        // One-time debug: verify threats have hp/type after formatObs
        if (!this._loggedThreatFields && policyObs.threats?.length > 0) {
            this._loggedThreatFields = true;
            console.log(`[CombatDemo] threat fields after formatObs: ${JSON.stringify(policyObs.threats[0])}`);
            console.log(`[CombatDemo] raw obs threat fields: ${obs.threats?.[0] ? JSON.stringify(Object.keys(obs.threats[0])) : 'none'}`);
        }
        // Add actual action labels (what the expert system decided)
        const demo = {
            obs: policyObs,
            action: {
                move_fwd: action.forward ? 1.0 : (action.back ? -1.0 : 0.0),
                move_strafe: action.right ? 1.0 : (action.left ? -1.0 : 0.0),
                yaw: action.dyaw || 0,
                pitch: action.dpitch || 0,
                jump: action.jump ? 1.0 : 0.0,
                sprint: action.sprint ? 1.0 : 0.0,
                attack: action.attack ? 1.0 : 0.0,
                use: action.use ? 1.0 : 0.0,
                sneak: action.sneak ? 1.0 : 0.0,
            },
            mode,
            tick: obs.tick,
        };
        this._combatDemoStream.write(JSON.stringify(demo) + '\n');
        this._combatDemoCount++;
        if (this._combatDemoCount % 1000 === 0) {
            console.log(`[CombatDemo] ${this._combatDemoCount} frames collected (mode=${mode})`);
        }
    }

    /**
     * Collect imitation sample during Baritone-controlled movement (idle/continue).
     * When Baritone direct inputs available: train ALL 7 heads (including attack/yaw/pitch).
     * Otherwise: train only movement heads [0-3].
     */
    _collectImitation(obs, constraints) {
        if (!obs || !this.lastObs) return;
        const speed = Math.sqrt((obs.vx || 0) ** 2 + (obs.vz || 0) ** 2);
        const hasActualKeys = [
            'keyFwd', 'keyBack', 'keyLeft', 'keyRight',
            'keyJump', 'keySprint', 'keyAttack', 'keyUse', 'keySneak'
        ].some(k => obs[k] !== undefined && obs[k] !== null);
        const directActive = hasActualKeys && (
            obs.keyFwd || obs.keyBack || obs.keyLeft || obs.keyRight ||
            obs.keyJump || obs.keySprint || obs.keyAttack || obs.keyUse || obs.keySneak
        );
        const b = obs.baritone || {};
        const baritoneActive = !hasActualKeys && (b.inFwd || b.inBack || b.inAttack || b.inUse || b.inJump || b.inLeft || b.inRight || b.inSprint || b.inSneak);
        // Collect when moving OR when Baritone is actively doing something (mining = no speed but attack)
        if (speed < 0.05 && !directActive && !baritoneActive) return;
        const { actions, preciseTeacher } = this._inferActionsFromBaritone(obs);
        const obsVec = obsToVec(obs, constraints);
        this._imitationBuffer.push({ s: obsVec, a: actions, allHeads: preciseTeacher });
        if (this._imitationBuffer.length > this._imitBufMax) this._imitationBuffer.shift();

        // Train from imitation buffer every 50 ticks
        this._imitTrainTick++;
        if (this._imitTrainTick % 50 === 0 && this._imitationBuffer.length >= 8) {
            for (let i = 0; i < 4; i++) {
                const sample = this._imitationBuffer[Math.floor(Math.random() * this._imitationBuffer.length)];
                // Baritone direct inputs: train all 7 heads. Velocity fallback: movement only.
                const headMask = sample.allHeads ? null : [0, 1, 2, 3];
                this.qNet.pretrainStep(sample.s, sample.a, headMask);
            }
        }
    }

    /**
     * Issue a Baritone goto command toward an escape point in world space.
     * Fire-and-forget (async, no await).
     */
    _baritoneFleeGoto(obs, escWorldDx, escWorldDz, dist = 20) {
        const bridge = this.agent.bot?._bridge;
        if (!bridge || obs.px === undefined) return;
        // Use safe Y: never below 70 to avoid underwater pathing.
        // If bot is above 70, use current Y+5 to prefer uphill escape.
        const rawY = Math.round(this.agent.bot?.entity?.position?.y ?? 64);
        const safeY = Math.max(70, rawY + 5);
        const tx = Math.round(obs.px + escWorldDx * dist);
        const tz = Math.round(obs.pz + escWorldDz * dist);
        const cmd = `goto ${tx} ${safeY} ${tz}`;
        bridge.sendCommand('baritone', { command: cmd }).catch(() => {});
        this._baritoneFleeActive = true;
        this._baritoneFleeIssuedAt = Date.now();
        this._baritoneFleeTarget = { tx, tz };
        console.log(`[MicroCtrl] Baritone flee goto ${tx} ${safeY} ${tz}`);
    }

    /**
     * Cancel Baritone and switch back to raw escape control.
     */
    _cancelBaritoneFlee() {
        if (!this._baritoneFleeActive) return;
        const bridge = this.agent.bot?._bridge;
        bridge?.sendCommand('baritone', { command: 'cancel' }).catch(() => {});
        this._baritoneFleeActive = false;
        this._baritoneFleeTarget = null;
    }

    /**
     * Start continuous micro control loop.
     * Listens to rl_obs, publishes proposals to scheduler every tick.
     * Does NOT stop until explicitly told to.
     */
    /**
     * Imitation pre-training: generate synthetic flee states, run escape planner,
     * train Q to mimic planner's fb/lr/jump/sprint decisions (heads 0-3).
     */
    pretrainFromEscapePlanner(steps = 3000) {
        if (this.qNet.totalUpdates > 0) return; // only pretrain from scratch
        console.log(`[MicroQ] Imitation pre-training: ${steps} steps...`);
        const fleeConstraints = { _mode: 'flee', maxRetreatRadius: 24, _anchorDist: 0, _anchorDx: 0, _anchorDz: 0 };
        let trained = 0;
        for (let i = 0; i < steps; i++) {
            const obs = makeSyntheticFleeObs();
            const plannerAction = this.escapePlanner.tick(obs, 'flee', fleeConstraints);
            const expertFb = plannerAction.forward ? 2 : (plannerAction.back ? 0 : 1);
            const expertLr = plannerAction.left ? 0 : (plannerAction.right ? 2 : 1);
            const expertJump = plannerAction.jump ? 1 : 0;
            const expertSprint = plannerAction.sprint ? 1 : 0;
            const expertActions = [expertFb, expertLr, expertJump, expertSprint, 0, 2, 1, 0, 0];
            const obsVec = obsToVec(obs, fleeConstraints);
            this.qNet.pretrainStep(obsVec, expertActions, [0, 1, 2, 3]);
            trained++;
        }
        console.log(`[MicroQ] Imitation pre-training done: ${trained} steps`);
        this.escapePlanner.reset(); // clear internal state polluted by synthetic observations
    }

    start() {
        if (this.running) return;
        const bridge = this.agent.bot?._bridge;
        const scheduler = this.agent.scheduler;
        if (!bridge || !scheduler) return;

        // Pre-train Q to imitate escape planner before first tick
        this.pretrainFromEscapePlanner(3000);

        this.running = true;
        this.policyClient?.connect?.();
        console.log('[MicroCtrl] Starting continuous micro control');

        this._onMsg = async (data) => {
            if (!this.running) return;
            try {
                const raw = typeof data === 'string' ? data : data.toString();
                const msg = JSON.parse(raw);
                if (msg.type !== 'rl_obs') return;

                const obs = { ...msg, threats: Array.isArray(msg.threats) ? msg.threats : [], terrain: Array.isArray(msg.terrain) ? msg.terrain : [] };
                const mode = scheduler.currentMode;
                this._obsCount = (this._obsCount || 0) + 1;
                this._lastObsAt = Date.now();
                // One-time log to verify Baritone state injection
                if (this._obsCount === 50 && !this._loggedBaritoneCheck) {
                    this._loggedBaritoneCheck = true;
                    console.log(`[MicroCtrl] Baritone in rl_obs: ${obs.baritone ? JSON.stringify(obs.baritone) : 'NOT_PRESENT'}`);
                }
                // Log once when Baritone becomes active (pathing with inputs)
                if (obs.baritone?.pathing && !this._loggedBaritoneActive) {
                    this._loggedBaritoneActive = true;
                    console.log(`[MicroCtrl] Baritone ACTIVE: ${JSON.stringify(obs.baritone)}`);
                }
                if (!obs.baritone?.pathing) this._loggedBaritoneActive = false;

                // Update scheduler's state bus
                scheduler.updateState(obs);

                // Log first active obs
                if ((mode === 'flee' || mode === 'fight') && !this._loggedFirstActive) {
                    this._loggedFirstActive = true;
                    console.log(`[MicroCtrl] First active obs in ${mode}: hp=${obs.hp} threats=${obs.threats?.length} obsTotal=${this._obsCount}`);
                }

                // Track mode transitions
                if (mode !== this._lastMode) {
                    this.policyClient?.resetHidden?.();
                    this._modeStartTime = Date.now();
                    this._lastMode = mode;
                    this._fightHeadingEma = null;
                    // Force pitch toward horizontal on mode change to flee/fight.
                    // digDown sets pitch=90° (look down) — if flee/fight starts with pitch=90,
                    // bot stares at ground and every attack misses. Reset immediately.
                    if ((mode === 'flee' || mode === 'fight') && Math.abs(obs.pitch || 0) > 15) {
                        const resetPitch = -(obs.pitch || 0); // snap to 0
                        const bridge = this.agent.bot?._bridge;
                        if (bridge) {
                            bridge.sendCommand('rl_action', { dpitch: resetPitch }).catch(() => {});
                            console.log(`[MicroCtrl] Pitch reset on ${mode} entry: ${(obs.pitch||0).toFixed(0)}° → 0° (dpitch=${resetPitch.toFixed(0)})`);
                        }
                    }
                }

                // === GLOBAL water: swim up via scheduler (not direct rl_action — that gets overridden) ===
                if (obs.inWater) {
                    const currentPitch = obs.pitch || 0;
                    const swimPitch = Math.max(-15, Math.min(15, (-15 - currentPitch) * 0.4));
                    // Publish swim action through scheduler so agent.js sends it (not overridden)
                    scheduler.publish('micro', {
                        priority: 10, // highest priority — override everything
                        ttlMs: 200,
                        action: { forward: true, jump: true, sprint: true, dyaw: 0, dpitch: swimPitch },
                    });
                    // One-time water handling: only cancel baritone in flee/fight/hide.
                    // In navigate/continue: baritone has its own water pathfinding, don't interrupt.
                    // Old bug: this canceled goto-1200 whenever bot crossed a river → bot returned to spawn.
                    if (!this._waterAutoSwimIssued) {
                        this._waterAutoSwimIssued = true;
                        const isEscapeWater = mode === 'flee' || mode === 'fight' || mode === 'hide';
                        if (isEscapeWater) {
                            const bridge = this.agent.bot?._bridge;
                            if (this.agent.bot?._baritoneActive || this._baritoneFleeActive) {
                                if (bridge) {
                                    bridge.sendCommand('baritone', { command: 'cancel' }).catch(() => {});
                                    this.agent.bot._baritoneActive = false;
                                    if (this._baritoneFleeActive) this._cancelBaritoneFlee();
                                }
                            }
                            const landPos = this.escapePlanner?._lastLandPos;
                            if (bridge) {
                                const tx = landPos ? Math.round(landPos.x) : 25;
                                const tz = landPos ? Math.round(landPos.z) : -50;
                                bridge.sendCommand('baritone', { command: `goto ${tx} 80 ${tz}` }).catch(() => {});
                                console.log(`[MicroCtrl] Water+escape (mode=${mode}) — goto land ${tx} 80 ${tz}`);
                            }
                        } else {
                            console.log(`[MicroCtrl] Water in ${mode} — swim only, keeping baritone goal`);
                        }
                    }
                    this.lastObs = obs;
                    return;
                }
                if (!obs.inWater) this._waterAutoSwimIssued = false;

                // === Stuck detection: runs BEFORE navigate return, on ALL ticks ===
                {
                    const hasThreat = obs.threats?.some(t => t && t.dist < 25);
                    const spx = obs.px ?? 0, spz = obs.pz ?? 0;
                    if (hasThreat) {
                        if (!this._stuckCheckStart) {
                            this._stuckCheckPx = spx; this._stuckCheckPz = spz;
                            this._stuckCheckStart = Date.now();
                        } else {
                            const moved = Math.sqrt((spx - this._stuckCheckPx) ** 2 + (spz - this._stuckCheckPz) ** 2);
                            if (moved > 8) {
                                this._stuckCheckPx = spx; this._stuckCheckPz = spz;
                                this._stuckCheckStart = Date.now();
                            } else if (Date.now() - this._stuckCheckStart > 45000) {
                                const escX = Math.round(spx + (Math.random() - 0.5) * 120);
                                const escZ = Math.round(spz + (Math.random() - 0.5) * 120);
                                console.log(`[MicroCtrl] FLEE STUCK 45s (moved=${moved.toFixed(1)}m pos=${spx.toFixed(0)},${spz.toFixed(0)}) — baritone escape to ${escX} 70 ${escZ}, suspending micro 15s`);
                                const stuckBridge = this.agent.bot?._bridge;
                                if (stuckBridge) {
                                    stuckBridge.sendCommand('baritone', { command: `goto ${escX} 70 ${escZ}` }).catch(() => {});
                                }
                                this._stuckEscapeUntil = Date.now() + 15000;
                                this._stuckCheckPx = spx; this._stuckCheckPz = spz;
                                this._stuckCheckStart = Date.now();
                            }
                        }
                    } else {
                        this._stuckCheckStart = null;
                        this._stuckEscapeUntil = 0;
                    }
                    // If stuck-escape active, skip ALL micro control
                    if (this._stuckEscapeUntil && Date.now() < this._stuckEscapeUntil) {
                        this.lastObs = obs;
                        return;
                    }
                }

                // During idle/continue: collect Baritone imitation data for BOTH Q-net and Policy
                if (mode !== 'flee' && mode !== 'fight' && mode !== 'hide') {
                    // Cancel any active Baritone flee when mode leaves flee
                    if (this._baritoneFleeActive) this._cancelBaritoneFlee();
                    const idleConstraints = { ...scheduler.constraints, _mode: mode };
                    this._collectImitation(obs, idleConstraints);
                    // Send obs to policy server so value head + action heads learn navigation
                    const idlePolicyObs = PolicyClient.formatObs(obs, idleConstraints, mode);
                    this.policyClient?.sendObs?.(idlePolicyObs);
                    // Collect Baritone navigation demos for Policy offline training
                    // Only collect when Baritone is actively pathing — otherwise we record
                    // garbage (bot standing still with micro-velocity, jump=!onGround, etc.)
                    const b = obs.baritone;
                    if (b && b.pathing) {
                        const navSpeed = Math.sqrt((obs.vx||0)**2 + (obs.vz||0)**2);
                        if (navSpeed > 0.03 || b.inFwd || b.inBack || b.inJump || b.inAttack) {
                            const navAction = {
                                forward: !!b.inFwd, back: !!b.inBack, left: !!b.inLeft, right: !!b.inRight,
                                jump: !!b.inJump, sprint: !!b.inSprint, attack: !!b.inAttack, use: !!b.inUse,
                                sneak: !!b.inSneak, dyaw: 0, dpitch: 0,
                            };
                            this._collectCombatDemo(obs, navAction, idleConstraints, 'navigate');
                        }
                    }
                    this.lastObs = obs;
                    this.lastActions = null;
                    this._lastWasQMode = false;
                    this._heldActions = null;
                    this._holdMode = null;
                    this._holdTicksRemaining = 0;
                    return;
                }

                // Build constraints with computed spatial + temporal features
                const constraints = { ...scheduler.constraints, _mode: mode };
                const pos = this.agent.bot?.entity?.position;
                if (constraints.anchor && pos) {
                    const dx = constraints.anchor.x - pos.x;
                    const dz = constraints.anchor.z - pos.z;
                    constraints._anchorDist = Math.sqrt(dx * dx + dz * dz);
                    // Anchor direction relative to player facing
                    const yawRad = (this.agent.bot?.entity?.yaw || 0) * Math.PI / 180;
                    constraints._anchorDx = (dx * Math.cos(yawRad) + dz * Math.sin(yawRad));
                    constraints._anchorDz = (-dx * Math.sin(yawRad) + dz * Math.cos(yawRad));
                } else {
                    constraints._anchorDist = 0;
                    constraints._anchorDx = 0;
                    constraints._anchorDz = 0;
                }
                // Temporal
                constraints._recentDamage = this.agent.bot?.lastDamageTime ? (Date.now() - this.agent.bot.lastDamageTime < 3000) : false;
                constraints._timeInMode = this._modeStartTime ? (Date.now() - this._modeStartTime) / 1000 : 0;
                constraints._lastDyaw = this._lastDyaw || 0;
                constraints._speed = pos ? Math.sqrt((obs.vx||0)**2 + (obs.vz||0)**2) : 0;

                const microFrozen = !!constraints.freezeMicro ||
                    (constraints.freezeMicroUntil && Date.now() < constraints.freezeMicroUntil);
                if (microFrozen) {
                    this.lastObs = obs;
                    this.lastActions = null;
                    this._lastWasQMode = false;
                    this._heldActions = null;
                    this._holdMode = null;
                    this._holdTicksRemaining = 0;
                    scheduler.clearProposal?.('micro');
                    return;
                }

                // Learn from previous step — only store real Q actions (fight mode)
                // Escape mode uses planner, not Q — don't pollute Q buffer with placeholder actions
                if (this.lastObs && this.lastActions && this._lastWasQMode) {
                    const reward = this._reward(mode, this.lastObs, obs, constraints);
                    this.qNet.store(this.lastObs, this.lastActions, reward, obs, (obs.hp || 20) <= 0, constraints);
                    // Split credit: Q heads vs heuristic heads
                    const qRatio = (this._lastQHeadCount || 0) / this.qNet.heads.length;
                    if (qRatio > 0) this.tracker.record('q', reward * qRatio);
                    if (qRatio < 1) this.tracker.record('rule', reward * (1 - qRatio));
                    if (this.qNet.buffer.length >= 4) {
                        setImmediate(() => { try { this.qNet.update(4); } catch(e) {} });
                    }
                }

                const isEscapeMode = mode === 'flee' || mode === 'hide';
                let action;

                if (isEscapeMode) {
                    // Cancel any lingering Baritone navigation during flee/fight
                    if (this._baritoneFleeActive) this._cancelBaritoneFlee();

                    // Track outcome of policy explore — multi-indicator composite scoring
                    this._lifetimeTicks++; // count ticks alive for long-term indicator
                    if (this._exploreHpAtStart !== null) {
                        this._exploreTickCount++;
                        // Accumulate speed for speed indicator
                        this._exploreSpeedSum += Math.sqrt((obs.vx||0)**2 + (obs.vz||0)**2);
                        if (this._exploreTickCount >= 20) {
                            this._endExploreEpisode(obs);
                        }
                    }

                    // Check for policy action FIRST (from previous tick's obs), then send new obs
                    const policyAction = this.policyClient?.getLatestAction?.(200);
                    const policyObs = PolicyClient.formatObs(obs, constraints, mode);
                    this.policyClient?.sendObs?.(policyObs);
                    if (this._obsCount % 100 === 0) {
                        console.log(`[Policy] check: hasAction=${!!policyAction} connected=${this.policyClient?.connected} obs=${this.policyClient?._obsCount} exploreRate=${this._exploreRate.toFixed(2)}`);
                    }
                    if (policyAction && Math.random() < this._exploreRate) {
                        // Policy network controls (the real brain)
                        action = PolicyClient.actionToRlAction(policyAction);
                        this.lastActions = null;
                        this._lastWasQMode = false;
                        this._lastQHeadCount = 0;
                        this._lastPolicyExplore = true;
                        // Record start state for multi-indicator evaluation
                        this._exploreHpAtStart = obs.hp || 20;
                        this._exploreDistAtStart = obs.threats?.reduce((m, t) => Math.min(m, t?.dist ?? 99), 99) ?? 99;
                        this._exploreSpeedSum = 0;
                        this._exploreTickCount = 0;
                        this._exploreModeAtStart = mode;
                        this._exploreDeathDuringExplore = false;
                        const s = this.policyClient?.stats;
                        const t0 = (obs?.threats || [])[0];
                        console.log(`[Policy] EXPLORE mode=${mode} lat=${s?.avgLatencyMs??0}ms hit=${s?.hitRate??0}% | fwd=${action.forward?1:0} back=${action.back?1:0} atk=${action.attack?1:0} dy=${(action.dyaw||0).toFixed(1)} | mob=${t0?.type||'-'}@${t0?.dist?.toFixed(1)||'-'} hp=${obs.hp} rate=${this._exploreRate.toFixed(2)}`);
                        // PPO: collect rollout step
                        this._collectRolloutStep(policyObs, policyAction, obs);
                    } else {
                        // Fallback: escape planner only (no Q-net blend)
                        const plannerAction = this.escapePlanner.tick(obs, mode, constraints);
                        this._lastQHeadCount = 0;

                        action = {
                            forward: !!plannerAction.forward,
                            back: !!plannerAction.back,
                            left: !!plannerAction.left,
                            right: !!plannerAction.right,
                            jump: !!plannerAction.jump,
                            sprint: !!plannerAction.sprint,
                            attack: !!plannerAction.attack,
                            use: !!plannerAction.use,
                            sneak: !!plannerAction.sneak,
                            dyaw: plannerAction.dyaw,
                            dpitch: plannerAction.dpitch || 0,
                        };

                        this.lastActions = null;
                        this._lastWasQMode = false;
                    }
                } else {
                    // === FIGHT: cancel any active Baritone flee, then policy or Q-blend ===
                    if (this._baritoneFleeActive) this._cancelBaritoneFlee();
                    // Check for policy action FIRST (from previous tick), then send new obs
                    const fightPolicyAction = this.policyClient?.getLatestAction?.(200);
                    const fightPolicyObs = PolicyClient.formatObs(obs, constraints, mode);
                    this.policyClient?.sendObs?.(fightPolicyObs);
                    if (fightPolicyAction && Math.random() < this._exploreRate) {
                        action = PolicyClient.actionToRlAction(fightPolicyAction);
                        this.lastActions = null;
                        this._lastWasQMode = false;
                        this._lastQHeadCount = 0;
                        this._lastPolicyExplore = true;
                        this._exploreHpAtStart = obs.hp || 20;
                        this._exploreDistAtStart = obs.threats?.reduce((m, t) => Math.min(m, t?.dist ?? 99), 99) ?? 99;
                        this._exploreSpeedSum = 0;
                        this._exploreTickCount = 0;
                        this._exploreModeAtStart = mode;
                        this._exploreDeathDuringExplore = false;
                        const t0 = (obs?.threats || [])[0];
                        console.log(`[Policy] EXPLORE fight | fwd=${action.forward?1:0} back=${action.back?1:0} atk=${action.attack?1:0} dy=${(action.dyaw||0).toFixed(1)} | mob=${t0?.type||'-'}@${t0?.dist?.toFixed(1)||'-'} hp=${obs.hp} rate=${this._exploreRate.toFixed(2)}`);
                        this._collectRolloutStep(fightPolicyObs, fightPolicyAction, obs);
                    } else {
                        // Fallback: fight heuristic only (no Q-net blend)
                        const heuristicActions = bootstrapFightActions(obs);
                        const [fb, lr, jump, sprint, attack, yawIdx, pitchIdx] = heuristicActions;
                        this._lastQHeadCount = 0;
                        const threatDist = obs.threats?.reduce((min, t) => Math.min(min, t?.dist ?? 99), 99) ?? 99;
                        // Compute heading error to nearest threat.
                        // dx/dz are PLAYER-RELATIVE (dx=right, dz=front) from Java StateReporter.
                        // atan2(dx, dz) gives angle from forward: 0=ahead, +90=right, -90=left.
                        // This IS the heading error directly — no need to subtract playerYaw.
                        const fightTarget = (obs.threats || []).reduce((c, t) => (!c || (t.dist||99) < (c.dist||99)) ? t : c, null);
                        let fightDyaw = 0;
                        if (fightTarget) {
                            // dx=right(+), dz=front(+) → atan2(dx,dz) = angle from forward
                            const headingError = Math.atan2(fightTarget.dx || 0, fightTarget.dz || 0) * (180 / Math.PI);
                            // EMA smooth to prevent oscillation when mob circles player
                            // Without smoothing: mob goes left→right→left, dyaw oscillates, crosshair never locks on
                            if (!this._fightHeadingEma) this._fightHeadingEma = headingError;
                            this._fightHeadingEma = 0.5 * headingError + 0.5 * this._fightHeadingEma;
                            const smoothed = this._fightHeadingEma;
                            if (threatDist < 4) {
                                fightDyaw = Math.max(-30, Math.min(30, smoothed));
                            } else {
                                // Approach: need faster turn to face mob before sprinting.
                                // Old: ±15 * 0.5 = 7.5°/tick max → 90°/12=12 ticks to face mob
                                // New: ±25 direct → 90°/25=4 ticks to face mob
                                fightDyaw = Math.max(-25, Math.min(25, smoothed * 0.8));
                            }
                        } else {
                            this._fightHeadingEma = null;
                        }
                        // Compute dpitch to aim at mob (pitch=0=horizontal, +down, -up)
                        // MC needs crosshair on entity hitbox to hit — pitch=30° (looking at ground) = miss
                        // Always correct pitch toward mob (not PITCH_VALUES — that pushed pitch to 58°!)
                        let fightDpitch = 0;
                        if (fightTarget) {
                            // dy from threats is world-space (mob.y - player.y), negative = mob below
                            const mobDy = fightTarget.dy || 0;
                            // Target pitch: look toward mob center (mob is ~1 block tall from feet)
                            // atan2(-dy, horizontal_dist) where positive pitch = look down
                            const horizDist = Math.max(0.5, Math.sqrt((fightTarget.dx||0)**2 + (fightTarget.dz||0)**2));
                            const targetPitch = Math.atan2(-(mobDy + 0.9), horizDist) * (180 / Math.PI);
                            // targetPitch: 0 = horizontal, positive = down, negative = up
                            const pitchError = targetPitch - (obs.pitch || 0);
                            // Large cap: pitch=90° from mining needs fast correction (was ±15 = 6 ticks).
                            // Now ±45 with gain 0.8 → 90°→0° in 2-3 ticks.
                            fightDpitch = Math.max(-45, Math.min(45, pitchError * 0.8));
                        }
                        action = {
                            forward: fb === 2, back: fb === 0,
                            left: lr === 0, right: lr === 2,
                            jump: !!jump && !!obs.onGround,
                            sprint: !!sprint, attack: !!attack,
                            dyaw: fightDyaw,
                            dpitch: fightDpitch,
                        };
                        // Fight trace
                        if (this._obsCount % 10 === 0) {
                            const t0 = obs.threats?.[0];
                            console.log(`[FIGHT] fwd=${action.forward?1:0} back=${action.back?1:0} atk=${action.attack?1:0} spr=${action.sprint?1:0} jmp=${action.jump?1:0} dy=${action.dyaw} cd=${(obs.attackCooldown||0).toFixed(1)} | mob=${t0?.type||'-'}@${t0?.dist?.toFixed(1)||'-'} hp=${obs.hp}`);
                        }
                    }
                }

                // Collect combat demo data for PyTorch imitation training
                // Only collect from escape planner / fight heuristic (not policy explore)
                if ((mode === 'flee' || mode === 'fight') && action && !this._lastPolicyExplore) {
                    this._collectCombatDemo(obs, action, constraints, mode);
                } else if (this._obsCount % 200 === 0) {
                    console.log(`[MicroCtrl] SKIP demo: mode=${mode} action=${!!action} policyExplore=${this._lastPolicyExplore} isEscape=${isEscapeMode}`);
                }
                this._lastPolicyExplore = false;

                // Publish proposal to scheduler
                scheduler.publish('micro', {
                    priority: 2,
                    ttlMs: 300,
                    action,
                });

                this.lastObs = obs;
                // For fight mode, store the blended action array for Q learning
                // (flee/hide sets lastActions + _lastWasQMode above)
                if (!isEscapeMode) {
                    if (!this.lastActions) {
                        this._lastWasQMode = true;
                        this.lastActions = [
                            action.forward ? 2 : (action.back ? 0 : 1),
                            action.left ? 0 : (action.right ? 2 : 1),
                            action.jump ? 1 : 0,
                            action.sprint ? 1 : 0,
                            action.attack ? 1 : 0,
                            YAW_VALUES.indexOf(action.dyaw) >= 0 ? YAW_VALUES.indexOf(action.dyaw) : 2,
                            PITCH_VALUES.indexOf(action.dpitch) >= 0 ? PITCH_VALUES.indexOf(action.dpitch) : 1,
                            action.use ? 1 : 0,
                            action.sneak ? 1 : 0,
                        ];
                    }
                }

            } catch (e) {
                // Log errors periodically (not just first 5 obs — was hiding critical bugs!)
                if (this._obsCount < 5 || this._obsCount % 500 === 0) {
                    console.error(`[MicroCtrl] onMsg error (obs=${this._obsCount}):`, e.message, e.stack?.split('\n')[1]);
                }
            }
        };

        this._ensureBridgeHooks();
        this._bridgeWatch = setInterval(() => this._ensureBridgeHooks(), 250);

        // Log stats periodically
        this._statsInterval = setInterval(() => {
            const mode = this.agent.scheduler?.currentMode || 'unknown';
            const alpha = this.tracker.getAlpha(this.qNet.totalUpdates);
            const effAlpha = Math.max(alpha, this.qNet.totalUpdates >= 1000 ? 0.05 : 0);
            const stats = this.tracker.getStats();
            const obsAge = this._lastObsAt ? (Date.now() - this._lastObsAt) : -1;
            console.log(`[MicroCtrl] obs=${this._obsCount||0} mode=${mode} updates=${this.qNet.totalUpdates} alpha=${alpha.toFixed(2)} eff=${effAlpha.toFixed(2)} ruleR=${stats.ruleScore.toFixed(3)} qR=${stats.qScore.toFixed(3)} rN=${stats.ruleN} qN=${stats.qN} vetoes=${stats.vetoes} obsAge=${obsAge}`);
        }, 15000);
    }

    stop() {
        this.running = false;
        const bridge = this._boundBridge || this.agent.bot?._bridge;
        if (this._boundWs && this._onMsg) {
            this._boundWs.removeListener('message', this._onMsg);
        }
        this._boundBridge = null;
        this._boundWs = null;
        this._bridgeObsEnabled = false;
        try { bridge?.sendCommand('rl_observe', { enabled: false }); } catch(e) {}
        if (this._bridgeWatch) clearInterval(this._bridgeWatch);
        if (this._statsInterval) clearInterval(this._statsInterval);
        this.policyClient?.destroy?.();
        console.log('[MicroCtrl] Stopped');
    }

    resetEpisode(reason = 'reset') {
        this.lastObs = null;
        this.lastActions = null;
        this._heldActions = null;
        this._holdMode = null;
        this._holdTicksRemaining = 0;
        this._modeStartTime = 0;
        this._lastMode = 'idle';
        this._lastDyaw = 0;
        this._stallTicks = 0;
        this._lastObsAt = 0;
        this._smoothedDyaw = 0;
        this._loggedFirstActive = false;
        this.escapePlanner?.reset();
        this.policyClient?.resetHidden?.();
        console.log(`[MicroCtrl] episode reset (${reason})`);
    }

    kickObservation(reason = 'manual') {
        const bridge = this.agent.bot?._bridge;
        if (!bridge?.connected) return;
        this._bridgeObsEnabled = false;
        this._lastObserveEnableAt = 0;
        this._ensureBridgeHooks();
        console.log(`[MicroCtrl] kick rl_observe (${reason})`);
    }

    _publishEmergencyEscape(mode, reason = 'stale_obs') {
        const scheduler = this.agent.scheduler;
        const obs = this.lastObs || scheduler?.state;
        if (!scheduler || !obs || (mode !== 'flee' && mode !== 'hide' && mode !== 'fight')) return;
        const now = Date.now();
        if (now - (this._lastEmergencyAt || 0) < 250) return;
        this._lastEmergencyAt = now;

        const heuristic = bootstrapEscapeActions(mode, obs);
        const [fb, lr, jump, sprint, attack, yawIdx, pitchIdx] = heuristic;
        const speed = Math.sqrt((obs.vx || 0) ** 2 + (obs.vz || 0) ** 2);
        const threatDist = obs.threats?.reduce((min, t) => Math.min(min, t?.dist ?? 99), 99) ?? 99;
        scheduler.publish('micro', {
            priority: 3,
            ttlMs: 350,
            action: {
                forward: fb === 2,
                back: fb === 0,
                left: lr === 0,
                right: lr === 2,
                jump: !!jump && !!obs.onGround && speed >= 0.05 && threatDist < 2.5,
                sprint: !!sprint,
                attack: mode === 'fight' ? !!attack : false,
                dyaw: YAW_VALUES[yawIdx],
                dpitch: mode === 'fight' ? PITCH_VALUES[pitchIdx] : 0,
            },
        });
        console.log(`[MicroCtrl] emergency ${mode} proposal (${reason})`);
    }

    _ensureBridgeHooks() {
        if (!this.running) return;
        const bridge = this.agent.bot?._bridge;
        if (!bridge) return;

        if (bridge !== this._boundBridge || bridge.ws !== this._boundWs) {
            if (this._boundWs && this._onMsg) {
                this._boundWs.removeListener('message', this._onMsg);
            }
            this._boundBridge = bridge;
            this._boundWs = bridge.ws || null;
            this._bridgeObsEnabled = false;
            if (this._boundWs && this._onMsg) {
                this._boundWs.on('message', this._onMsg);
            }
        }

        const activeMode = this.agent.scheduler?.currentMode;
        const activeSurvivalMode = activeMode === 'flee' || activeMode === 'hide' || activeMode === 'fight';
        const obsAge = this._lastObsAt ? (Date.now() - this._lastObsAt) : Infinity;
        const needsObserveRefresh =
            bridge.connected &&
            (!this._bridgeObsEnabled || obsAge > 1000);

        if (needsObserveRefresh && Date.now() - this._lastObserveEnableAt > 250) {
            this._lastObserveEnableAt = Date.now();
            bridge.sendCommand('rl_observe', { enabled: true })
                .then(() => {
                    this._bridgeObsEnabled = true;
                    if (obsAge > 1000) {
                        console.log(`[MicroCtrl] rl_observe re-enabled after stale obs (${Math.round(obsAge)}ms)`);
                    }
                })
                .catch(() => {});
        }

        if (activeSurvivalMode && obsAge > 350) {
            this._publishEmergencyEscape(activeMode, 'obs_timeout');
        }
    }

    _reward(mode, prev, obs, constraints) {
        let r = 0.005;
        const hpD = (obs.hp||0) - (prev.hp||0);
        const pt = prev.threats || [], ct = obs.threats || [];
        const ctNearestDist = ct.reduce((m, t) => Math.min(m, t?.dist ?? 99), 99);
        const ptNearestDist = pt.reduce((m, t) => Math.min(m, t?.dist ?? 99), 99);

        // === HARD DAMAGE PENALTY — the most important signal ===
        if (hpD < 0) {
            r -= 0.3 * Math.abs(hpD); // strong: every half heart lost hurts
            // Extra: took damage while mob is very close = terrible
            if (ct.length > 0 && ctNearestDist < 3) r -= 0.5;
        }
        if (hpD > 0) r += 0.05 * hpD;

        // === DEATH ===
        if ((obs.hp||20) <= 0) r -= 5.0;

        // === INACTION PENALTY: mob close + not moving = bad ===
        if (ct.length > 0 && ctNearestDist < 4) {
            const speed = Math.sqrt((obs.vx||0)**2 + (obs.vz||0)**2);
            if (speed < 0.05) r -= 0.2; // standing still near mob
        }

        // === Mode-specific ===
        if (mode === 'flee' || mode === 'hide') {
            if (pt.length > 0 && ct.length > 0) {
                const distDelta = ctNearestDist - ptNearestDist;
                r += distDelta * 0.2; // reward distance gained
                if (distDelta < 0) r -= 0.3 * Math.abs(distDelta); // strong: every block closer is bad
            }
            if (prev.los && !obs.los) r += 0.5; // broke line of sight
            if (ct.length === 0 && pt.length > 0) r += 2.0; // fully escaped
            // Strong penalty for entering water — Q should learn to avoid it
            if (!prev.inWater && obs.inWater) r -= 1.5;
            if (obs.inWater && pt.length > 0 && ct.length > 0) {
                const distDelta = ctNearestDist - ptNearestDist;
                if (distDelta >= 0) r -= 0.2; // in water but not gaining distance = bad
            }
        } else if (mode === 'fight') {
            // Compare nearest mob's HP: find the mob that's closest in both frames
            const ctNearest = ct.reduce((c, t) => (!c || (t.dist||99) < (c.dist||99)) ? t : c, null);
            const ptNearest = pt.reduce((c, t) => (!c || (t.dist||99) < (c.dist||99)) ? t : c, null);
            if (ctNearest && ptNearest && (ctNearest.hp||20) < (ptNearest.hp||20)) r += 0.5;
            if (ct.length < pt.length) r += 2.0;
            // Reward attacking when cooldown is ready
            if (obs.attackCooldown >= 0.9) r += 0.05; // waiting for full cooldown is good
        }

        // Goal-conditioning: penalize drifting too far from anchor
        if (constraints?.anchor && constraints._anchorDist) {
            const maxR = constraints.maxRetreatRadius || 24;
            const dist = constraints._anchorDist;
            if (dist > maxR) {
                r -= 0.05 * (dist - maxR); // growing penalty beyond allowed radius
            }
            if (dist > maxR * 1.5) {
                r -= 0.2; // hard penalty for way too far
            }
        }

        // === Navigation reward: progress toward Baritone goal ===
        if (obs.baritone?.pathing && prev.baritone?.pathing) {
            const prevGoalDist = prev.baritone.goalDist || 0;
            const curGoalDist = obs.baritone.goalDist || 0;
            if (prevGoalDist > 0 && curGoalDist > 0) {
                const progress = prevGoalDist - curGoalDist;
                r += progress * 0.1; // +0.1 per block closer to goal
            }
            const speed = Math.sqrt((obs.vx||0)**2 + (obs.vz||0)**2);
            if (speed < 0.03) r -= 0.02; // stuck while navigating
        }
        // Arrival bonus: was pathing, now arrived (goalDist dropped below 3)
        if (prev.baritone?.pathing && !obs.baritone?.pathing &&
            (prev.baritone.goalDist || 99) < 5) {
            r += 0.5;
        }

        return r;
    }
}
