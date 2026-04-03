/**
 * Real-Time Scheduler: proposal-based, many producers, single actuator.
 *
 * Every tick:
 *   1. All layers publish proposals (reflex, micro, tactical, planner)
 *   2. Scheduler picks the best valid proposal
 *   3. Single rl_action sent to game
 *
 * Layers don't block each other. They run at their own frequency.
 */

import fs from 'fs';
import path from 'path';

export class RTScheduler {
    constructor(agent) {
        this.agent = agent;
        this.proposals = new Map(); // producer -> latest proposal
        this.currentMode = 'idle';  // flee/fight/hide/eat/continue/idle
        this.constraints = {};       // from planner: anchor, maxRetreatRadius, etc.
        this.lastAction = null;
        this.tickCount = 0;

        // State bus: latest state snapshot, updated every tick
        this.state = null;

        // Hysteresis: don't flip modes every tick
        this._modeSwitchCooldown = 0;
        this._lastModeSwitch = 0;

        // Override events: tactical → planner communication
        this.activeOverride = null; // { mode, reason, since, ttlMs, resumeHint, source }
        this.plannerInhibited = false;

        // Post-flee hold: set by tactical layer to tell scheduler "don't drop flee yet"
        this.postFleeUntil = 0;
        this._lastNearestDist = 99; // for computing closing rate at timeout check
    }

    /**
     * Tactical layer declares an override — inhibits planner until resolved.
     */
    declareOverride(mode, reason, ttlMs = 10000, resumeHint = 'return_anchor', source = 'tactical') {
        const isRefresh =
            this.activeOverride &&
            this.activeOverride.mode === mode &&
            this.activeOverride.source === source;
        if (isRefresh) {
            this.activeOverride.reason = reason;
            this.activeOverride.since = Date.now();
            this.activeOverride.ttlMs = ttlMs;
            this.activeOverride.resumeHint = resumeHint;
            this.plannerInhibited = true;
            return;
        }
        this.activeOverride = {
            mode,
            reason,
            since: Date.now(),
            ttlMs,
            resumeHint,
            source,
        };
        this.plannerInhibited = true;
        // Any flee override (reflex or tactical) sets post-flee hold
        // so the bot doesn't immediately run back toward the threat.
        if (mode === 'flee') {
            this.postFleeUntil = Date.now() + ttlMs + 5000; // hold 5s after override TTL
        }
        console.log(`[Scheduler] OVERRIDE: ${mode} source=${source} reason=${reason} ttl=${ttlMs}ms resume=${resumeHint}`);
    }

    /**
     * Tactical layer resolves the override — planner can resume.
     */
    resolveOverride(outcome = 'safe') {
        if (this.activeOverride) {
            console.log(`[Scheduler] Override resolved: ${this.activeOverride.mode} → ${outcome}`);
        }
        this.activeOverride = null;
        this.plannerInhibited = false;
    }

    /**
     * Check if override has expired.
     */
    checkOverrideTimeout() {
        if (this.activeOverride) {
            const elapsed = Date.now() - this.activeOverride.since;
            if (elapsed > this.activeOverride.ttlMs) {
                // Flee override extension gates:
                // 1) tactical post-flee hold (distance/approach aware)
                // 2) critical low HP while threats are still nearby
                if (this.activeOverride.mode === 'flee') {
                    const threats = this.state?.threats || [];
                    const nearestDist = threats.reduce((min, t) => Math.min(min, t?.dist ?? 99), 99);
                    const hp = this.state?.hp ?? 20;
                    const postFleeHoldActive = !!(this.postFleeUntil && Date.now() < this.postFleeUntil);
                    const shouldExtendFromPostFlee = postFleeHoldActive && nearestDist < 20;
                    const criticalLowHpHold = hp <= 6 && nearestDist < 16;
                    if (shouldExtendFromPostFlee || criticalLowHpHold) {
                        // Melee range: always extend. Mid-range: only if approaching.
                        // Critical HP: keep fleeing while threats remain nearby.
                        this.activeOverride.since = Date.now();
                        this.activeOverride.ttlMs = 2000;
                        return;
                    }
                }
                console.log(`[Scheduler] Override expired after ${elapsed}ms`);
                if (this.currentMode === this.activeOverride.mode) {
                    this.setMode('continue', 'override_timeout');
                }
                this.resolveOverride('timeout');
            }
        }
    }

    /**
     * Get structured event for planner to read.
     */
    getPlannerEvent() {
        if (!this.activeOverride) return null;
        return {
            type: 'tactical_override',
            mode: this.activeOverride.mode,
            reason: this.activeOverride.reason,
            durationMs: Date.now() - this.activeOverride.since,
            resumeHint: this.activeOverride.resumeHint,
            source: this.activeOverride.source,
        };
    }

    /**
     * Publish a proposal from any layer.
     */
    publish(producer, proposal) {
        this.proposals.set(producer, {
            ...proposal,
            producer,
            timestamp: Date.now(),
        });
    }

    clearProposal(producer) {
        this.proposals.delete(producer);
    }

    reset(reason = 'reset') {
        this.proposals.clear();
        this.currentMode = 'idle';
        this.constraints = {};
        this.lastAction = null;
        this.state = null;
        this.activeOverride = null;
        this.plannerInhibited = false;
        this.postFleeUntil = 0;
        this._lastNearestDist = 99;
        this._lastModeSwitch = 0;
        console.log(`[Scheduler] reset (${reason})`);
    }

    /**
     * Update current mode (from tactical layer).
     */
    setMode(mode, source = 'tactical') {
        const now = Date.now();
        // Block tactical from overriding reflex override
        if (mode === 'continue' && source === 'tactical' &&
            this.activeOverride && this.activeOverride.source === 'reflex' &&
            this.currentMode !== 'continue') {
            return;
        }
        // Minimum mode duration: survival modes must persist at least 2s before dropping to continue
        const MIN_SURVIVAL_DURATION = 2000;
        const SURVIVAL_MODES = ['flee', 'fight', 'hide'];
        if (mode === 'continue' && SURVIVAL_MODES.includes(this.currentMode)) {
            const elapsed = now - this._lastModeSwitch;
            if (elapsed < MIN_SURVIVAL_DURATION) return; // too soon to drop survival mode
        }
        if (mode !== this.currentMode && now - this._lastModeSwitch > 500) {
            console.log(`[Scheduler] mode: ${this.currentMode} -> ${mode} (from ${source})`);
            this.currentMode = mode;
            this._lastModeSwitch = now;
        }
    }

    /**
     * Update constraints (from planner).
     */
    setConstraints(constraints) {
        this.constraints = { ...this.constraints, ...constraints };
    }

    /**
     * Update state bus (called every tick from Java rl_obs).
     */
    updateState(obs) {
        // Track nearest threat distance for closing rate at timeout check
        if (this.state) {
            const prevThreats = this.state.threats || [];
            this._lastNearestDist = prevThreats.reduce((min, t) => Math.min(min, t?.dist ?? 99), 99);
        }
        this.state = obs;
        try {
            const controlState = this.agent?.control?.getState?.() || {};
            this.agent?.skillDemoLogger?.onObservation?.(obs, {
                owner: controlState.owner || 'idle',
                ownerReason: controlState.reason || null,
                schedulerMode: this.currentMode,
            });
        } catch (e) {}
    }

    /**
     * Pick the best proposal and return the action to send.
     * Called every tick by the main loop.
     */
    resolve() {
        this.tickCount++;
        const now = Date.now();
        const SURVIVAL_MODES = ['flee', 'fight', 'hide'];
        const HOLD_LAST_ACTION_MS = 450;

        // Collect valid proposals (not expired)
        const valid = [];
        for (const [producer, p] of this.proposals) {
            const age = now - p.timestamp;
            const ttl = p.ttlMs || 200; // default 200ms TTL
            if (age > ttl) {
                this.proposals.delete(producer);
                continue;
            }
            valid.push(p);
        }

        if (valid.length === 0) {
            if (
                SURVIVAL_MODES.includes(this.currentMode) &&
                this.lastAction?.action &&
                now - (this.lastAction.timestamp || 0) <= HOLD_LAST_ACTION_MS
            ) {
                return this.lastAction.action;
            }
            return null;
        }

        // Sort: higher priority first, then fresher, then higher confidence
        valid.sort((a, b) => {
            if ((b.priority || 0) !== (a.priority || 0)) return (b.priority || 0) - (a.priority || 0);
            return (b.timestamp || 0) - (a.timestamp || 0);
        });

        const best = valid[0];
        this.lastAction = best;
        return best.action || null;
    }

    getState() {
        return {
            mode: this.currentMode,
            proposals: Array.from(this.proposals.keys()),
            constraints: this.constraints,
            tickCount: this.tickCount,
        };
    }
}
